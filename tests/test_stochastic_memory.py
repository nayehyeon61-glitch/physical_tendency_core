"""Probability identities and stateful integration invariants, without ERA5/KAI."""

import pytest
import torch
from torch import nn

from pgcfm.stochastic_memory import (
    EntropyConditionedMemoryCore, StochasticMemoryCore, ensemble_crps,
    gaussian_entropy, gaussian_kl, gaussian_nll,
)


class TinyMemoryCore(nn.Module):
    """A causal, projected memory solver fixture; no optional project imports."""

    def __init__(self):
        super().__init__()
        self.net = nn.Module()
        self.net.trunk = nn.Sequential(nn.Conv2d(6, 4, 1), nn.Tanh())
        self.net.head = nn.Conv2d(4, 3, 1)
        self.register_buffer("t6", torch.full((1, 3, 1, 1), 0.2))
        self.register_buffer("keep", torch.tensor([1., 1., 0.]).view(1, 3, 1, 1))
        self.stride = 1
        self.reset()

    def reset(self, history=None, hours0=None, x0=None):
        self.m, self.fc, self.innovations = None, None, []
        if history is not None:
            self.m = history.mean(1) * 0.2

    def _trunk(self, x, hours):
        return self.net.trunk(torch.cat([x, self.m], 1))

    def finish(self, x, forcing):
        # A non-invertible analogue of PositiveMass plus an exact forced channel.
        first = x[:, :1].clamp_min(0.)
        last = x[:, 2:] if forcing is None else forcing[:, None]
        return torch.cat([first, x[:, 1:2], last], 1)

    def step(self, x, hours, tisr_next):
        innovation = torch.zeros_like(x) if self.fc is None else x - self.fc
        self.innovations.append(innovation.detach().clone())
        self.m = 0.2 * x if self.m is None else 0.8 * self.m + 0.2 * x
        self.fc = self.finish(x + self.net.head(self._trunk(x, hours)) * self.t6 * self.keep, tisr_next)
        return self.fc


def fixture():
    torch.manual_seed(11)
    model = StochasticMemoryCore(TinyMemoryCore()).eval()
    x = torch.full((2, 3, 8, 16), 0.1)
    history = x[:, None].repeat(1, 3, 1, 1, 1)
    return model, x, history


def test_gaussian_helpers_match_torch_and_broadcast():
    mu = torch.tensor([[0., 1.], [2., 3.]], dtype=torch.float64, requires_grad=True)
    sigma = torch.tensor([0.3, 1.2], dtype=torch.float64, requires_grad=True)
    ref = torch.tensor(0.7, dtype=torch.float64)
    target = torch.tensor(0.2, dtype=torch.float64)
    p = torch.distributions.Normal(mu, sigma)
    q = torch.distributions.Normal(mu, ref)
    assert torch.allclose(gaussian_nll(target, mu, sigma), -p.log_prob(target))
    assert torch.allclose(gaussian_entropy(sigma), torch.distributions.Normal(0., sigma).entropy())
    assert torch.allclose(gaussian_kl(mu, sigma, ref), torch.distributions.kl_divergence(p, q))
    assert gaussian_kl(mu, sigma, sigma).shape == mu.shape
    loss = gaussian_nll(target, mu, sigma).mean() + gaussian_kl(mu, sigma, ref).mean()
    loss.backward()
    assert torch.isfinite(mu.grad).all() and torch.isfinite(sigma.grad).all()


@pytest.mark.parametrize("invalid", [0., -1., float("nan"), float("inf")])
def test_scales_cannot_be_invalid(invalid):
    with pytest.raises(ValueError):
        gaussian_nll(torch.tensor(0.), torch.tensor(0.), invalid)
    with pytest.raises(ValueError):
        gaussian_entropy(invalid)
    with pytest.raises(ValueError):
        gaussian_kl(torch.tensor(0.), 1., invalid)


def test_small_positive_scales_are_floored_and_finite():
    value = gaussian_nll(torch.tensor(1.), torch.tensor(0.), 1e-30)
    assert torch.isfinite(value)
    assert torch.equal(value, gaussian_nll(torch.tensor(1.), torch.tensor(0.), 1e-6))
    assert gaussian_entropy(torch.tensor(.01)).item() < 0  # differential entropy may be negative


@pytest.mark.parametrize("fair", [True, False])
def test_sorted_crps_matches_pairwise_formula_after_projection(fair):
    torch.manual_seed(4)
    samples = torch.randn(2, 5, 3).clamp_min(0.).requires_grad_()
    target = torch.rand(2, 3)
    first = (samples - target[:, None]).abs().mean(1)
    pair = (samples[:, :, None] - samples[:, None]).abs().sum((1, 2))
    denominator = 5 * (4 if fair else 5)
    expected = first - .5 * pair / denominator
    actual = ensemble_crps(samples, target, member_dim=1, fair=fair)
    assert torch.allclose(actual, expected, atol=1e-7)
    actual.mean().backward()
    assert torch.isfinite(samples.grad).all()
    assert torch.equal(ensemble_crps(samples[:, :1], target, member_dim=1),
                       (samples[:, 0] - target).abs())


def test_deterministic_mode_matches_original_solver_and_reset():
    model, x, history = fixture()
    hours = torch.zeros(len(x))
    forcing = torch.full((2, 3, 8, 16), 0.35)
    model.core.reset(history, hours, x)
    baseline, state = [], x
    for k in range(3):
        state = model.core.step(state, hours + 24 * k, forcing[:, k])
        baseline.append(state)
    expected = torch.stack(baseline, 1)
    actual = model.rollout(x, hours, 3, forcing, history, stochastic=False)
    assert torch.equal(actual, expected)
    assert torch.equal(model.rollout(x, hours, 3, forcing, history, stochastic=False), expected)


def test_seeded_members_are_independent_causal_and_physically_projected():
    model, x, history = fixture()
    forcing = torch.full((2, 4, 8, 16), .7)
    hours = torch.zeros(len(x))
    y = model.rollout(x, hours, 4, forcing, history, seed=9)
    innovations = [v.clone() for v in model.core.innovations]
    assert torch.equal(y, model.rollout(x, hours, 4, forcing, history, seed=9))
    assert not torch.equal(y, model.rollout(x, hours, 4, forcing, history, seed=10))
    assert not torch.equal(y[0, :, :2], y[1, :, :2])
    assert torch.equal(y[:, :, 2], forcing)
    assert bool((y[:, :, 0] >= 0.).all())
    assert all(torch.equal(e, torch.zeros_like(e)) for e in innovations)


def test_distribution_loss_backpropagates_to_scale_and_existing_core():
    model, x, history = fixture()
    model.reset(history, torch.zeros(2), x)
    sample, location, scale = model.step(x, torch.zeros(2), return_distribution=True,
                                       generator=torch.Generator().manual_seed(8))
    # Channel 2 is deterministic known forcing, excluded from Gaussian scores.
    loss = gaussian_nll(x[:, :2] + .05, location[:, :2], scale[:, :2]).mean()
    loss.backward()
    assert torch.equal(model.core.fc, sample)
    assert torch.isfinite(model.noise_head.weight.grad).all()
    assert model.noise_head.weight.grad.abs().sum() > 0
    assert model.core.net.head.weight.grad.abs().sum() > 0
    assert model.core.net.trunk[0].weight.grad.abs().sum() > 0
    assert torch.equal(scale[:, 2], torch.zeros_like(scale[:, 2]))


def test_tiny_increment_scales_sample_the_same_floored_gaussian_that_is_scored():
    model, x, history = fixture()
    model.core.t6.fill_(1e-9)
    hours = torch.zeros(len(x))
    model.reset(history, hours, x)
    sample, location, scale = model.step(x, hours, return_distribution=True,
                                       generator=torch.Generator().manual_seed(19))
    assert torch.equal(scale[:, :2], torch.full_like(scale[:, :2], 1e-6))
    assert torch.equal(scale[:, 2], torch.zeros_like(scale[:, 2]))
    noise = torch.randn(x.shape, generator=torch.Generator().manual_seed(19))
    assert torch.equal(sample, model.core.finish(location + scale * noise, None))
    target = location[:, :2] + 2e-6
    expected = -torch.distributions.Normal(location[:, :2], scale[:, :2]).log_prob(target)
    assert torch.allclose(gaussian_nll(target, location[:, :2], scale[:, :2]), expected)


def test_rollout_rejects_ambiguous_random_state_and_empty_trajectory():
    model, x, _ = fixture()
    with pytest.raises(ValueError):
        model.rollout(x, torch.zeros(2), 0)
    with pytest.raises(ValueError):
        model.rollout(x, torch.zeros(2), 1, seed=1, generator=torch.Generator())


class TinyCausalMemoryCore(TinyMemoryCore):
    """The existing MemCore's split remember/_advance and teacher-forcing API."""

    mem, mem_ch, mz = 3, 3, True

    def reset(self, history=None, hours0=None, x0=None):
        self.m, self.fc, self.seq, self.started = None, None, None, False
        self.innovations = []
        if history is not None:
            forecasts = []
            for k in range(history.shape[1]):
                nxt = history[:, k + 1] if k + 1 < history.shape[1] else x0
                forcing = None if nxt is None else nxt[:, 2]
                self.remember(history[:, k])
                self.fc = self._advance(history[:, k], hours0 - (history.shape[1] - k) * 24, forcing)
                forecasts.append(self.fc)
            self.seq = torch.stack(forecasts, 1)

    def remember(self, x):
        innovation = torch.zeros_like(x) if self.fc is None else x - self.fc
        self.innovations.append(innovation.detach().clone())
        self.m = 0.2 * x if self.m is None else 0.8 * self.m + 0.2 * x + 0.05 * innovation

    def _advance(self, x, hours, forcing):
        return self.finish(x + self.net.head(self._trunk(x, hours)) * self.t6 * self.keep, forcing)

    def step(self, x, hours, tisr_next):
        if not (getattr(self, "freeze_mem", False) and self.started):
            self.remember(x)
        self.started = True
        self.fc = self._advance(x, hours, tisr_next)
        return self.fc


def entropy_fixture(enabled=True, reference=0.2):
    torch.manual_seed(15)
    model = EntropyConditionedMemoryCore(TinyCausalMemoryCore(),
                                        reference_relative_scale=reference,
                                        entropy_enabled=enabled).eval()
    x = torch.full((2, 3, 8, 16), .1)
    history = torch.stack([x * .6, x * .8, x], 1)
    return model, x, history


def test_entropy_feature_is_centered_negative_gaussian_entropy_not_kl():
    model, _, _ = entropy_fixture(reference=torch.tensor([.2, .3, .4]))
    relative = torch.tensor([.1, .6, .7]).view(1, 3, 1, 1).expand(2, 3, 8, 16)
    expected = -torch.log(relative / model.reference_relative_scale) * model.core.keep
    assert torch.allclose(model.entropy_feature(relative), expected)
    h_difference = (gaussian_entropy(model.reference_relative_scale) - gaussian_entropy(relative)) * model.core.keep
    assert torch.allclose(model.entropy_feature(relative), h_difference, atol=1e-7)
    assert model.entropy_feature(relative)[:, 0].mean() > 0
    assert model.entropy_feature(relative)[:, 1].mean() < 0  # KL could not be negative
    assert torch.equal(model.entropy_feature(relative)[:, 2], torch.zeros_like(relative[:, 2]))


def test_entropy_zero_projection_matches_original_solver_and_warmup_exactly():
    model, x, history = entropy_fixture()
    hours = torch.zeros(len(x))
    forcing = torch.full((2, 3, 8, 16), .4)
    model.core.reset(history, hours, x)
    expected_seq = model.core.seq.clone()
    expected, state = [], x
    for k in range(3):
        state = model.core.step(state, hours + 24 * k, forcing[:, k])
        expected.append(state)
    actual = model.rollout(x, hours, 3, forcing, history, stochastic=False)
    assert torch.equal(model.seq, expected_seq)
    assert torch.equal(actual, torch.stack(expected, 1))


def test_entropy_off_has_identical_parameters_and_controls_only_the_input():
    on, x, history = entropy_fixture(True)
    off, _, _ = entropy_fixture(False)
    assert [(n, tuple(p.shape)) for n, p in on.named_parameters()] == [
        (n, tuple(p.shape)) for n, p in off.named_parameters()]
    with torch.no_grad():
        on.entropy_to_memory.weight.fill_(.2)
    off.load_state_dict(on.state_dict())
    y_on = on.rollout(x, torch.zeros(2), 2, history=history, stochastic=False)
    y_off = off.rollout(x, torch.zeros(2), 2, history=history, stochastic=False)
    assert not torch.equal(y_on[:, :, :2], y_off[:, :, :2])
    assert torch.equal(off.last_entropy_input, torch.zeros_like(off.last_entropy_input))


def test_entropy_changes_memory_and_drift_and_has_finite_gradients():
    model, x, history = entropy_fixture()
    with torch.no_grad():
        model.entropy_to_memory.weight.fill_(.15)
    model.reset(history, torch.zeros(2), x)
    sample, location, scale = model.step(x, torch.zeros(2), return_distribution=True,
                                       generator=torch.Generator().manual_seed(8))
    assert model.last_entropy_input.abs().sum() > 0
    loss = gaussian_nll(x[:, :2] + .03, location[:, :2], scale[:, :2]).mean()
    loss.backward()
    for parameter in [model.noise_head.weight, model.entropy_to_memory.weight,
                      model.core.net.head.weight, model.core.net.trunk[0].weight]:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert torch.equal(model.core.fc, sample)


def test_entropy_scale_is_computed_before_its_own_memory_injection():
    model, x, _ = entropy_fixture()
    with torch.no_grad():
        model.noise_head.weight.fill_(.04)
    model.reset()
    _, first_location, first_scale = model.step(x, torch.zeros(2), stochastic=False, return_distribution=True)
    first_entropy = model.last_entropy_input.clone()
    with torch.no_grad():
        model.entropy_to_memory.weight.fill_(.7)
    model.reset()
    _, second_location, second_scale = model.step(x, torch.zeros(2), stochastic=False, return_distribution=True)
    assert torch.equal(first_scale, second_scale)
    assert torch.equal(first_entropy, model.last_entropy_input)
    assert not torch.equal(first_location, second_location)


def test_entropy_history_is_causal_and_reset_reproduces_seeded_members():
    model, x, history = entropy_fixture()
    with torch.no_grad():
        model.entropy_to_memory.weight.fill_(.1)
    hours = torch.zeros(2)
    model.reset(history, hours, x)
    warm_seq, warm_mem = model.seq.clone(), model.core.m.clone()
    different_weather_same_forcing = x.clone()
    different_weather_same_forcing[:, :2] += 100
    model.reset(history, hours, different_weather_same_forcing)
    assert torch.equal(model.seq, warm_seq) and torch.equal(model.core.m, warm_mem)
    assert model.seq.shape == history.shape
    forcing = torch.full((2, 3, 8, 16), .5)
    a = model.rollout(x, hours, 3, forcing, history, seed=2)
    innovations = model.core.innovations[-3:]
    assert all(torch.equal(e, torch.zeros_like(e)) for e in innovations[1:])
    assert torch.equal(a, model.rollout(x, hours, 3, forcing, history, seed=2))
    assert not torch.equal(a[0, :, :2], a[1, :, :2])
    assert torch.equal(a[:, :, 2], forcing)


def test_entropy_tiny_scale_uses_actual_increment_density_and_masks_forcing():
    model, x, history = entropy_fixture(reference=.2)
    model.core.t6.fill_(1e-9)
    model.reset(history, torch.zeros(2), x)
    _, _, scale = model.step(x, torch.zeros(2), return_distribution=True)
    assert torch.equal(scale[:, :2], torch.full_like(scale[:, :2], 1e-6))
    expected = -torch.log((scale[:, :2] / model.core.t6[:, :2]) / .2)
    assert torch.allclose(model.last_entropy_input[:, :2], expected)
    assert torch.equal(model.last_entropy_input[:, 2], torch.zeros_like(model.last_entropy_input[:, 2]))


def test_entropy_reference_and_replay_arguments_are_validated():
    with pytest.raises(ValueError):
        EntropyConditionedMemoryCore(TinyCausalMemoryCore(), reference_relative_scale=0.)
    with pytest.raises(ValueError):
        EntropyConditionedMemoryCore(TinyCausalMemoryCore(), reference_relative_scale=torch.ones(2, 3, 1, 1))
    model, x, history = entropy_fixture()
    with pytest.raises(ValueError):
        model.reset(history, x0=x)
    with pytest.raises(ValueError):
        model.entropy_feature(torch.zeros_like(x))


def test_optional_real_memory_core_on_small_grid():
    # Dependencies are optional for this module's tests; a configured full
    # research environment additionally exercises the unchanged real core.
    for package in ["numpy", "torchcfm", "netCDF4", "torchdiffeq", "torchcubicspline",
                    "pandas", "xarray", "matplotlib", "properscoring", "ruamel.yaml"]:
        pytest.importorskip(package)
    from pgcfm import contract
    from pgcfm.kai_tend import CachedFeatures, core_rollout, make_core, operator_features
    import numpy as np

    lat = np.linspace(-78.75, 78.75, 8)
    contract.register_latitudes(lat)
    mean, std = contract.REFERENCE_MEAN.copy(), contract.REFERENCE_STD.copy()
    xfull = torch.zeros(1, 67, 8, 16)
    x = xfull[:, contract.DYNAMIC_CHANNELS]
    core = make_core(torch.zeros(8, 8, 16), np.full(67, .1), mean, std, lat,
                     hidden=8, blocks=(1,), norm="gn", mem=3, mem_ch=4, mz=True, gfeat=True)
    features = operator_features(mean, std, xfull, states=x)
    core.gscale.copy_(features.scale)
    core.features = CachedFeatures(features)
    core.eval()
    history = x[:, None].repeat(1, 3, 1, 1, 1)
    forcing = x[:, contract.DYNAMIC_CHANNELS.index(contract.CHANNEL_INDEX["tisr"])][:, None].repeat(1, 2, 1, 1)
    baseline = core_rollout(core, x, torch.zeros(1), 2, forcing, prev=history)
    wrapped = StochasticMemoryCore(core).eval()
    actual = wrapped.rollout(x, torch.zeros(1), 2, forcing, history, stochastic=False)
    assert torch.equal(actual, baseline)
    sample = wrapped.rollout(x, torch.zeros(1), 2, forcing, history, seed=1)
    assert sample.shape == (1, 2, 66, 8, 16) and torch.isfinite(sample).all()
    # A zero-initialized direct entropy projection leaves this real memory
    # core's deterministic warm-up and dynamics unchanged as well.
    baseline = core_rollout(core, x, torch.zeros(1), 2, forcing, prev=history)
    entropy_wrapped = EntropyConditionedMemoryCore(core, reference_relative_scale=.2).eval()
    actual = entropy_wrapped.rollout(x, torch.zeros(1), 2, forcing, history, stochastic=False)
    assert torch.equal(actual, baseline)
