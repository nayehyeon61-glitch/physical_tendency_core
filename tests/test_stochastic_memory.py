"""Probability identities and stateful integration invariants, without ERA5/KAI."""

import pytest
import torch
from torch import nn

from pgcfm.stochastic_memory import (
    StochasticMemoryCore, ensemble_crps, gaussian_entropy, gaussian_kl, gaussian_nll,
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
