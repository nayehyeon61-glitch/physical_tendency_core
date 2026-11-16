"""Probability, causality and split invariants of the controlled CPU study."""

import numpy as np
import pytest
import torch

from pgcfm.memory_entropy_toy import (ENTROPY_INPUT_ARMS, EntropyInputClosure, PhysicalClosure, ToyConfig,
                                     centered_negative_entropy, evaluate_model, generate_trajectories,
                                     make_closure, physical_tendency, split_data, train_model, train_reference_scale)
from pgcfm.stochastic_memory import gaussian_entropy


def test_generator_reproducible_and_independent_splits():
    config = ToyConfig()
    data = split_data(config, n_train=8, n_validation=8, n_test=8, length=48)
    assert np.array_equal(data["train"], generate_trajectories(config, 8, 48, 1729))
    assert not np.array_equal(data["train"], data["validation"])
    assert not np.array_equal(data["validation"], data["test"])
    assert np.isfinite(data["train"]).all()


def test_noise_free_generator_is_seed_reproducible_and_finite():
    x = generate_trajectories(ToyConfig(noise_std=0, burn_in=0), 8, 80, 3)
    assert np.isfinite(x).all()
    assert np.mean(x[:, -10:] ** 2) < np.mean(x[:, :10] ** 2)
    with pytest.raises(ValueError):
        ToyConfig(dt=-1)


def test_no_memory_no_noise_control_is_exact_known_physical_step():
    config = ToyConfig(memory_strength=0, noise_std=0, burn_in=0)
    x = generate_trajectories(config, 4, 40, 2)
    np.testing.assert_allclose(x[:, 1:], x[:, :-1] + config.dt * physical_tendency(x[:, :-1]), atol=1e-7)
    # The hidden memory time cannot affect observations when its coefficient is zero.
    other = generate_trajectories(ToyConfig(memory_strength=0, noise_std=0, burn_in=0,
                                          memory_time=3), 4, 40, 2)
    np.testing.assert_array_equal(x, other)


def test_markov_control_has_independent_force_but_ou_control_retains_color():
    def residual_correlation(white):
        config = ToyConfig(memory_strength=0, white_forcing=white)
        x = generate_trajectories(config, 128, 160, 76)
        r = (x[:, 1:] - x[:, :-1]) / config.dt - physical_tendency(x[:, :-1])
        corr = np.corrcoef(r[:, 1:].ravel(), r[:, :-1].ravel())[0, 1]
        return corr, float(r.std())
    white_corr, white_std = residual_correlation(True)
    ou_corr, _ = residual_correlation(False)
    assert abs(white_corr) < .04
    assert abs(white_std - .6) < .03
    assert abs(ou_corr - np.exp(-.1 / .4)) < .04


def test_conditional_colored_law_is_causal_and_matches_ar1_formula():
    torch.manual_seed(4)
    model = PhysicalClosure("memory_colored", .1, .05)
    model.rho_raw.data.fill_(.7)
    x = torch.randn(3, 10) * .2
    mean, scale, _ = model.parameters_at(x[:, :-1])
    cm, cs = model.observed_law(x)
    previous_z = (x[:, 1:-1] - mean[:, :-1]) / scale[:, :-1]
    torch.testing.assert_close(cm[:, 1:], mean[:, 1:] + model.rho * scale[:, 1:] * previous_z)
    torch.testing.assert_close(cs[:, 1:], scale[:, 1:] * torch.sqrt(1 - model.rho ** 2))
    changed = x.clone()
    changed[:, 7:] += 5
    cm_changed, cs_changed = model.observed_law(changed)
    torch.testing.assert_close(cm[:, :6], cm_changed[:, :6])
    torch.testing.assert_close(cs[:, :6], cs_changed[:, :6])


def test_rollout_seed_and_member_invariants_without_future_inputs():
    history = torch.linspace(-.3, .3, 16).repeat(2, 1)
    model = PhysicalClosure("memory_colored", .1, .05)
    a, _ = model.rollout(history, 20, 16, 7)
    b, _ = model.rollout(history, 20, 16, 7)
    c, _ = model.rollout(history, 20, 16, 8)
    torch.testing.assert_close(a, b)
    assert not torch.equal(a, c)
    assert not torch.equal(a[0], a[1])
    deterministic = PhysicalClosure("memory", .1, .05)
    d, entropy = deterministic.rollout(history, 20, 16, 7)
    torch.testing.assert_close(d[0], d[-1])
    assert entropy is None


def test_loss_has_finite_mean_scale_memory_and_color_gradients():
    torch.manual_seed(9)
    observed = torch.as_tensor(generate_trajectories(ToyConfig(), 8, 40, 30))
    model = PhysicalClosure("memory_colored_kl", .1, .05)
    # Zero heads intentionally start at G(x). Move heads to let gradients reach
    # the shared trunk/GRU on this first unit-test backward pass.
    torch.nn.init.normal_(model.drift_head.weight, std=.02)
    torch.nn.init.normal_(model.scale_head.weight, std=.02)
    loss = model.loss(observed)
    loss.backward()
    for name in ("drift_head.weight", "scale_head.weight", "gru.weight_ih_l0", "rho_raw"):
        grad = dict(model.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, name


def test_reference_scale_and_evaluation_use_fixed_history_only():
    observed = generate_trajectories(ToyConfig(), 16, 64, 10)
    scale = train_reference_scale(observed[:8], .1)
    assert scale > 0
    model = PhysicalClosure("white", .1, scale)
    metrics, leads, paths = evaluate_model(model, observed[8:], history=16, horizon=20, members=16)
    assert metrics["finite"] and len(leads) == 20
    assert paths["members"].shape == (16, 8, 20)
    assert np.isfinite(metrics["mean_rollout_crps"])
    changed_future = observed[8:].copy()
    changed_future[:, 16:] += 3
    _, _, other_paths = evaluate_model(model, changed_future, history=16, horizon=20, members=16)
    np.testing.assert_array_equal(paths["members"], other_paths["members"])


def test_centered_negative_entropy_is_gaussian_integral_difference():
    scale = torch.tensor([.02, .05, .1], requires_grad=True)
    reference = torch.tensor(.05)
    e = centered_negative_entropy(scale, reference)
    torch.testing.assert_close(e, gaussian_entropy(reference) - gaussian_entropy(scale))
    torch.testing.assert_close(e, -torch.log(scale / reference))
    e.sum().backward()
    torch.testing.assert_close(scale.grad, -1 / scale.detach())
    with pytest.raises(ValueError):
        centered_negative_entropy(torch.tensor([0.]), reference)


def test_entropy_pair_has_identical_initial_weights_and_capacity():
    models = []
    for arm in ENTROPY_INPUT_ARMS:
        torch.manual_seed(17)
        models.append(make_closure(arm, .1, .05))
    for model in models[1:]:
        assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in models[0].parameters())
        for key, value in models[0].state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
    sequence = torch.randn(3, 12) * .2
    laws = [model.parameters_at(sequence)[:2] for model in models]
    for mean, scale in laws[1:]:
        torch.testing.assert_close(mean, laws[0][0])
        torch.testing.assert_close(scale, laws[0][1])


def test_entropy_off_is_explicit_zero_input_with_same_nonzero_scale_network():
    on = make_closure("memory_entropy_on", .1, .05)
    on.scale_head.bias.data.fill_(.4)
    on.entropy_to_memory.weight.data.fill_(.2)
    torch.nn.init.normal_(on.drift_head.weight, std=.1)
    off = make_closure("memory_entropy_off", .1, .05)
    off.load_state_dict(on.state_dict())
    x = torch.randn(3, 10) * .2
    off_law = off.parameters_at(x)
    assert torch.count_nonzero(off.last_entropy_input) == 0
    assert torch.count_nonzero(off.last_raw_entropy_input) == x.numel()
    on_law = on.parameters_at(x)
    assert not torch.allclose(off_law[0], on_law[0])
    on.entropy_enabled = False
    zeroed_on_law = on.parameters_at(x)
    for a, b in zip(off_law, zeroed_on_law):
        torch.testing.assert_close(a, b)


def test_entropy_recurrence_is_causal_streamable_and_reset_by_empty_hidden():
    model = make_closure("memory_entropy_on", .1, .05)
    model.scale_head.bias.data.fill_(.3)
    model.entropy_to_memory.weight.data.fill_(.2)
    torch.nn.init.normal_(model.drift_head.weight, std=.1)
    x = torch.randn(4, 12) * .2
    full_mean, full_scale, full_hidden = model.parameters_at(x)
    means, scales, hidden = [], [], None
    for t in range(x.shape[1]):
        mean, scale, hidden = model.parameters_at(x[:, t:t + 1], hidden)
        means.append(mean)
        scales.append(scale)
    torch.testing.assert_close(torch.cat(means, 1), full_mean)
    torch.testing.assert_close(torch.cat(scales, 1), full_scale)
    torch.testing.assert_close(hidden, full_hidden)
    # A new call with no hidden state starts a fresh trajectory, despite the
    # detached last-entropy diagnostics left by a preceding call.
    repeated = model.parameters_at(x)
    torch.testing.assert_close(repeated[0], full_mean)
    changed = x.clone()
    changed[:, 7:] += 3
    other = model.parameters_at(changed)
    torch.testing.assert_close(other[0][:, :7], full_mean[:, :7])
    torch.testing.assert_close(other[1][:, :7], full_scale[:, :7])


def test_entropy_feedback_carries_mean_gradient_to_scale_head_only_when_on():
    on = make_closure("memory_entropy_on", .1, .05)
    on.scale_head.bias.data.fill_(.3)
    on.entropy_to_memory.weight.data.fill_(.2)
    torch.nn.init.normal_(on.drift_head.weight, std=.1)
    off = make_closure("memory_entropy_off", .1, .05)
    off.load_state_dict(on.state_dict())
    x = torch.randn(4, 12) * .2
    for model in (on, off):
        mean, _, _ = model.parameters_at(x)
        mean.sum().backward()
    grad = on.scale_head.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    assert on.entropy_to_memory.weight.grad.abs().sum() > 0
    off_grad = off.scale_head.weight.grad
    assert off_grad is None or torch.count_nonzero(off_grad) == 0


def test_entropy_sampling_off_trains_identical_law_but_returns_mean_path():
    data = split_data(ToyConfig(), n_train=8, n_validation=4, n_test=4, length=40)
    models = []
    for arm in ("memory_entropy_on", "memory_entropy_mean"):
        torch.manual_seed(23)
        model = make_closure(arm, .1, .05)
        train_model(model, data["train"], data["validation"], seed=23, iterations=3,
                    sequence=16, warmup=4, batch_size=4)
        models.append(model)
    for key, value in models[0].state_dict().items():
        torch.testing.assert_close(value, models[1].state_dict()[key], rtol=0, atol=0)
    history = torch.as_tensor(data["test"][:, :16])
    sampled, _ = models[0].rollout(history, 10, 8, 2)
    mean_path, entropy = models[1].rollout(history, 10, 8, 2)
    assert not torch.equal(sampled[0], sampled[1])
    torch.testing.assert_close(mean_path[0], mean_path[-1])
    assert entropy is not None  # trained uncertainty law still exists when sampling is disabled
    altered = data["test"].copy()
    altered[:, 16:] += 5
    _, _, a = evaluate_model(models[0], data["test"], history=16, horizon=10, members=8)
    _, _, b = evaluate_model(models[0], altered, history=16, horizon=10, members=8)
    np.testing.assert_array_equal(a["members"], b["members"])
