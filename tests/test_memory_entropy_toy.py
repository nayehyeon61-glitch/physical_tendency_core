"""Probability, causality and split invariants of the controlled CPU study."""

import numpy as np
import pytest
import torch

from pgcfm.memory_entropy_toy import (PhysicalClosure, ToyConfig, evaluate_model,
                                     generate_trajectories, physical_tendency, split_data, train_reference_scale)


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
