"""Controlled CPU study of a learned physical closure, memory and stochastic innovations.

This is a scalar mechanism experiment, NOT a reduced ERA5 benchmark. The only
observations used by learners are x and the known physical tendency G(x). Hidden
memory and forcing from the generator are never returned to a learner. Time is
dimensionless; a Gaussian transition's entropy is differential entropy in these
fixed coordinate units, not thermodynamic entropy.
"""

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn

from .stochastic_memory import gaussian_entropy, gaussian_kl, gaussian_nll


@dataclass(frozen=True)
class ToyConfig:
    dt: float = 0.1
    memory_time: float = 1.0
    memory_strength: float = 0.8
    noise_time: float = 0.4
    noise_std: float = 0.6
    white_forcing: bool = False
    burn_in: int = 100

    def __post_init__(self):
        if self.dt <= 0 or self.memory_time <= 0 or self.noise_time <= 0:
            raise ValueError("dt and time constants must be positive")
        if self.noise_std < 0 or self.memory_strength < 0 or self.burn_in < 0:
            raise ValueError("noise_std, memory_strength and burn_in must be nonnegative")
        if self.dt > self.memory_time:
            raise ValueError("the explicit memory update needs dt <= memory_time")


def physical_tendency(x):
    """Known restoring term, G(x) = -0.8 x - 0.1 x^3 (per unit time)."""
    return -0.8 * x - 0.1 * x ** 3


def generate_trajectories(config: ToyConfig, n: int, length: int, seed: int):
    """Return only observed x, shape (independent trajectory, time).

    eta has an exact stationary OU update (or independent per-step forcing with
    white_forcing; this is NOT a continuous white-noise SDE limit). x and memory use
    explicit Euler with the old x,m, and the new eta. In continuous time the
    memory is the exponential integral of past x with time constant memory_time.
    """
    if n < 1 or length < 2:
        raise ValueError("n >= 1 and length >= 2 are required")
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.5, n)
    m = np.zeros(n)
    eta = rng.normal(0, config.noise_std, n)
    rho = 0.0 if config.white_forcing else math.exp(-config.dt / config.noise_time)
    ou_std = config.noise_std * math.sqrt(1 - rho ** 2)
    out = []
    for t in range(config.burn_in + length):
        if t >= config.burn_in:
            out.append(x.copy())
        eta = rho * eta + ou_std * rng.normal(size=n)
        old_x = x
        x = old_x + config.dt * (physical_tendency(old_x) - config.memory_strength * m + eta)
        m = m + config.dt / config.memory_time * (old_x - m)
    return np.stack(out, axis=1).astype(np.float32)


ARMS = {
    "markov": (False, False, False),
    "memory": (True, False, False),
    "white": (False, True, False),
    "memory_white": (True, True, False),
    "memory_colored": (True, True, True),
    "memory_colored_kl": (True, True, True),
}


class PhysicalClosure(nn.Module):
    """G(x) plus a learned closure, optionally with recurrent observed history.

    The stochastic displacement is sigma_t * z_t, z_t = rho*z_(t-1) +
    sqrt(1-rho^2)*epsilon_t. It is a discrete transition model, not a claim to
    solve an SDE by adding dt-independent noise. Conditional NLL uses exactly
    this same transition law. During observed warmup z_(t-1) is inferred from
    the LAST OBSERVED transition; in forecasts it is the member's sampled z.
    """

    def __init__(self, arm: str, dt: float, reference_scale: float, width: int = 16):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(arm)
        if dt <= 0 or reference_scale <= 0 or width < 2:
            raise ValueError("dt, reference_scale and width must be positive")
        self.arm, self.dt, self.width = arm, float(dt), int(width)
        self.memory, self.stochastic, self.colored = ARMS[arm]
        self.register_buffer("reference_scale", torch.tensor(float(reference_scale)))
        self.gru = nn.GRU(2, width, batch_first=True) if self.memory else None
        self.trunk = nn.Sequential(nn.Linear(2 + (width if self.memory else 0), width), nn.Tanh(),
                                   nn.Linear(width, width), nn.Tanh())
        self.drift_head = nn.Linear(width, 1)
        self.scale_head = nn.Linear(width, 1) if self.stochastic else None
        self.rho_raw = nn.Parameter(torch.zeros(())) if self.colored else None
        nn.init.zeros_(self.drift_head.weight)
        nn.init.zeros_(self.drift_head.bias)
        if self.scale_head is not None:
            nn.init.zeros_(self.scale_head.weight)
            nn.init.zeros_(self.scale_head.bias)

    @property
    def rho(self):
        return 0.98 * torch.tanh(self.rho_raw) if self.colored else self.reference_scale * 0

    def parameters_at(self, x, hidden=None):
        """x: (B,T); return base mean, marginal scale, next GRU hidden."""
        features = torch.stack((x, physical_tendency(x)), dim=-1)
        if self.memory:
            h, hidden = self.gru(features, hidden)
            features = torch.cat((features, h), dim=-1)
        z = self.trunk(features)
        correction = 2.0 * torch.tanh(self.drift_head(z).squeeze(-1))
        mean = x + self.dt * (physical_tendency(x) + correction)
        # Bounded scale prevents numerical overflow; its bounds are fixed in
        # train-residual units and are not selected on validation or test data.
        scale = None if self.scale_head is None else (
            self.reference_scale * torch.exp(self.scale_head(z).squeeze(-1).clamp(-4, 3)))
        return mean, scale, hidden

    def observed_law(self, sequence):
        """Causal teacher-forced law of all observed next-step transitions."""
        mean, scale, _ = self.parameters_at(sequence[:, :-1])
        if not self.colored:
            return mean, scale
        z = (sequence[:, 1:] - mean) / scale
        # Each conditional at t sees z_(t-1), which uses observations only up
        # to x_t. Never insert the current/future residual into its own mean.
        shift = torch.cat((torch.zeros_like(z[:, :1]), z[:, :-1]), dim=1)
        factors = torch.cat((torch.ones_like(scale[:, :1]),
                             torch.ones_like(scale[:, 1:]) * torch.sqrt(1 - self.rho ** 2)), dim=1)
        return mean + self.rho * scale * shift, scale * factors

    def loss(self, sequence, warmup: int = 8, kl_weight: float = 0.01):
        if not 0 <= warmup < sequence.shape[1] - 1:
            raise ValueError("warmup must leave at least one scored transition")
        mean, scale = self.observed_law(sequence)
        y, mean = sequence[:, warmup + 1:], mean[:, warmup:]
        if scale is None:
            return ((y - mean) / self.reference_scale).square().mean()
        scale = scale[:, warmup:]
        loss = gaussian_nll(y, mean, scale).mean()
        if self.arm.endswith("_kl"):
            loss = loss + kl_weight * gaussian_kl(mean, scale, self.reference_scale).mean()
        return loss

    @torch.no_grad()
    def rollout(self, history, horizon: int, members: int, seed: int):
        """Members (M,B,H) from observed IC history; accepts NO future truth."""
        if history.ndim != 2 or history.shape[1] < 2 or horizon < 1 or members < 1:
            raise ValueError("history (B,T>=2), horizon>=1 and members>=1 are required")
        self.eval()
        rng = torch.Generator(device=history.device).manual_seed(seed)
        B = history.shape[0]
        expanded = history.unsqueeze(0).expand(members, -1, -1).reshape(members * B, -1)
        past_mean, past_scale, hidden = self.parameters_at(expanded[:, :-1])
        previous_z = ((expanded[:, -1] - past_mean[:, -1]) / past_scale[:, -1]
                      if self.colored else None)
        x = expanded[:, -1:]
        paths, entropies = [], []
        for _ in range(horizon):
            mean, scale, hidden = self.parameters_at(x, hidden)
            mean = mean[:, 0]
            if self.stochastic:
                scale = scale[:, 0]
                eps = torch.randn(mean.shape, generator=rng, device=mean.device, dtype=mean.dtype)
                if self.colored:
                    z = self.rho * previous_z + torch.sqrt(1 - self.rho ** 2) * eps
                    conditional_scale = scale * torch.sqrt(1 - self.rho ** 2)
                    previous_z = z
                else:
                    z, conditional_scale = eps, scale
                x_next = mean + scale * z
                entropies.append(gaussian_entropy(conditional_scale).reshape(members, B))
            else:
                x_next = mean
            paths.append(x_next.reshape(members, B))
            x = x_next[:, None]
        return torch.stack(paths, -1), (torch.stack(entropies, -1) if entropies else None)


def split_data(config: ToyConfig, n_train=128, n_validation=32, n_test=64, length=128,
               data_seed=1729):
    """Whole independent trajectories, not overlapping windows across splits."""
    return {name: generate_trajectories(config, n, length, data_seed + offset)
            for name, n, offset in (("train", n_train, 0), ("validation", n_validation, 1),
                                    ("test", n_test, 2))}


def train_reference_scale(training, dt):
    """One shared scale from training data only, in one-step displacement units."""
    residual = training[:, 1:] - training[:, :-1] - dt * physical_tendency(training[:, :-1])
    return max(float(np.std(residual)), 1e-4)


def train_model(model, train, validation, seed, iterations=300, batch_size=32, sequence=32,
                warmup=8, learning_rate=0.003, validate_every=25, kl_weight=0.01):
    """Equal optimizer-update budget; checkpoint by each arm's validation loss."""
    if sequence >= train.shape[1] or sequence <= warmup:
        raise ValueError("warmup < sequence < trajectory length is required")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)
    best, best_state, records = math.inf, None, []
    offsets = np.arange(sequence + 1)[None]
    val = torch.as_tensor(validation)
    for step in range(1, iterations + 1):
        model.train()
        ids = rng.integers(len(train), size=batch_size)
        starts = rng.integers(train.shape[1] - sequence, size=batch_size)
        batch = torch.as_tensor(train[ids[:, None], starts[:, None] + offsets])
        optimizer.zero_grad()
        loss = model.loss(batch, warmup, kl_weight)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss for {model.arm} at {step}")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step == 1 or step % validate_every == 0 or step == iterations:
            model.eval()
            with torch.no_grad():
                score = float(model.loss(val, warmup, kl_weight))
            records.append({"iteration": step, "training_loss": float(loss.detach()),
                            "validation_loss": score})
            if score < best:
                best, best_state = score, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return {"best_validation_loss": best, "best_iteration": min(
        records, key=lambda x: x["validation_loss"])["iteration"], "curve": records}


@torch.no_grad()
def evaluate_model(model, testing, history=16, horizon=40, members=64, seed=31415):
    """Evaluate one final untouched test split, including full free rollouts."""
    from .stochastic_memory import ensemble_crps
    if history + horizon > testing.shape[1]:
        raise ValueError("test trajectory is shorter than history + horizon")
    observed = torch.as_tensor(testing[:, :history])
    truth = torch.as_tensor(testing[:, history:history + horizon])
    members_out, entropy = model.rollout(observed, horizon, members, seed)
    valid = torch.isfinite(members_out)
    divergence = (~valid | (members_out.abs() > 20)).float().mean().item()
    if not bool(valid.all()):
        return {"divergence_fraction": divergence, "finite": False}, [], None
    mean = members_out.mean(0)
    rmse = (mean - truth).square().mean(0).sqrt()
    crps = ensemble_crps(members_out, truth, member_dim=0, fair=True).mean(0)
    lower, upper = torch.quantile(members_out, torch.tensor([.05, .95]), dim=0)
    coverage = ((truth >= lower) & (truth <= upper)).float().mean(0)
    spread = members_out.std(0, unbiased=False).square().mean(0).sqrt()
    law_mean, law_scale = model.observed_law(torch.as_tensor(testing))
    target = torch.as_tensor(testing[:, 1:])
    keep = slice(history - 1, None)
    one_mse = float((law_mean[:, keep] - target[:, keep]).square().mean())
    nll = None if law_scale is None else float(gaussian_nll(
        target[:, keep], law_mean[:, keep], law_scale[:, keep]).mean())
    pred_variance = float(members_out.var(unbiased=False))
    true_variance = float(truth.var(unbiased=False))
    temporal = {}
    # These are joint temporal diagnostics of free members, not their mean.
    for lag in (1, 5):
        if lag >= horizon:
            continue
        def acf(y):
            a, b = y[..., :-lag].flatten(), y[..., lag:].flatten()
            a, b = a - a.mean(), b - b.mean()
            return float((a * b).mean() / (a.square().mean() * b.square().mean()).sqrt().clamp_min(1e-12))
        pv = float((members_out[..., lag:] - members_out[..., :-lag]).square().mean())
        tv = float((truth[..., lag:] - truth[..., :-lag]).square().mean())
        temporal[f"acf{lag}"] = {"predicted": acf(members_out), "truth": acf(truth)}
        temporal[f"variogram{lag}"] = {"predicted": pv, "truth": tv, "absolute_error": abs(pv - tv)}
    summary = {"finite": True, "divergence_fraction": divergence,
               "mean_rollout_rmse": float(rmse.mean()), "mean_rollout_crps": float(crps.mean()),
               "coverage90": float(coverage.mean()), "rms_spread": float(spread.square().mean().sqrt()),
               "one_step_mse": one_mse, "one_step_nll": nll,
               "predicted_variance": pred_variance, "true_variance": true_variance,
               "variance_ratio": pred_variance / max(true_variance, 1e-12),
               "mean_conditional_entropy": None if entropy is None else float(entropy.mean()),
               "rho": float(model.rho), "temporal": temporal}
    leads = [{"lead_step": t + 1, "lead_time": (t + 1) * model.dt,
              "rmse": float(rmse[t]), "crps": float(crps[t]),
              "coverage90": float(coverage[t]), "rms_spread": float(spread[t])}
             for t in range(horizon)]
    return summary, leads, {"truth": truth.numpy(), "members": members_out.numpy()}
