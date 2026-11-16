"""Opt-in stochastic closure around the existing daily tendency/memory core.

Only torch is imported here; constructing the real core is the caller's job. The
adapter leaves the deterministic core unchanged and adds a learned conditional
diagonal Gaussian *before a final call to its physical projection*. Consequently
``gaussian_nll`` and ``gaussian_entropy`` describe that pre-projection Gaussian,
not the generally non-Gaussian distribution after ``PositiveMass``. Evaluate the
returned, projected ensemble with ``ensemble_crps``. Forced channels have zero
noise and must be excluded from Gaussian likelihood/entropy calculations.

The stochastic source is not globally conservative relative to the deterministic
forecast. ``PositiveMass`` preserves the moisture total of its noisy input, not
that of the deterministic location. No zero-integral noise projection is used:
it would change the Gaussian covariance and require a different density model.

The adapter does not assert a fluctuation-dissipation law or exact Mori-Zwanzig
decomposition. Its noise is white across time; memory remains in the original
core. A low-dimensional pilot is not evidence of ERA5 forecast improvement.
"""

import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


def _floating(value, like=None):
    if like is not None:
        return torch.as_tensor(value, dtype=like.dtype, device=like.device)
    value = torch.as_tensor(value)
    return value if value.is_floating_point() else value.to(torch.get_default_dtype())


def _scale(value, like=None, min_scale=1e-6):
    if not math.isfinite(min_scale) or min_scale <= 0:
        raise ValueError("min_scale must be finite and positive")
    value = _floating(value, like)
    if not bool(torch.isfinite(value).all()) or not bool((value > 0).all()):
        raise ValueError("Gaussian scales must be finite and strictly positive")
    # Reject invalid distributions rather than hiding negative/zero scales. The
    # floor only stabilizes very small, already-positive standard deviations.
    return value.clamp_min(min_scale)


def gaussian_nll(target, mean, scale, *, min_scale=1e-6):
    """Elementwise -log N(target; mean, scale**2), in the supplied coordinates.

    This is a pre-projection likelihood. No reduction or channel/area weighting
    is performed. A scale floor changes the modeled Gaussian accordingly.
    """
    mean = _floating(mean)
    target, scale = _floating(target, mean), _scale(scale, mean, min_scale)
    return 0.5 * ((target - mean) / scale).square() + scale.log() + 0.5 * math.log(2 * math.pi)


def gaussian_entropy(scale, *, min_scale=1e-6):
    """Elementwise differential entropy of a pre-projection Gaussian.

    Joint entropy is the sum only if its components are conditionally independent.
    Differential entropy depends on coordinates/units and may be negative.
    """
    scale = _scale(scale, min_scale=min_scale)
    return scale.log() + 0.5 * math.log(2 * math.pi * math.e)


def gaussian_kl(mean, scale, reference_scale, *, min_scale=1e-6):
    """Elementwise KL(N(mean,s**2) || N(mean,reference_s**2)).

    The means are IDENTICAL: this is a variance regularizer, not a path-space
    relative entropy, control-energy penalty, or full Schrodinger-bridge loss.
    ``mean`` supplies coordinates/device and the broadcast output shape.
    """
    mean = _floating(mean)
    scale = _scale(scale, mean, min_scale)
    reference_scale = _scale(reference_scale, mean, min_scale)
    log_ratio = scale.log() - reference_scale.log()
    return 0.5 * torch.expm1(2 * log_ratio) - log_ratio + 0.0 * mean


def ensemble_crps(samples, target, member_dim=0, *, fair=True):
    """Elementwise ensemble CRPS, valid for projected/non-Gaussian samples.

    ``fair=True`` uses the unbiased pairwise U-statistic for M >= 2 (independent
    identically distributed members are required). ``fair=False`` scores the
    empirical ensemble distribution. One member is its degenerate distribution's
    absolute error. Sorting avoids allocating an M-by-M difference tensor.
    """
    samples = _floating(samples)
    if samples.ndim == 0:
        raise ValueError("samples must have a member dimension")
    samples = samples.movedim(member_dim, 0)
    members = samples.shape[0]
    if members < 1:
        raise ValueError("at least one member is required")
    target = _floating(target, samples)
    first = (samples - target).abs().mean(0)
    if members == 1:
        return first
    ordered = samples.sort(dim=0).values
    weights = (2 * torch.arange(1, members + 1, device=samples.device, dtype=samples.dtype)
               - members - 1).view(members, *([1] * (samples.ndim - 1)))
    denominator = members * (members - 1 if fair else members)
    return first - (weights * ordered).sum(0) / denominator


class StochasticMemoryCore(nn.Module):
    """Learn sigma(x,m) while retaining an existing core's drift and memory.

    Required core interface is the current ``make_core(mem=...)`` interface:
    ``step``, ``reset``, ``finish``, ``_trunk``, ``t6``, ``keep``, and ``fc``.
    Trunk width is inferred from ``core.net.head.in_channels``. Use the original
    core's ``norm='gn'`` to avoid extra batch statistics/dropout during sigma
    evaluation. ``step(..., stochastic=False)`` calls only the original step.

    The returned distribution parameters are the *location and standard
    deviation before the final physical projection*, not the mean/variance of
    the projected ensemble. ``initial_scale``/``scale_floor`` are in units of the
    original per-channel daily increment standard deviation ``core.t6``.
    Active channels additionally have an absolute standard-deviation floor of
    1e-6 in normalized state coordinates, matching the Gaussian helpers' default
    floor. The same returned scale is used to sample; known-forcing scales stay
    exactly zero. This absolute floor can dominate a very small increment scale.
    """

    def __init__(self, core: nn.Module, *, initial_scale=0.1, scale_floor=1e-4,
                 step_hours=24.0):
        super().__init__()
        if not math.isfinite(scale_floor) or scale_floor <= 0:
            raise ValueError("scale_floor must be finite and positive")
        if not math.isfinite(initial_scale) or initial_scale <= scale_floor:
            raise ValueError("initial_scale must be finite and greater than scale_floor")
        if not math.isfinite(step_hours) or step_hours <= 0:
            raise ValueError("step_hours must be finite and positive")
        if not bool(torch.isfinite(core.t6).all()) or not bool((core.t6 > 0).all()):
            raise ValueError("core.t6 must contain finite positive increment scales")
        self.core, self.scale_floor, self.step_hours = core, float(scale_floor), float(step_hours)
        self.noise_head = nn.Conv2d(core.net.head.in_channels, core.t6.shape[1], 1)
        self.noise_head.to(device=core.t6.device, dtype=core.t6.dtype)
        nn.init.zeros_(self.noise_head.weight)
        initial_softplus = initial_scale - scale_floor
        inverse_softplus = initial_softplus + math.log(-math.expm1(-initial_softplus))
        nn.init.constant_(self.noise_head.bias, inverse_softplus)

    def reset(self, history=None, hours0=None, x0=None):
        """Reset every member's history; observations must end at/before the IC."""
        return self.core.reset(history, hours0, x0)

    def step(self, x, hours, tisr_next=None, *, stochastic=True,
             generator: Optional[torch.Generator] = None, return_distribution=False):
        """One causal step; optional return is (sample, location, stddev).

        ``tisr_next`` is known astronomical forcing, never a future weather target.
        ``return_distribution=True`` also evaluates sigma in deterministic mode.
        """
        location = self.core.step(x, hours, tisr_next)
        if not stochastic and not return_distribution:
            return location
        # step() has now updated the core's memory with x, so the sigma head
        # conditions only on this day's x, G(x), and causal memory.
        hidden = self.core._trunk(x, hours)
        scale = (F.softplus(self.noise_head(hidden)) + self.scale_floor) * self.core.t6 * self.core.keep
        # Match the density helpers' default numerical floor *before sampling*.
        # Forced channels remain deterministic and are not Gaussian scored.
        scale = torch.where(self.core.keep > 0, scale.clamp_min(1e-6), torch.zeros_like(scale))
        sample = location
        if stochastic:
            noise = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
            sample = self.core.finish(location + scale * noise, tisr_next)
            # The next autonomous state is this sample, not an observation of
            # the deterministic forecast. Keep e=(x-fc)/s exactly zero on it.
            self.core.fc = sample
        return (sample, location, scale) if return_distribution else sample

    def forward(self, x, hours, tisr_next=None, **kwargs):
        return self.step(x, hours, tisr_next, **kwargs)

    def rollout(self, x0, hours0, steps, tisr=None, history=None, *, stochastic=True,
                generator: Optional[torch.Generator] = None, seed=None):
        """Reset, then follow member trajectories without future weather inputs.

        Batch entries can be repeated ICs/members; each receives independent
        draws from the supplied generator. Repeating the same seed reproduces
        the complete rollout. Warm-up history must be repeated along the same
        batch dimension as the ICs. ``hours0`` is HOURS, not day-of-year.
        """
        if steps < 1:
            raise ValueError("steps must be positive")
        if seed is not None:
            if generator is not None:
                raise ValueError("supply a generator or seed, not both")
            generator = torch.Generator(device=x0.device).manual_seed(int(seed))
        hours0 = _floating(hours0, x0)
        self.reset(history, hours0, x0)
        x, out = x0, []
        stride = getattr(self.core, "stride", 1)
        for k in range(steps):
            forcing = None if tisr is None else tisr[:, k]
            x = self.step(x, hours0 + k * self.step_hours * stride, forcing,
                          stochastic=stochastic, generator=generator)
            out.append(x)
        return torch.stack(out, 1)


class EntropyConditionedMemoryCore(StochasticMemoryCore):
    """Feed centered negative Gaussian entropy directly into causal memory.

    ``core`` must be an existing ``make_core(mem>0, norm='gn', ...)`` memory
    core, with ``remember`` and ``_advance`` as well as ``mem_ch``. For the
    dimensionless increment eta=(X_next-location)/core.t6, the conditional
    pre-projection diagonal Gaussian has standard deviation ``s``. Its input is

        e = integral p log(p) - integral p_ref log(p_ref) = -log(s / s_ref).

    This is centered negative *differential* entropy per active channel. It is
    neither KL nor thermodynamic entropy. A zero-initialized bias-free 1x1 map
    adds e to the GRU memory. Setting ``entropy_enabled=False`` supplies e=0 to
    the same architecture and parameter count; a fresh zero map reproduces the
    original deterministic dynamics. There is no assumed benefit from this
    feature, since e is itself a function of the learned conditional scale.

    One day is acyclic: remember x -> sigma(x,G,m) -> e -> augmented m ->
    deterministic _advance -> optional noise -> finish. Sigma never reads the
    current entropy-augmented memory or the next observed weather state. Reset
    replays these same deterministic entropy-conditioned steps on observed
    *past* states and stores their forecasts in ``core.seq`` for an MZ loss.

    ``step``/``rollout`` retain the parent API, including HOURS for hours0, and
    the same pre-projection density, forcing, and conservation limitations.
    ``last_entropy_input`` and ``last_relative_scale`` expose the latest causal
    input; ``seq`` exposes warm-up forecasts. The noise is white in time.
    """

    def __init__(self, core: nn.Module, *, reference_relative_scale=0.1,
                 entropy_enabled=True, initial_scale=0.1, scale_floor=1e-4,
                 step_hours=24.0):
        if not getattr(core, "mem", 0) or not hasattr(core, "remember") or not hasattr(core, "_advance"):
            raise ValueError("entropy conditioning requires an existing causal memory core")
        super().__init__(core, initial_scale=initial_scale, scale_floor=scale_floor, step_hours=step_hours)
        reference = _floating(reference_relative_scale, core.t6)
        if not bool(torch.isfinite(reference).all()) or not bool((reference > 0).all()):
            raise ValueError("reference_relative_scale must be finite and positive")
        if reference.ndim == 1:
            reference = reference.view(1, -1, 1, 1)
        try:
            reference_shape = torch.broadcast_shapes(reference.shape, core.t6.shape)
        except RuntimeError as exc:
            raise ValueError("reference_relative_scale must broadcast over the core channels") from exc
        if reference_shape != core.t6.shape:
            raise ValueError("reference_relative_scale must be scalar or fixed per core channel")
        self.register_buffer("reference_relative_scale", reference.clone())
        self.entropy_enabled = bool(entropy_enabled)
        self.entropy_to_memory = nn.Conv2d(core.t6.shape[1], core.mem_ch, 1, bias=False)
        self.entropy_to_memory.to(device=core.t6.device, dtype=core.t6.dtype)
        nn.init.zeros_(self.entropy_to_memory.weight)
        self.last_entropy_input, self.last_relative_scale = None, None

    @property
    def mem(self):
        return self.core.mem

    @property
    def mz(self):
        return getattr(self.core, "mz", False)

    @property
    def seq(self):
        return self.core.seq

    @property
    def stride(self):
        return getattr(self.core, "stride", 1)

    def entropy_feature(self, relative_scale):
        """Centered integral p log p, masked to predicted channels (or zero)."""
        relative_scale = _floating(relative_scale, self.core.t6)
        if not bool(torch.isfinite(relative_scale).all()) or not bool((relative_scale > 0).all()):
            raise ValueError("relative_scale must be finite and positive")
        if not self.entropy_enabled:
            return torch.zeros_like(relative_scale) * self.core.keep
        return -(relative_scale.log() - self.reference_relative_scale.log()) * self.core.keep

    def reset(self, history=None, hours0=None, x0=None):
        """Replay only observed history, deterministically, with this adapter.

        ``history`` is (B,W,C,H,W), oldest first. ``hours0`` is the IC time and
        is required when W>0. ``x0`` supplies only the final warm-up forecast's
        known forcing channel. Its other channels cannot enter that forecast.
        """
        self.core.reset(None)
        self.last_entropy_input, self.last_relative_scale = None, None
        if history is None or history.shape[1] == 0:
            return
        if hours0 is None:
            raise ValueError("hours0 is required for causal history replay")
        hours0 = _floating(hours0, history)
        count = history.shape[1]
        inactive = torch.nonzero(self.core.keep.reshape(-1) == 0, as_tuple=False).flatten()
        if inactive.numel() > 1:
            raise ValueError("history replay supports at most one known forcing channel")
        forcing_index = int(inactive[0]) if inactive.numel() else None
        forecasts = []
        freeze = getattr(self.core, "freeze_mem", False)
        self.core.freeze_mem = False  # observed warm-up always updates memory
        try:
            for k in range(count):
                next_observed = history[:, k + 1] if k + 1 < count else x0
                forcing = (None if next_observed is None or forcing_index is None
                           else next_observed[:, forcing_index])
                # Preserve MemCore.reset's daily warm-up clock exactly. Its
                # optional forecast stride is applied after the IC by rollout.
                hours = hours0 - (count - k) * self.step_hours
                forecasts.append(self.step(history[:, k], hours, forcing, stochastic=False))
        finally:
            self.core.freeze_mem = freeze
        self.core.seq = torch.stack(forecasts, 1)
        # Match the original reset: freeze_mem begins at the first forecast IC,
        # not during the observed history used to initialize its memory.
        self.core.started = False

    def step(self, x, hours, tisr_next=None, *, stochastic=True,
             generator: Optional[torch.Generator] = None, return_distribution=False):
        update_memory = not (getattr(self.core, "freeze_mem", False) and getattr(self.core, "started", False))
        if update_memory:
            self.core.remember(x)
        # Sigma reads current x/G and the memory BEFORE the current entropy
        # injection; this order avoids an algebraic sigma <-> entropy loop.
        hidden = self.core._trunk(x, hours)
        raw_relative = F.softplus(self.noise_head(hidden)) + self.scale_floor
        absolute = (raw_relative * self.core.t6).clamp_min(1e-6)
        relative = absolute / self.core.t6
        entropy = self.entropy_feature(relative)
        self.last_entropy_input, self.last_relative_scale = entropy, relative
        if update_memory:
            self.core.m = self.core.m + self.entropy_to_memory(entropy)
        self.core.started = True
        location = self.core._advance(x, hours, tisr_next)
        scale = absolute * self.core.keep
        sample = location
        if stochastic:
            noise = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
            sample = self.core.finish(location + scale * noise, tisr_next)
        self.core.fc = sample
        return (sample, location, scale) if return_distribution else sample
