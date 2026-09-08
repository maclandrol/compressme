"""Analytic operator rewrites. Bounds concern local real arithmetic only."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class LayerBound:
    residual_spectral_norm: float
    residual_frobenius_norm: float
    anchor_error: float
    relative_operator_error: float
    uniform_l2_bound: float | None = None
    calibration_mean_squared_error: float | None = None
    scope: str = "same-input local real-arithmetic bound, evaluated numerically in float64"
    limitations: str = (
        "Not an interval-certified floating-point bound or an end-to-end output guarantee. "
        "Excludes runtime arithmetic; invalid after changing parameters or dtype."
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InputMoments:
    """Unlabelled input statistics; not teacher-output fitting or distillation."""
    count: int
    mean: Tensor
    covariance: Tensor


def _copy_prepared(target: Tensor, value: Tensor):
    # MPS does not support float64: cast on CPU before crossing the device boundary.
    # Direct CPU-float64 copy_ into MPS-float32 has produced NaNs on tested PyTorch.
    target.copy_(value.to(device="cpu", dtype=target.dtype).to(device=target.device))


def _norms(residual: Tensor) -> tuple[float, float]:
    return float(torch.linalg.matrix_norm(residual, ord=2)), float(torch.linalg.vector_norm(residual))


def _matrix(module: nn.Module) -> tuple[Tensor, Tensor]:
    if not module.weight.is_floating_point() or module.weight.is_complex():
        raise TypeError("Only real floating-point weights are supported")
    weight = module.weight.detach().cpu().double()
    if weight.ndim != 2 or not torch.isfinite(weight).all():
        raise ValueError("Expected a finite two-dimensional linear weight")
    bias = getattr(module, "bias", None)
    bias = torch.zeros(weight.shape[0], dtype=torch.float64) if bias is None else bias.detach().cpu().double()
    if not torch.isfinite(bias).all():
        raise ValueError("Bias must be finite")
    return weight, bias


class LowRankLinear(nn.Module):
    """Ordinary two-GEMM linear map. No original dense matrix is retained."""

    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True,
                 *, device=None, dtype=None):
        super().__init__()
        if not 1 <= rank <= min(in_features, out_features):
            raise ValueError("rank must be between 1 and min(in_features, out_features)")
        self.in_features, self.out_features, self.rank = in_features, out_features, rank
        self.in_channels, self.out_channels = in_features, out_features
        self.right = nn.Parameter(torch.empty(rank, in_features, device=device, dtype=dtype))
        self.left = nn.Parameter(torch.empty(out_features, rank, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype)) if bias else None
        self.register_buffer("anchor", torch.zeros(in_features, device=device, dtype=dtype))
        self.bound: LayerBound | None = None
        self._bound_versions = None

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(F.linear(x, self.right), self.left, self.bias)

    @property
    def weight(self) -> Tensor:
        """Compatibility only: direct consumers reconstruct a dense matrix, losing speed."""
        return self.left @ self.right

    def _seal_bound(self):
        self._bound_versions = self._versions()

    def _versions(self):
        return tuple((id(p), p._version, p.dtype, p.device) for p in [*self.parameters(), self.anchor])

    def bound_is_current(self):
        return self.bound is not None and self._bound_versions == self._versions()

    def error_bound(self, x: Tensor) -> Tensor:
        """Per-vector real-arithmetic upper bound for this layer at the same input.

        Coefficients are numerical float64 estimates, not rigorous interval arithmetic.
        Normal optimizer updates are detected; do not mutate parameters via .data.
        """
        if self.bound is None:
            raise RuntimeError("No source-relative error metadata is attached")
        if not self.bound_is_current():
            raise RuntimeError("Parameters/device/dtype changed; source-relative bound is stale")
        return torch.linalg.vector_norm(x - self.anchor, dim=-1) * self.bound.residual_spectral_norm + self.bound.anchor_error

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}, bias={self.bias is not None}"


class NormLinear(nn.Module):
    """Fixed 1D LayerNorm followed by a factored effective affine operator."""

    def __init__(self, in_features: int, out_features: int, rank: int, eps: float = 1e-5,
                 *, device=None, dtype=None):
        super().__init__()
        self.in_features, self.out_features, self.rank, self.eps = in_features, out_features, rank, eps
        self.projection = LowRankLinear(in_features, out_features, rank, True, device=device, dtype=dtype)
        self.bound: LayerBound | None = None
        self._bound_eps = eps

    def forward(self, x: Tensor) -> Tensor:
        z = F.layer_norm(x, (self.in_features,), eps=self.eps)
        return self.projection(z)

    def error_bound(self, x: Tensor) -> Tensor:
        if self.eps != self._bound_eps:
            raise RuntimeError("Normalization changed; source-relative bound is stale")
        z = F.layer_norm(x, (self.in_features,), eps=self.eps)
        return self.projection.error_bound(z)


def factorize_linear(module: nn.Module, rank: int, *, moments: InputMoments | None = None) -> LowRankLinear:
    """Spectral SVD, or closed-form input-covariance-weighted factorisation.

    With moments, left singular vectors of W C^(1/2) define an optimal rank-r
    output projection for the supplied input covariance. This is established
    activation-aware low-rank approximation, not a new compression theorem.
    """
    w, b = _matrix(module)
    m, d = w.shape
    if not 1 <= rank <= min(m, d):
        raise ValueError("Invalid rank")
    mean = torch.zeros(d, dtype=torch.float64)
    if moments is None:
        u, s, vh = torch.linalg.svd(w, full_matrices=False)
        root = s[:rank].sqrt()
        left, right = u[:, :rank] * root, root[:, None] * vh[:rank]
    else:
        mean = moments.mean.detach().cpu().double()
        cov = moments.covariance.detach().cpu().double()
        if moments.count < 1 or mean.shape != (d,) or cov.shape != (d, d):
            raise ValueError("Input moments do not match the linear layer")
        if not torch.isfinite(mean).all() or not torch.isfinite(cov).all():
            raise ValueError("Input moments must be finite")
        eig, vec = torch.linalg.eigh((cov + cov.T) / 2)
        if eig.min() < -1e-8 * max(1.0, float(eig.abs().max())):
            raise ValueError("Covariance must be positive semidefinite")
        root_cov = vec * eig.clamp_min(0).sqrt()
        cov = root_cov @ root_cov.T
        u, _, _ = torch.linalg.svd(w @ root_cov, full_matrices=False)
        left = u[:, :rank]
        right = left.T @ w
    # Preserve the affine anchor, including for originally bias-free modules.
    bias = b + (w - left @ right) @ mean
    source = module.weight
    out = LowRankLinear(d, m, rank, bool(getattr(module, "bias", None) is not None or moments is not None), device=source.device, dtype=source.dtype)
    with torch.no_grad():
        _copy_prepared(out.left, left)
        _copy_prepared(out.right, right)
        if out.bias is not None:
            _copy_prepared(out.bias, bias)
        _copy_prepared(out.anchor, mean)
    out.train(module.training)
    for parameter in (out.left, out.right):
        parameter.requires_grad_(source.requires_grad)
    if out.bias is not None:
        source_bias = getattr(module, "bias", None)
        out.bias.requires_grad_(source.requires_grad if source_bias is None else source_bias.requires_grad)
    stored = out.left.detach().cpu().double() @ out.right.detach().cpu().double()
    residual = w - stored
    spectral, frobenius = _norms(residual)
    stored_bias = torch.zeros_like(b) if out.bias is None else out.bias.detach().cpu().double()
    stored_mean = out.anchor.detach().cpu().double()
    anchor_error = float(torch.linalg.vector_norm(residual @ stored_mean + b - stored_bias))
    source_norm = float(torch.linalg.matrix_norm(w, ord=2))
    mse = None
    if moments is not None:
        mse = max(0.0, float(torch.trace(residual @ cov @ residual.T))) + float(torch.linalg.vector_norm(residual @ mean + b - stored_bias)) ** 2
    out.bound = LayerBound(spectral, frobenius, anchor_error, spectral / source_norm if source_norm else spectral, calibration_mean_squared_error=mse)
    out._seal_bound()
    return out


def factorize_norm_linear(norm: nn.LayerNorm, linear: nn.Linear, rank: int) -> NormLinear:
    """Minimax rank-r approximation on LayerNorm's centred, bounded domain.

    In reals: sup_x ||f(x)-f_r(x)||_2 = sqrt(d) sigma_(r+1)(W diag(gamma) P).
    Returned metadata uses residuals of the actually stored factors and bias.
    """
    w, b = _matrix(linear)
    m, d = w.shape
    if type(norm) is not nn.LayerNorm or tuple(norm.normalized_shape) != (d,) or not math.isfinite(norm.eps) or norm.eps <= 0:
        raise ValueError("Requires fixed nn.LayerNorm over the last feature dimension, eps > 0")
    gamma = torch.ones(d, dtype=torch.float64) if norm.weight is None else norm.weight.detach().cpu().double()
    beta = torch.zeros(d, dtype=torch.float64) if norm.bias is None else norm.bias.detach().cpu().double()
    if not torch.isfinite(gamma).all() or not torch.isfinite(beta).all():
        raise ValueError("Normalization affine parameters must be finite")
    effective = w * gamma[None, :]
    effective = effective - effective.mean(dim=1, keepdim=True)
    constant = w @ beta + b
    with torch.random.fork_rng(devices=[]):
        proxy = nn.Linear(d, m, dtype=torch.float64)
    with torch.no_grad():
        proxy.weight.copy_(effective)
        proxy.bias.copy_(constant)
    factored = factorize_linear(proxy, rank)
    out = NormLinear(d, m, rank, norm.eps, device=linear.weight.device, dtype=linear.weight.dtype)
    with torch.no_grad():
        _copy_prepared(out.projection.left, factored.left)
        _copy_prepared(out.projection.right, factored.right)
        _copy_prepared(out.projection.bias, constant)
    stored = out.projection.left.detach().cpu().double() @ out.projection.right.detach().cpu().double()
    # Residual acts only on the centred normalisation subspace.
    residual = effective - stored
    residual -= residual.mean(dim=1, keepdim=True)
    spectral, frobenius = _norms(residual)
    anchor_error = float(torch.linalg.vector_norm(constant - out.projection.bias.detach().cpu().double()))
    source_norm = float(torch.linalg.matrix_norm(effective, ord=2))
    out.bound = LayerBound(spectral, frobenius, anchor_error, spectral / source_norm if source_norm else spectral,
                           uniform_l2_bound=math.sqrt(d) * spectral + anchor_error)
    out.projection.bound = out.bound
    out.projection._seal_bound()
    # Fixed affine folding does not retain independently trainable norm parameters.
    # A wholly frozen source stays frozen; otherwise the fused map is trainable.
    trainable = any(p.requires_grad for block in (norm, linear) for p in block.parameters())
    out.requires_grad_(trainable)
    out.projection._seal_bound()
    out.train(linear.training)
    return out
