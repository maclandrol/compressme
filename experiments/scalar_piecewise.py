"""Exact-real scalar affine/hinge/affine normal form; frozen inference prototype.

For activation L_alpha(z)=alpha*z+(1-alpha)*ReLU(z), every nonzero
first-layer slope a defines t=-b/a. Its left-hand affine contribution is
q*(a*x+b), where q=alpha for a>0 and q=1 for a<0, and its hinge
contribution is (1-alpha)*abs(a)*ReLU(x-t). Zero slopes are constants.
No input-sign or finite-domain assumption is needed for this identity.

Stored coefficients keep the original dtype. Their rounded representation
has the certified real-evaluation error bound E0+E1*abs(x), separately for
each output. That bound excludes source/candidate floating-point execution
rounding and overflow. Numerical checks below are probes, not a universal
floating-point guarantee. This is a classic hinge/spline normal form, not a
claim of a new representation theorem. Input gradients at knots can differ
under PyTorch's activation subgradient conventions, so this is inference only.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ScalarPiecewiseResult:
    model: nn.Module
    report: dict


def _fraction(value):
    return Fraction.from_float(float(value))


def _upward(value):
    if not value:
        return 0.0
    try:
        rounded = float(value)
    except OverflowError:
        return math.inf
    return math.nextafter(rounded, math.inf)


class ScalarPiecewise(nn.Module):
    """One scalar input per row, arbitrary leading shapes; no source retained."""
    def __init__(self, knots, jumps, slope, intercept):
        super().__init__()
        self._knots = nn.Parameter(knots, requires_grad=False)
        # Concatenation enables one output GEMM including the affine slope.
        self._coefficients = nn.Parameter(torch.cat((jumps, slope[:, None]), dim=1), requires_grad=False)
        self._intercept = nn.Parameter(intercept, requires_grad=False)
        self.eval()

    def train(self, mode=True):
        if mode:
            raise ValueError("Scalar hinge normal form supports frozen inference only")
        return super().train(False)

    def forward(self, x):
        if x.ndim < 1 or x.shape[-1] != 1:
            raise ValueError("Expected one scalar feature per row")
        if x.dtype != self._knots.dtype or x.device != self._knots.device:
            raise ValueError("Use the compiled explicit dtype/device")
        if torch.is_autocast_enabled(x.device.type):
            raise ValueError("Autocast is outside this explicit-dtype prototype")
        if self.training or (torch.is_grad_enabled() and
                             (x.requires_grad or any(p.requires_grad for p in self.parameters()))):
            raise ValueError("This rewrite preserves outputs, not activation subgradients at knots")
        features = torch.cat((F.relu(x - self._knots), x), dim=-1)
        return F.linear(features, self._coefficients, self._intercept)


def build_scalar_piecewise(source, *, validation_inputs=None,
                           absolute_tolerance=1e-5, relative_tolerance=1e-5):
    """Return a numerically checked smaller candidate, or unchanged-copy rollback.

    source must be an ordinary eval Sequential(Linear(1,h), ReLU/LeakyReLU,
    Linear(h,m)). validation_inputs may be a Tensor or iterable of Tensors;
    caller cases augment all knots, knot neighborhoods, and signed probes.
    Every stored model coefficient is included in the parameter/state counts.
    Validation data are not retained and no fitting or training is performed.
    """
    if (type(source) is not nn.Sequential or len(source) != 3
            or type(source[0]) is not nn.Linear or type(source[2]) is not nn.Linear
            or type(source[1]) not in (nn.ReLU, nn.LeakyReLU)):
        raise ValueError("Expected Linear(1,h) -> ReLU/LeakyReLU -> Linear(h,m)")
    if any(m.training or "forward" in vars(m) or any(getattr(m, k, None) for k in
           ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks"))
           for m in source.modules()):
        raise ValueError("Use unmodified, hook-free eval modules")
    first, activation, last = source
    h, m = first.out_features, last.out_features
    if (first.in_features != 1 or last.in_features != h
            or tuple(first.weight.shape) != (h, 1) or tuple(last.weight.shape) != (m, h)
            or (first.bias is not None and tuple(first.bias.shape) != (h,))
            or (last.bias is not None and tuple(last.bias.shape) != (m,))):
        raise ValueError("Linear metadata and weights must match the scalar chain")
    state = tuple(source.parameters()) + tuple(source.buffers())
    device, dtype = first.weight.device, first.weight.dtype
    if (dtype not in (torch.float32, torch.float64) or device.type == "meta"
            or any(p.device != device or p.dtype != dtype or p.layout != torch.strided
                   or not torch.isfinite(p).all() for p in state)):
        raise ValueError("Use finite float32/float64 tensors on one materialized device")
    if torch.is_autocast_enabled(device.type):
        raise ValueError("Disable autocast while constructing exact coefficients")
    if any(getattr(p, "_backward_hooks", None) for p in state):
        raise ValueError("Tensor hooks are unsupported")
    alpha = 0.0 if type(activation) is nn.ReLU else activation.negative_slope
    if not math.isfinite(alpha):
        raise ValueError("Activation slope must be finite")
    if any(not math.isfinite(t) or t < 0 for t in (absolute_tolerance, relative_tolerance)):
        raise ValueError("Numerical tolerances must be finite and nonnegative")
    alpha = _fraction(alpha)
    a = [_fraction(x) for x in first.weight.detach().cpu().flatten().tolist()]
    b = ([_fraction(x) for x in first.bias.detach().cpu().tolist()]
         if first.bias is not None else [Fraction(0)] * h)
    c = [[_fraction(x) for x in row] for row in last.weight.detach().cpu().tolist()]
    d = ([_fraction(x) for x in last.bias.detach().cpu().tolist()]
         if last.bias is not None else [Fraction(0)] * m)
    slope, intercept = [Fraction(0)] * m, list(d)
    knots, columns = [], []
    for j in range(h):
        if a[j] == 0:
            constant = b[j] if b[j] >= 0 else alpha * b[j]
            for i in range(m):
                intercept[i] += c[i][j] * constant
            continue
        q = alpha if a[j] > 0 else Fraction(1)
        for i in range(m):
            slope[i] += c[i][j] * q * a[j]
            intercept[i] += c[i][j] * q * b[j]
        jump = [(1 - alpha) * abs(a[j]) * c[i][j] for i in range(m)]
        if any(jump):
            knots.append(-b[j] / a[j])
            columns.append(jump)
    jumps = [[column[i] for column in columns] for i in range(m)]
    original_count = sum(p.numel() for p in source.parameters())
    count = len(knots) * (m + 1) + 2 * m
    report = {"method": "scalar_hinge_normal_form", "source_parameters": original_count,
              "candidate_parameters": count, "removed_parameters": original_count - count,
              "hidden_units": h, "retained_hinges": len(knots), "output_width": m,
              "dtype": str(dtype), "device": str(device),
              "mathematical_scope": "All real scalar inputs; exact unrounded coefficient identity",
              "training": "Frozen inference; activation subgradients at knots can differ",
              "interval_cache": "None; all model coefficient storage is counted"}
    if count >= original_count:
        report["status"] = "no_storage_saving"
        return ScalarPiecewiseResult(copy.deepcopy(source), report)
    try:
        with torch.inference_mode(False), torch.no_grad():
            def tensor(values):
                return torch.tensor(values, dtype=dtype, device=device)
            candidate = ScalarPiecewise(tensor([float(x) for x in knots]),
                tensor([[float(x) for x in row] for row in jumps]).reshape(m, len(knots)),
                tensor([float(x) for x in slope]), tensor([float(x) for x in intercept]))
    except (OverflowError, RuntimeError) as exc:
        report.update(status="rejected_coefficient_range", reason=str(exc))
        return ScalarPiecewiseResult(copy.deepcopy(source), report)
    if any(not torch.isfinite(p).all() for p in candidate.parameters()):
        report["status"] = "rejected_coefficient_range"
        return ScalarPiecewiseResult(copy.deepcopy(source), report)

    # Exact rational residual certificate for the stored coefficients, viewed
    # as real numbers. No source tensors or rational arrays enter the model.
    stored_knots = [_fraction(x) for x in candidate._knots.detach().cpu().tolist()]
    stored_coeff = [[_fraction(x) for x in row]
                    for row in candidate._coefficients.detach().cpu().tolist()]
    stored_intercept = [_fraction(x) for x in candidate._intercept.detach().cpu().tolist()]
    e0, e1 = [], []
    for i in range(m):
        offset, growth = abs(stored_intercept[i] - intercept[i]), abs(stored_coeff[i][-1] - slope[i])
        for j in range(len(knots)):
            residual = abs(stored_coeff[i][j] - jumps[i][j])
            offset += residual * abs(knots[j]) + abs(stored_coeff[i][j]) * abs(stored_knots[j] - knots[j])
            growth += residual
        e0.append(_upward(offset)); e1.append(_upward(growth))
    report["stored_coefficient_real_error_bound"] = {
        "formula": "abs(error_i(x)) <= offset_i + growth_i * abs(x)",
        "offset": e0, "growth": e1,
        "excludes": "Floating-point execution rounding and overflow in either implementation"}

    probes = torch.tensor([-100., -10., -1., -1e-6, 0., 1e-6, 1., 10., 100.], dtype=dtype, device=device)
    k = candidate._knots.detach()
    margin = torch.finfo(dtype).eps * 16 * torch.maximum(k.abs(), torch.ones_like(k))
    probes = torch.cat((probes, k, k - margin, k + margin))
    probes = probes[torch.isfinite(probes)].reshape(-1, 1)
    cases = [probes, torch.zeros(1, dtype=dtype, device=device),
             torch.zeros(0, 1, dtype=dtype, device=device),
             torch.zeros(2, 3, 1, dtype=dtype, device=device)]
    if validation_inputs is not None:
        cases.extend([validation_inputs] if isinstance(validation_inputs, torch.Tensor) else validation_inputs)
    max_abs, max_relative = 0.0, 0.0
    with torch.inference_mode():
        for x in cases:
            reference, result = source(x), candidate(x)
            if reference.shape != result.shape or not torch.isfinite(reference).all() or not torch.isfinite(result).all():
                max_abs = max_relative = math.inf
                break
            if reference.numel():
                reference_cpu = reference.detach().to(device="cpu", dtype=torch.float64)
                result_cpu = result.detach().to(device="cpu", dtype=torch.float64)
                delta = (reference_cpu - result_cpu).flatten()
                max_abs = max(max_abs, delta.abs().max().item())
                max_relative = max(max_relative, (torch.linalg.vector_norm(delta) /
                    torch.linalg.vector_norm(reference_cpu).clamp_min(1e-30)).item())
    accepted = max_abs <= absolute_tolerance and max_relative <= relative_tolerance
    report.update(status="accepted_on_numerical_probes" if accepted else "rejected_numerical_probe",
                  numerical_max_abs=max_abs, numerical_relative_l2=max_relative,
                  validation_cases=len(cases), absolute_tolerance=absolute_tolerance,
                  relative_tolerance=relative_tolerance,
                  candidate_state_bytes=sum(p.numel() * p.element_size() for p in candidate.state_dict().values()))
    return ScalarPiecewiseResult(candidate if accepted else copy.deepcopy(source), report)
