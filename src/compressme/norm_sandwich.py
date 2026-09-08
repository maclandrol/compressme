"""Exact affine -> LayerNorm -> affine contraction, without rank truncation.

The identity is in real arithmetic. QR, coefficient storage and changed
operation order introduce floating-point differences. This standalone
prototype does not provide an interval-certified numerical error bound.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from .runtime import autocast_enabled


@dataclass
class NormSandwichReport:
    input_width: int
    normalized_width: int
    output_width: int
    denominator_width: int
    parameters_before: int
    parameters_after: int
    tensor_bytes_before: int
    tensor_bytes_after: int
    method: str = "exact_affine_layernorm_affine_contraction"
    guarantee: str = "same mathematical function in real arithmetic; no rank truncation"
    limitation: str = (
        "QR and coefficient casting are numerical; runtime association changes. "
        "Independent updates change the original parameter coupling."
    )

    def to_dict(self):
        return asdict(self)


class NormSandwich(nn.Module):
    """Evaluate a contracted normalised affine operator with the same x API.

    Stores D, R, c only: y = D[x;1] / sqrt(||R[x;1]||^2/n + eps) + c.
    R has every row from reduced QR; no small singular values are discarded.
    The source affine and normalisation modules are not retained.
    """

    def __init__(self, in_features: int, normalized_features: int,
                 out_features: int, eps: float = 1e-5, *, device=None, dtype=None):
        super().__init__()
        if min(in_features, normalized_features, out_features) < 1:
            raise ValueError("All feature dimensions must be positive")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and strictly positive")
        self.in_features = in_features
        self.normalized_features = normalized_features
        self.out_features = out_features
        self.eps = float(eps)
        self.denominator_width = min(normalized_features, in_features + 1)
        self.numerator = nn.Parameter(torch.empty(out_features, in_features+1,
                                                 device=device, dtype=dtype))
        self.denominator = nn.Parameter(torch.empty(self.denominator_width, in_features+1,
                                                   device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        if autocast_enabled(x.device.type):
            raise ValueError("NormSandwich does not support autocast; use explicit model/input dtypes")
        if x.shape[-1] != self.in_features:
            raise ValueError("Input feature dimension does not match the contracted block")
        if x.dtype != self.numerator.dtype or x.device != self.numerator.device:
            raise ValueError("Input dtype/device must match the contracted block; autocast is not audited")
        # Half-precision square accumulation easily overflows before division.
        # Coefficients remain stored in their original dtype.
        accumulation_dtype = (torch.float32 if x.dtype in (torch.float16, torch.bfloat16)
                              else x.dtype)
        aug = torch.cat((x, torch.ones_like(x[..., :1])), dim=-1).to(accumulation_dtype)
        numerator = F.linear(aug, self.numerator.to(accumulation_dtype))
        denominator = F.linear(aug, self.denominator.to(accumulation_dtype))
        scale = (denominator.square().sum(-1, keepdim=True) /
                 self.normalized_features + self.eps).sqrt()
        return (numerator / scale + self.bias.to(accumulation_dtype)).to(x.dtype)

    def extra_repr(self):
        return (f"in_features={self.in_features}, normalized_features={self.normalized_features}, "
                f"out_features={self.out_features}, denominator_width={self.denominator_width}, "
                f"eps={self.eps}")


def _has_hooks(module):
    return any(bool(value) for name, value in vars(module).items()
               if name.endswith("_hooks") and isinstance(value, dict))


def compose_norm_sandwich(first: nn.Linear, norm: nn.LayerNorm,
                          last: nn.Linear) -> tuple[NormSandwich, NormSandwichReport]:
    """Compose standalone standard affine/LayerNorm/affine modules exactly.

    All modules must use one dtype/device and have finite materialized real
    parameters. The caller must establish the immediate dataflow and account
    for every external use before removing source modules. The returned report
    counts the isolated three-module block. It can report negative savings.
    """
    if type(first) is not nn.Linear or type(last) is not nn.Linear or type(norm) is not nn.LayerNorm:
        raise TypeError("Requires standard nn.Linear, nn.LayerNorm, nn.Linear modules")
    modules = (first, norm, last)
    if any(_has_hooks(m) or "forward" in vars(m) for m in modules):
        raise ValueError("Custom hooks or instance forwards are not supported")
    d, n, m = first.in_features, first.out_features, last.out_features
    if last.in_features != n or tuple(norm.normalized_shape) != (n,):
        raise ValueError("LayerNorm must cover exactly the intermediate feature dimension")
    if not math.isfinite(norm.eps) or norm.eps <= 0:
        raise ValueError("LayerNorm eps must be finite and strictly positive")
    parameters = [p for module in modules for p in module.parameters()]
    if len({id(p) for p in parameters}) != len(parameters):
        raise ValueError("Shared parameters require a separate compiler analysis")
    device, dtype = first.weight.device, first.weight.dtype
    if dtype not in (torch.float64, torch.float32, torch.float16, torch.bfloat16) or device.type == "meta":
        raise ValueError("Requires materialized float64/32/16 or bfloat16 parameters")
    for parameter in parameters:
        if parameter.device != device or parameter.dtype != dtype or not torch.isfinite(parameter).all():
            raise ValueError("Parameters must be finite and have one dtype and device")
    A = first.weight.detach().cpu().double()
    a = (torch.zeros(n, dtype=torch.float64) if first.bias is None
         else first.bias.detach().cpu().double())
    W = last.weight.detach().cpu().double()
    b = (torch.zeros(m, dtype=torch.float64) if last.bias is None
         else last.bias.detach().cpu().double())
    gamma = (torch.ones(n, dtype=torch.float64) if norm.weight is None
             else norm.weight.detach().cpu().double())
    beta = (torch.zeros(n, dtype=torch.float64) if norm.bias is None
            else norm.bias.detach().cpu().double())
    # Left centering is exact as an algebraic identity. The augmented bias is
    # centred too; it need not lie in the linear span of the centred columns.
    T = torch.cat((A-A.mean(0, keepdim=True), (a-a.mean())[:, None]), dim=1)
    # Full reduced QR handles both rank-deficient and wide T without truncation.
    R = torch.linalg.qr(T, mode="r").R
    D = (W*gamma[None, :]) @ T
    c = W @ beta + b
    result = NormSandwich(d, n, m, norm.eps, device=device, dtype=dtype)
    with torch.no_grad():
        result.numerator.copy_(D.to(dtype=dtype).to(device))
        result.denominator.copy_(R.to(dtype=dtype).to(device))
        result.bias.copy_(c.to(dtype=dtype).to(device))
    if not all(torch.isfinite(p).all() for p in result.parameters()):
        raise ValueError("Contracted coefficients overflow the source dtype")
    result.numerator.requires_grad_(any(p is not None and p.requires_grad for p in
                                       (first.weight, first.bias, norm.weight, last.weight)))
    result.denominator.requires_grad_(any(p is not None and p.requires_grad for p in
                                         (first.weight, first.bias)))
    result.bias.requires_grad_(any(p is not None and p.requires_grad for p in
                                  ((last.weight if norm.bias is not None else None), norm.bias, last.bias)))
    result.train(last.training)
    report = NormSandwichReport(
        d, n, m, result.denominator_width,
        sum(p.numel() for p in parameters), sum(p.numel() for p in result.parameters()),
        sum(p.numel()*p.element_size() for p in parameters),
        sum(p.numel()*p.element_size() for p in result.parameters()),
    )
    return result, report
