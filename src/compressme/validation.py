"""Empirical whole-model checks. No empirical metric is a universal guarantee."""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
from collections.abc import Mapping
import random
from typing import Any
import torch


@dataclass
class Example:
    args: tuple = ()
    kwargs: dict | None = None

    def call(self, model):
        return model(*self.args, **(self.kwargs or {}))


def tensor_leaves(value: Any, path="output") -> dict[str, Any]:
    if isinstance(value, torch.Tensor):
        return {path: value}
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            result.update(tensor_leaves(item, f"{path}[{key!r}]"))
        return result
    if is_dataclass(value) and not isinstance(value, type):
        return tensor_leaves({f.name: getattr(value, f.name) for f in fields(value)}, path)
    if isinstance(value, (tuple, list)):
        result = {}
        for i, item in enumerate(value):
            result.update(tensor_leaves(item, f"{path}[{i}]"))
        return result
    if value is None or isinstance(value, (str, bool, int, float)):
        return {path: value}
    raise TypeError(f"Unsupported output at {path}: {type(value).__name__}; supply an output_selector")


@contextmanager
def seeded(seed: int):
    py_state = random.getstate()
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    mps_state = torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
    try:
        import numpy as np
        numpy_state = np.random.get_state()
    except ImportError:
        np, numpy_state = None, None
    try:
        random.seed(seed)
        torch.manual_seed(seed)
        if np is not None:
            np.random.seed(seed % (2**32))
        yield
    finally:
        random.setstate(py_state)
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)
        if np is not None:
            np.random.set_state(numpy_state)


def _structure(value):
    if isinstance(value, torch.Tensor):
        return ("tensor",)
    if isinstance(value, Mapping):
        return (type(value), tuple((k, _structure(v)) for k,v in value.items()))
    if is_dataclass(value) and not isinstance(value,type):
        return (type(value), tuple((f.name, _structure(getattr(value,f.name))) for f in fields(value)))
    if isinstance(value,(tuple,list)):
        return (type(value),tuple(_structure(v) for v in value))
    return (type(value),)


def compare_outputs(reference: Any, candidate: Any) -> dict[str, dict]:
    """Measure numerical equality and tensor-byte equality separately.

    `exact` means equal tensor values; `bitwise` additionally distinguishes
    signed zero. Numerical acceptance thresholds do not silently become byte
    equality requirements. Nonfinite floating outputs remain unsupported.
    """
    if _structure(reference) != _structure(candidate):
        raise ValueError("Output structure changed")
    a, b = tensor_leaves(reference), tensor_leaves(candidate)
    if a.keys() != b.keys():
        raise ValueError("Output structure changed")
    metrics = {}
    for key, left in a.items():
        right = b[key]
        if isinstance(left, torch.Tensor):
            if not isinstance(right, torch.Tensor) or left.shape != right.shape or left.dtype != right.dtype:
                raise ValueError(f"Output shape or dtype changed at {key}")
            if left.is_complex():
                raise TypeError("Complex outputs require an explicit real output_selector")
            raw_left, raw_right = left.detach().cpu().resolve_neg(), right.detach().cpu().resolve_neg()
            bitwise = torch.equal(raw_left.contiguous().reshape(-1).view(torch.uint8),
                                  raw_right.contiguous().reshape(-1).view(torch.uint8))
            if not (left.is_floating_point() or left.is_complex()):
                if not torch.equal(left, right):
                    raise ValueError(f"Discrete output changed at {key}")
                metrics[key] = {"exact": True, "bitwise": bitwise, "relative_l2": 0.0, "max_abs": 0.0}
                continue
            x, y = raw_left.double(), raw_right.double()
            if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                raise ValueError(f"Non-finite output at {key}")
            delta = torch.linalg.vector_norm(x - y).item()
            scale = torch.linalg.vector_norm(x).item()
            metrics[key] = {"relative_l2": delta / max(scale, 1e-12),
                            "max_abs": float((x-y).abs().max()) if x.numel() else 0.0,
                            "rmse": delta / max(x.numel(), 1) ** 0.5,
                            "exact": torch.equal(left, right), "bitwise": bitwise}
        else:
            if type(left) is not type(right) or left != right:
                raise ValueError(f"Non-tensor output changed at {key}")
    return metrics


def validate(reference, candidate, examples: list[Example], *, seed=0, output_selector=None,
             relative_tolerance=0.01, absolute_tolerance=None) -> dict:
    if not examples:
        raise ValueError("Validation requires at least one example")
    if relative_tolerance < 0 or (absolute_tolerance is not None and absolute_tolerance < 0):
        raise ValueError("Tolerances must be nonnegative")
    modes = [(m, m.training) for model in (reference, candidate) for m in model.modules()]
    results = []
    try:
        reference.eval(); candidate.eval()
        with torch.inference_mode():
            for i, example in enumerate(examples):
                with seeded(seed + i):
                    left = example.call(reference)
                with seeded(seed + i):
                    right = example.call(candidate)
                if output_selector is not None:
                    left, right = output_selector(left), output_selector(right)
                metrics = compare_outputs(left, right)
                if not metrics:
                    raise ValueError("Validation did not inspect any tensor outputs")
                results.append(metrics)
    finally:
        for m, training in modes:
            m.training = training
    accepted = all(v["relative_l2"] <= relative_tolerance and
                   (absolute_tolerance is None or v["max_abs"] <= absolute_tolerance)
                   for item in results for v in item.values())
    return {"accepted": accepted,
            "bitwise_identical": all(v["bitwise"] for item in results for v in item.values()),
            "examples": len(examples), "relative_tolerance": relative_tolerance,
            "absolute_tolerance": absolute_tolerance, "metrics": results,
            "scope": "empirical agreement on these examples only; not biological accuracy or distributional equivalence"}
