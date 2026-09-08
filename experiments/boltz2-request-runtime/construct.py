"""Instantiate the pinned native Boltz class from safely decoded original weights."""
import argparse
import gc
import inspect
import json
import pathlib
import sys
import time
from dataclasses import asdict

import torch

def load_original(shared_state, safe_loader_dir, kind="conf", sampling_steps=2, recycling_steps=0, diffusion_samples=1):
    sys.path.insert(0, str(safe_loader_dir))
    from safe_state import load_tensor_state
    from boltz.main import Boltz2DiffusionParams, BoltzSteeringParams, PairformerArgsV2, MSAModuleArgs
    from boltz.model.models.boltz2 import Boltz2
    state, hparams = load_tensor_state(shared_state, f"boltz2_{kind}.ckpt")
    accepted = set(inspect.signature(Boltz2).parameters)
    filtered = {k: v for k, v in hparams.items() if k in accepted}
    ignored = sorted(set(hparams) - accepted)
    # Same inference overrides as upstream CLI; ignore obsolete checkpoint fields
    # as Lightning's load_from_checkpoint does for a constructor without **kwargs.
    steering = BoltzSteeringParams()
    if kind == "aff":
        steering.contact_guidance_update = False
    filtered.update(
        ema=False,
        use_kernels=False,
        diffusion_process_args=asdict(Boltz2DiffusionParams()),
        pairformer_args=asdict(PairformerArgsV2()),
        msa_args=asdict(MSAModuleArgs()),
        steering_args=asdict(steering),
        predict_args={"recycling_steps": recycling_steps, "sampling_steps": sampling_steps,
                      "diffusion_samples": diffusion_samples, "max_parallel_samples": 1,
                      "write_confidence_summary": kind == "conf", "write_full_pae": True,
                      "write_full_pde": True},
    )
    start = time.perf_counter()
    model = Boltz2(**filtered).eval()
    model.load_state_dict(state, strict=True)
    del state
    gc.collect()
    return model, {"ignored_obsolete_hparams": ignored, "inference_constructor_kwargs": filtered,
                   "parameters": sum(p.numel() for p in model.parameters()),
                   "buffers": sum(p.numel() for p in model.buffers()),
                   "construct_and_load_seconds": time.perf_counter()-start}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-state", default="/private/tmp/compressme-models/boltz2/shared-state")
    parser.add_argument("--safe-loader-dir", default="/private/tmp/compressme-agent-boltz2-audit")
    parser.add_argument("--kind", choices=["conf", "aff"], default="conf")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("highest")
    model, report = load_original(args.shared_state, args.safe_loader_dir, args.kind)
    report["torch"] = torch.__version__
    report["model_class"] = f"{type(model).__module__}.{type(model).__name__}"
    pathlib.Path(args.output).write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k != "inference_constructor_kwargs"}), flush=True)
