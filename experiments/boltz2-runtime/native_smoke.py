"""Native Boltz-2 diagnostic; shortened sampling tests execution, not structure quality."""
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import pathlib
import time
import traceback

import torch
from construct import load_original

def summarize(value):
    if isinstance(value, torch.Tensor):
        v = value.detach().cpu().contiguous()
        raw = v.reshape(-1).view(torch.uint8).numpy().tobytes()
        result = {"shape": list(v.shape), "dtype": str(v.dtype), "sha256": hashlib.sha256(raw).hexdigest(), "numel": v.numel()}
        if v.is_floating_point():
            result["finite"] = bool(torch.isfinite(v).all())
            if v.numel():
                result.update(min=float(v.min()), max=float(v.max()))
        return result
    if isinstance(value, dict):
        return {str(k): summarize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [summarize(v) for v in value]
    return value

def tensor_leaves(value, path="output"):
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from tensor_leaves(v, f"{path}/{k}")
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            yield from tensor_leaves(v, f"{path}/{i}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["cpu", "mps"], required=True)
    p.add_argument("--kind", choices=["conf", "aff"], default="conf")
    p.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--sampling-steps", type=int, default=2)
    p.add_argument("--recycling-steps", type=int, default=0)
    p.add_argument("--diffusion-samples", type=int, default=1)
    p.add_argument("--input", default=str(pathlib.Path(__file__).with_name("tiny_complex.yaml")))
    p.add_argument("--work", required=True)
    p.add_argument("--structure-dir", help="Native confidence prediction folder required for affinity")
    p.add_argument("--molecules", default=str(pathlib.Path(__file__).with_name("mols")))
    p.add_argument("--shared-state", default="/private/tmp/compressme-models/boltz2/shared-state")
    p.add_argument("--safe-loader-dir", default="/private/tmp/compressme-agent-boltz2-audit")
    a = p.parse_args()
    out = pathlib.Path(a.work)
    out.mkdir(parents=True, exist_ok=True)
    report = {"arguments": vars(a), "stages": {}, "versions": {name: importlib.metadata.version(name) for name in ["torch", "boltz", "numpy", "rdkit", "pytorch-lightning"]}}
    stage = "imports"
    try:
        from boltz.main import process_inputs
        from boltz.data.types import Manifest
        from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
        from boltz.data.write.writer import BoltzWriter, BoltzAffinityWriter
        from pytorch_lightning import seed_everything
        from rdkit import Chem
        torch.set_num_threads(4)
        torch.set_grad_enabled(False)
        torch.set_float32_matmul_precision("highest")
        Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
        seed_everything(1729)
        report["mps_available"] = torch.backends.mps.is_available()
        stage = "preprocess"
        start = time.perf_counter()
        process_inputs(data=[pathlib.Path(a.input)], out_dir=out, ccd_path=out/"unused.pkl",
                       mol_dir=pathlib.Path(a.molecules), use_msa_server=False,
                       msa_server_url="https://api.colabfold.com", msa_pairing_strategy="greedy",
                       boltz2=True, preprocessing_threads=1)
        processed = out / "processed"
        manifest = Manifest.load(processed / "manifest.json")
        if len(manifest.records) != 1:
            raise RuntimeError("Expected one successfully parsed input")
        dm = Boltz2InferenceDataModule(manifest=manifest, target_dir=pathlib.Path(a.structure_dir) if a.kind == "aff" else processed/"structures",
             msa_dir=processed/"msa", mol_dir=pathlib.Path(a.molecules), num_workers=0,
             constraints_dir=processed/"constraints", template_dir=processed/"templates", extra_mols_dir=processed/"mols",
             affinity=a.kind == "aff", override_method="other" if a.kind == "aff" else None)
        batch = next(iter(dm.predict_dataloader()))
        report["stages"][stage] = {"seconds": time.perf_counter()-start,
            "tensor_shapes": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k,v in batch.items() if isinstance(v,torch.Tensor)}}
        print("PREPROCESS_OK", flush=True)
        stage = "construct_load"
        model, metadata = load_original(a.shared_state, a.safe_loader_dir, a.kind, a.sampling_steps, a.recycling_steps, a.diffusion_samples)
        report["stages"][stage] = metadata
        print("STRICT_LOAD_OK", metadata["parameters"], flush=True)
        stage = "device_transfer"
        model.to(a.device)
        batch = dm.transfer_batch_to_device(batch, torch.device(a.device), 0)
        if a.device == "mps": torch.mps.synchronize()
        print("TRANSFER_OK", flush=True)
        stage = "predict_step"
        seed_everything(1729)
        start = time.perf_counter()
        precision = torch.autocast(a.device, dtype=torch.bfloat16) if a.precision == "bf16" else contextlib.nullcontext()
        with precision:
            result = model.predict_step(batch, 0)
        if a.device == "mps": torch.mps.synchronize()
        report["stages"][stage] = {"seconds": time.perf_counter()-start, "outputs": summarize(result)}
        if result.get("exception"):
            raise RuntimeError("Native predict_step returned exception flag")
        leaves = dict(tensor_leaves(result))
        bad = [name for name,v in leaves.items() if v.is_floating_point() and not bool(torch.isfinite(v).all())]
        if bad:
            raise RuntimeError(f"Nonfinite output tensors: {bad}")
        from safetensors.torch import save_file
        save_file({name: v.detach().cpu().contiguous() for name,v in leaves.items()}, str(out/"output-tensors.safetensors"))
        report["stages"][stage]["finite_tensor_leaves"] = len(leaves)
        print("PREDICT_OK", time.perf_counter()-start, flush=True)
        stage = "native_writer"
        writer = BoltzWriter(data_dir=processed/"structures", output_dir=out/"predictions", output_format="mmcif", boltz2=True) if a.kind == "conf" else BoltzAffinityWriter(data_dir=a.structure_dir, output_dir=out/"predictions")
        writer.write_on_batch_end(None, model, result, None, batch, 0, 0)
        report["stages"][stage] = {"files": [str(path.relative_to(out)) for path in sorted((out/"predictions").rglob("*")) if path.is_file()]}
        report["accepted"] = True
    except Exception as error:
        report["accepted"] = False
        report["failure_stage"] = stage
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
        traceback.print_exc()
    (out/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in {"stages","traceback"}}), flush=True)

if __name__ == "__main__": main()
