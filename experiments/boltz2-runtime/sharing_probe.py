"""Complete native Boltz-2 outputs before/after generic frozen storage sharing."""
import argparse
import hashlib
import inspect
import json
import pathlib
import random
import time
import traceback

import numpy as np
import torch
from construct import load_original
from native_smoke import tensor_leaves

def comparison(first, second):
    a, b = dict(tensor_leaves(first)), dict(tensor_leaves(second))
    if a.keys() != b.keys(): raise ValueError("Tensor leaf structure changed")
    details = {}
    for name, left in a.items():
        right = b[name]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"Tensor metadata changed: {name}")
        x, y = left.detach().cpu().contiguous(), right.detach().cpu().contiguous()
        bits = torch.equal(x.reshape(-1).view(torch.uint8), y.reshape(-1).view(torch.uint8))
        item = {"shape": list(x.shape), "dtype": str(x.dtype), "bitwise_equal": bits,
                "numeric_equal": torch.equal(x, y), "bytes": x.numel()*x.element_size()}
        if x.is_floating_point():
            item["finite"] = bool(torch.isfinite(x).all() and torch.isfinite(y).all())
            item["max_abs"] = float((x-y).abs().max()) if x.numel() else 0.
            item["mixed_tolerance_1e5"] = bool(torch.allclose(x, y, atol=1e-5, rtol=1e-5))
        details[name] = item
    # Native non-tensor leaves are bool exception flags and dictionary keys.
    if first.get("exception") != second.get("exception"):
        raise ValueError("Exception flag changed")
    return {"all_bitwise_equal": all(v["bitwise_equal"] for v in details.values()),
            "all_finite": all(v.get("finite", True) for v in details.values()),
            "tensor_leaves": len(details), "logical_bytes": sum(v["bytes"] for v in details.values()),
            "max_abs": max(v.get("max_abs", 0.) for v in details.values()), "details": details}

def batch_from_existing(base, molecules, structure_dir=None):
    from boltz.data.types import Manifest
    from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
    base = pathlib.Path(base)/"processed"
    dm = Boltz2InferenceDataModule(manifest=Manifest.load(base/"manifest.json"),
        target_dir=pathlib.Path(structure_dir) if structure_dir else base/"structures", msa_dir=base/"msa",
        mol_dir=pathlib.Path(molecules), num_workers=0, constraints_dir=base/"constraints",
        template_dir=base/"templates", extra_mols_dir=base/"mols", affinity=structure_dir is not None,
        override_method="other" if structure_dir else None)
    return dm, next(iter(dm.predict_dataloader()))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", required=True, choices=["cpu", "mps"])
    p.add_argument("--shared-state", default="/private/tmp/compressme-models/boltz2/shared-state")
    p.add_argument("--safe-loader-dir", default="/private/tmp/compressme-agent-boltz2-audit")
    p.add_argument("--native-dir", default=str(pathlib.Path(__file__).parent))
    p.add_argument("--output", required=True)
    a = p.parse_args()
    torch.set_grad_enabled(False); torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    root = pathlib.Path(a.native_dir)
    report = {"args": vars(a), "torch": torch.__version__, "schedule": {"conf": [3,200,1], "aff": [5,200,3]}}
    stage = "load_originals"
    try:
        from compressme import share_frozen_parameters
        report["sharing_source_sha256"] = hashlib.sha256(pathlib.Path(inspect.getsourcefile(share_frozen_parameters)).read_bytes()).hexdigest()
        bundle = torch.nn.ModuleDict()
        for kind, recycles, samples in [("conf",3,1),("aff",5,3)]:
            model, metadata = load_original(a.shared_state,a.safe_loader_dir,kind,200,recycles,samples)
            bundle[kind] = model
            report[f"original_{kind}"] = metadata
        bundle.eval().requires_grad_(False).to(a.device)
        print("BOTH_STRICT_LOAD_OK", flush=True)
        batches = {}
        for kind in ["conf", "aff"]:
            random.seed(1729); np.random.seed(1729); torch.manual_seed(1729)
            conf_dir = root/f"{a.device}-fp32-default-sampling"
            base = conf_dir if kind == "conf" else root/f"{a.device}-aff-fp32-default-sampling"
            dm, batch = batch_from_existing(base, root/"mols", conf_dir/"predictions" if kind == "aff" else None)
            batches[kind] = dm.transfer_batch_to_device(batch,torch.device(a.device),0)
        def run(kind):
            random.seed(1729); np.random.seed(1729); torch.manual_seed(1729)
            output = bundle[kind].predict_step(batches[kind],0)
            if a.device == "mps": torch.mps.synchronize()
            if output.get("exception"): raise ValueError("Native output exception")
            return output
        stage = "original_predictions"
        originals = {kind: run(kind) for kind in ["conf","aff"]}
        print("BOTH_ORIGINAL_OUTPUTS_OK", flush=True)
        stage = "original_self_repeat"
        report[stage] = {kind: comparison(originals[kind],run(kind)) for kind in ["conf","aff"]}
        print("SELF_REPEAT", {k:v["all_bitwise_equal"] for k,v in report[stage].items()}, flush=True)
        stage = "share_frozen_storage"
        ids = {name:id(value) for name,value in bundle.named_parameters(remove_duplicate=False)}
        result = share_frozen_parameters(bundle,inplace=True)
        report[stage] = result.report
        report["parameter_objects_preserved"] = ids == {name:id(value) for name,value in bundle.named_parameters(remove_duplicate=False)}
        print("SHARING_OK", {k:v for k,v in result.report.items() if k not in {"parameter_replacements", "retained_ineligible_parameters"}}, flush=True)
        stage = "candidate_predictions"
        report[stage] = {kind: comparison(originals[kind],run(kind)) for kind in ["conf","aff"]}
        report["accepted_bitwise"] = report["parameter_objects_preserved"] and all(v["all_bitwise_equal"] and v["all_finite"] for v in report[stage].values())
        print("CANDIDATE", {k:v["all_bitwise_equal"] for k,v in report[stage].items()}, flush=True)
    except Exception as error:
        report.update(accepted_bitwise=False,failure_stage=stage,error=f"{type(error).__name__}: {error}",traceback=traceback.format_exc())
        traceback.print_exc()
    pathlib.Path(a.output).write_text(json.dumps(report,indent=2)+"\n")
    print("SAVED",a.output,"ACCEPTED",report["accepted_bitwise"],flush=True)

if __name__ == "__main__": main()
