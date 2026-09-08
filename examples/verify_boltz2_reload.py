"""Fresh-process standard artifact load and full native original-output check."""
import argparse
import hashlib
import inspect
import json
import pathlib
import random
import sys
import traceback

import numpy as np
import torch
from safetensors.torch import load_file

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device",required=True,choices=["cpu","mps"])
    p.add_argument("--artifact",default=str(pathlib.Path(__file__).resolve().parents[1]/"artifacts/boltz2-shared"))
    p.add_argument("--native-dir",default=str(pathlib.Path(__file__).resolve().parents[1]/"experiments/boltz2-runtime"))
    p.add_argument("--output",required=True)
    a=p.parse_args()
    torch.set_num_threads(4);torch.set_grad_enabled(False);torch.set_float32_matmul_precision("highest")
    def forbidden(*args,**kwargs):
        raise AssertionError("torch.load is forbidden throughout this reload test")
    torch.load=forbidden
    report={"args":vars(a),"torch":torch.__version__,"torch_load_forbidden":True}
    stage="artifact_load"
    try:
        sys.path.insert(0,a.native_dir)
        from native_smoke import tensor_leaves
        from sharing_probe import batch_from_existing, comparison
        from compressme import load_boltz2
        from compressme.sharing import share_frozen_parameters
        report["loader_sha256"]=hashlib.sha256(pathlib.Path(inspect.getsourcefile(load_boltz2)).read_bytes()).hexdigest()
        report["sharing_source_sha256"]=hashlib.sha256(pathlib.Path(inspect.getsourcefile(share_frozen_parameters)).read_bytes()).hexdigest()
        bundle=load_boltz2(a.artifact,device=a.device)
        report["model_keys"]=list(bundle)
        report["all_frozen"]=all(not v.requires_grad for v in bundle.parameters())
        report["parameters"]=sum(p.numel() for p in bundle.parameters())
        report["final_storage_report"]=bundle._compressme_boltz2_storage_report
        print("ARTIFACT_LOAD_OK",report["parameters"],flush=True)
        root=pathlib.Path(a.native_dir)
        checks={}
        for kind,member in [("conf","confidence"),("aff","affinity")]:
            stage=f"full_output_{kind}"
            conf_dir=root/f"{a.device}-fp32-default-sampling"
            base=conf_dir if kind=="conf" else root/f"{a.device}-aff-fp32-default-sampling"
            reference=root/"cpu-fp32-reference-tensors" if kind=="conf" and a.device=="cpu" else base
            # Upstream featurization randomly augments reference coordinates
            # (featurizerv2.py:1498), independently of diffusion's later RNG.
            # Match the reference native_smoke input-generation seed as well.
            random.seed(1729);np.random.seed(1729);torch.manual_seed(1729)
            dm,batch=batch_from_existing(base,root/"mols",conf_dir/"predictions" if kind=="aff" else None)
            batch=dm.transfer_batch_to_device(batch,torch.device(a.device),0)
            random.seed(1729);np.random.seed(1729);torch.manual_seed(1729)
            actual=bundle[member].predict_step(batch,0)
            if a.device=="mps":torch.mps.synchronize()
            if actual.get("exception"):raise ValueError("Native prediction exception")
            expected=load_file(str(reference/"output-tensors.safetensors"))
            flattened={name:value for name,value in tensor_leaves(actual)}
            checks[kind]=comparison(expected,flattened)
            print("OUTPUT_CHECK",kind,checks[kind]["all_bitwise_equal"],checks[kind]["max_abs"],flush=True)
        report["comparisons"]=checks
        report["accepted_bitwise"]=report["all_frozen"] and all(v["all_bitwise_equal"] and v["all_finite"] for v in checks.values())
    except Exception as error:
        report.update(accepted_bitwise=False,failure_stage=stage,error=f"{type(error).__name__}: {error}",traceback=traceback.format_exc())
        traceback.print_exc()
    pathlib.Path(a.output).write_text(json.dumps(report,indent=2)+"\n")
    print("SAVED",a.output,"ACCEPTED",report["accepted_bitwise"],flush=True)

    if not report["accepted_bitwise"]:
        raise SystemExit(1)

if __name__=="__main__":main()
