"""Save the validated sharing transform using the general compressme serializer."""
import argparse
import gc
import hashlib
import inspect
import json
import pathlib
import torch
from construct import load_original

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shared-state", default="/private/tmp/compressme-models/boltz2/shared-state")
    p.add_argument("--safe-loader-dir", default="/private/tmp/compressme-agent-boltz2-audit")
    p.add_argument("--output", default="/private/tmp/compressme-models/boltz2/compiled-bundle")
    p.add_argument("--evidence-dir", default=str(pathlib.Path(__file__).parent))
    a = p.parse_args()
    from compressme import share_frozen_parameters, save
    torch.set_grad_enabled(False); torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    bundle = torch.nn.ModuleDict()
    for kind, recycles, samples in [("conf",3,1),("aff",5,3)]:
        model, _ = load_original(a.shared_state,a.safe_loader_dir,kind,200,recycles,samples)
        bundle[{"conf":"confidence", "aff":"affinity"}[kind]] = model
    bundle.eval().requires_grad_(False)
    result = share_frozen_parameters(bundle,inplace=True)
    del model
    gc.collect()
    evidence = {}
    source_hash = hashlib.sha256(pathlib.Path(inspect.getsourcefile(share_frozen_parameters)).read_bytes()).hexdigest()
    for backend in ["cpu", "mps"]:
        path = pathlib.Path(a.evidence_dir)/f"sharing-{backend}.json"
        report = json.loads(path.read_text())
        if report["sharing_source_sha256"] != source_hash or not report["accepted_bitwise"]:
            raise ValueError("Complete output evidence is not accepted for this transform")
        evidence[backend] = {"report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
             "all_output_bytes_equal": True,
             "tensor_leaves": sum(v["tensor_leaves"] for v in report["candidate_predictions"].values()),
             "logical_value_bytes": sum(v["logical_bytes"] for v in report["candidate_predictions"].values()),
             "scope": "One 20-aa protein and ethanol ligand; same-backend original, seeded full default diffusion/recycling, both native models"}
    result.report["complete_output_validation"] = evidence
    result.report["sharing_source_sha256"] = source_hash
    directory = save(result,a.output,packing=False)
    print("SAVED",str(directory),flush=True)
    print(json.dumps({path.name:path.stat().st_size for path in pathlib.Path(directory).iterdir() if path.is_file()}),flush=True)

if __name__ == "__main__": main()
