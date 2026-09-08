"""Validate every exported field and all optional Mol-JEPA modalities.

Run with --checkpoint /path/to/original/model.safetensors --device cpu|mps.
No inputs are used for fitting; the optional vectors are deterministic random
arrays to exercise API branches, not realistic biological benchmark data.
"""
import argparse
import json
from pathlib import Path
import torch
from compressme import Example, load_moljepa, validate
from moljepa import ROOT, original, examples


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--device",choices=["cpu","mps"],default="cpu")
    p.add_argument("--artifact",type=Path,default=ROOT/"artifacts/moljepa-full")
    p.add_argument("--report",type=Path,default=ROOT/"benchmarks/mol_jepa_export_optional.json")
    args = p.parse_args()
    torch.set_num_threads(4)
    reference = original(args.checkpoint).to(args.device)
    candidate = load_moljepa(args.artifact,device=args.device)
    smiles = json.loads((ROOT/"benchmarks/verification_smiles.json").read_text())["smiles"]
    inputs = examples(smiles)
    generator = torch.Generator().manual_seed(9922)
    supplied = []
    for spec in reference.model.modalities_spec:
        if spec["name"] == "graph": continue
        dims = (3,spec["node_dim"]) if spec["output"] == "atoms" else (spec["dim"],)
        supplied.append({spec["name"]:torch.randn(dims,generator=generator)})
    inputs.append(Example((smiles[:len(supplied)],),
                          {"embeddings_data":supplied,"return_attn":True}))
    # Also check different atom counts within the same optional modality batch.
    uma = [{"uma":torch.randn(n,128,generator=generator)} for n in (1,3,7)]
    inputs.append(Example((smiles[:3],),{"embeddings_data":uma,"return_attn":True}))
    result = validate(reference,candidate,inputs,relative_tolerance=1e-5,absolute_tolerance=1e-5)
    result.update(device=args.device, dtype="float32",
                  tested_optional_modalities=[next(iter(item)) for item in supplied])
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(result,indent=2)+"\n")
    maxima = {key:max(m[key] for batch in result["metrics"] for m in batch.values())
              for key in ("max_abs","relative_l2")}
    print(json.dumps({"accepted":result["accepted"],"maxima_all_fields":maxima},indent=2))
    if not result["accepted"]: raise SystemExit(1)


if __name__ == "__main__": main()
