"""Build and verify compressed artifacts from the audited public Mol-JEPA weights.

Usage from project root:
  .venv/bin/python examples/moljepa.py build --checkpoint /path/to/model.safetensors
  .venv/bin/python examples/moljepa.py predict --device mps CCO 'c1ccccc1'
  .venv/bin/python examples/moljepa.py verify --checkpoint /path/to/model.safetensors --device mps
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
import shutil
import statistics
import time
import torch
from safetensors.torch import load_file
from compressme import (Example, CompressionResult, load_moljepa, optimize_moljepa,
                        parameter_count, specialize_moljepa_smiles, state_bytes, validate)
from compressme.moljepa_io import moljepa_factory

ROOT = Path(__file__).resolve().parents[1]


def original(checkpoint):
    model = moljepa_factory(ROOT / "vendor" / "moljepa")
    model.load_state_dict(load_file(str(checkpoint)), strict=True)
    return model.eval()


def examples(smiles):
    # Fixed verification inputs are never used for training or fitting.
    return [Example((smiles[i:i+8],), {"return_attn": True}) for i in range(0, len(smiles), 8)]


def build(args):
    model = original(args.checkpoint)
    smiles = json.loads((ROOT / "benchmarks" / "verification_smiles.json").read_text())
    if isinstance(smiles, dict):
        smiles = smiles["smiles"]
    result = optimize_moljepa(model, validation=examples(smiles))
    if result.report["status"] != "accepted_on_validation_examples":
        raise RuntimeError("Original-output agreement gate rejected this export")
    result.report["source_revision"] = "4c912b450175f31b5ba913a5dc921c03b27b985a"
    full_dir = args.output / "moljepa-full"
    result.save(full_dir,packing=args.packing)
    shutil.copytree(ROOT / "vendor" / "moljepa", full_dir / "architecture", dirs_exist_ok=True)
    small_model, spec_report = specialize_moljepa_smiles(result.model)
    small_report = dict(result.report)
    small_report.update(parameters_after=parameter_count(small_model),
                        tensor_bytes_after=state_bytes(small_model),
                        input_contract="embeddings_data=None",
                        specialization=spec_report)
    small_result = CompressionResult(small_model, small_report)
    small_dir = args.output / "moljepa-smiles"
    small_result.save(small_dir,packing=args.packing)
    shutil.copytree(ROOT / "vendor" / "moljepa", small_dir / "architecture", dirs_exist_ok=True)
    # Exercise fresh constructor + compressed tensor-only reload, without
    # requiring the original weights inside either deliverable.
    exported = {"original_parameters":parameter_count(model)}
    for name, directory in (("full",full_dir),("smiles",small_dir)):
        recovered = load_moljepa(directory)
        agreement = validate(model,recovered,examples(smiles),
                             relative_tolerance=1e-5,absolute_tolerance=1e-5)
        if not agreement["accepted"]:
            raise RuntimeError(f"Reloaded {name} failed output validation")
        weight_file = json.loads((directory/"manifest.json").read_text()).get("weights_file","model.safetensors")
        exported[name] = {"parameters":parameter_count(recovered),
                          "weights_bytes":(directory/weight_file).stat().st_size,
                          "reload_validation":agreement}
    (args.output / "export_verification.json").write_text(json.dumps(exported,indent=2)+"\n")
    print(json.dumps({k:{kk:vv for kk,vv in v.items() if kk != "reload_validation"}
                           if isinstance(v,dict) else v for k,v in exported.items()},indent=2))


def verify(args):
    models = {"original":original(args.checkpoint).to(args.device),
              "compressed":load_moljepa(args.artifact,device=args.device)}
    smiles = json.loads((ROOT / "benchmarks" / "verification_smiles.json").read_text())
    if isinstance(smiles,dict):
        smiles = smiles["smiles"]
    agreement = validate(models["original"],models["compressed"],examples(smiles),
                         relative_tolerance=1e-5,absolute_tolerance=1e-5)
    # Warm each candidate, then interleave randomised calls to reduce order bias.
    batch = smiles[:args.batch_size]
    timings = {key:[] for key in models}
    def sync():
        if args.device == "mps": torch.mps.synchronize()
    with torch.inference_mode():
        for model in models.values():
            for _ in range(5): model(batch)
        rng = random.Random(739)
        for _ in range(20):
            names = list(models)
            rng.shuffle(names)
            for name in names:
                sync(); start = time.perf_counter()
                models[name](batch)
                sync(); timings[name].append(time.perf_counter()-start)
    medians = {k:statistics.median(v) for k,v in timings.items()}
    report = {"device":args.device,"batch_size":len(batch),"threads":torch.get_num_threads(),
              "agreement":agreement,"seconds":timings,"median_seconds":medians,
              "speedup":medians["original"]/medians["compressed"]}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"accepted":agreement["accepted"],"median_seconds":medians,"speedup":report["speedup"]},indent=2))
    if not agreement["accepted"]:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command",required=True)
    p = commands.add_parser("build")
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,default=ROOT/"artifacts")
    p.add_argument("--packing",action="store_true",help="Pack weights losslessly for smaller disk storage")
    p = commands.add_parser("predict")
    p.add_argument("smiles",nargs="+")
    p.add_argument("--artifact",type=Path,default=ROOT/"artifacts"/"moljepa-smiles")
    p.add_argument("--device",choices=["cpu","mps"],default="cpu")
    p = commands.add_parser("verify")
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--artifact",type=Path,default=ROOT/"artifacts"/"moljepa-smiles")
    p.add_argument("--device",choices=["cpu","mps"],default="cpu")
    p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--report",type=Path,default=ROOT/"benchmarks"/"verification.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.command == "build":
        build(args)
    elif args.command == "verify":
        verify(args)
    else:
        output = load_moljepa(args.artifact,device=args.device)(args.smiles)
        print(json.dumps({key:list(value.shape) for key,value in output.items()
                          if isinstance(value,torch.Tensor)},indent=2))


if __name__ == "__main__":
    main()
