"""Run a native single-complex Boltz YAML/FASTA with the compressed model pair.

Uses the original parser, data module, prediction methods and output writers.
The caller supplies the upstream molecular reference directory and any MSA
files named in the input. No MSA-server submission is made by this example.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

import torch


def predict(input_path, *, artifact, molecules, output, device="cpu", seed=1729, invariant_conditioning=False):
    from compressme import load_boltz2
    input_path, molecules, output = map(Path, (input_path, molecules, output))
    if not input_path.is_file() or not molecules.is_dir():
        raise ValueError("Supply an input file and the native Boltz molecular reference directory")
    if output.exists():
        raise FileExistsError("Choose a new output directory to keep prior predictions intact")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    bundle = load_boltz2(artifact, device=device)
    from boltz.main import process_inputs
    from boltz.data.types import Manifest
    from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
    from boltz.data.write.writer import BoltzWriter, BoltzAffinityWriter
    from pytorch_lightning import seed_everything
    from rdkit import Chem
    from safetensors.torch import save_file

    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
    output.mkdir(parents=True)
    seed_everything(seed)
    process_inputs(
        data=[input_path], out_dir=output, ccd_path=output / "unused.pkl",
        mol_dir=molecules, use_msa_server=False,
        msa_server_url="https://api.colabfold.com", msa_pairing_strategy="greedy",
        boltz2=True, preprocessing_threads=1,
    )
    processed = output / "processed"
    manifest = Manifest.load(processed / "manifest.json")
    if len(manifest.records) != 1:
        raise RuntimeError("The native parser did not produce exactly one complex")
    predictions = output / "predictions"
    report = {"device": device, "seed": seed, "artifact": str(artifact),
              "input": str(input_path), "precision": "float32", "outputs": {}}

    def tensor_leaves(value, path="output"):
        if isinstance(value, torch.Tensor):
            yield path, value
        elif isinstance(value, dict):
            for key, child in value.items():
                yield from tensor_leaves(child, f"{path}/{key}")
        elif isinstance(value, (tuple, list)):
            for index, child in enumerate(value):
                yield from tensor_leaves(child, f"{path}/{index}")

    members = ["confidence"]
    if manifest.records[0].affinity:
        members.append("affinity")
    with torch.no_grad():
        for member in members:
            affinity = member == "affinity"
            target_dir = predictions if affinity else processed / "structures"
            dm = Boltz2InferenceDataModule(
                manifest=manifest, target_dir=target_dir, msa_dir=processed / "msa",
                mol_dir=molecules, num_workers=0,
                constraints_dir=processed / "constraints", template_dir=processed / "templates",
                extra_mols_dir=processed / "mols", affinity=affinity,
                override_method="other" if affinity else None,
            )
            # Reference conformers are randomly augmented during featurization.
            seed_everything(seed)
            batch = next(iter(dm.predict_dataloader()))
            batch = dm.transfer_batch_to_device(batch, torch.device(device), 0)
            seed_everything(seed)
            model = bundle[member]
            if invariant_conditioning:
                from compressme.boltz2_runtime import boltz2_invariant_sampling
                import inspect
                report["runtime_sha256"] = hashlib.sha256(
                    Path(inspect.getsourcefile(inspect.unwrap(boltz2_invariant_sampling))).read_bytes()).hexdigest()
                context = boltz2_invariant_sampling(model, immutable_request=True)
            else:
                context = nullcontext(None)
            with context as runtime_report:
                result = model.predict_step(batch, 0)
            if result.get("exception"):
                raise RuntimeError(f"Native {member} prediction returned an exception")
            leaves = dict(tensor_leaves(result))
            if any(v.is_floating_point() and not bool(torch.isfinite(v).all())
                   for v in leaves.values()):
                raise RuntimeError(f"Native {member} prediction contains nonfinite values")
            save_file({name: value.detach().cpu().contiguous() for name, value in leaves.items()},
                      str(output / f"{member}-tensors.safetensors"))
            writer = (BoltzAffinityWriter(data_dir=predictions, output_dir=predictions)
                      if affinity else BoltzWriter(data_dir=processed / "structures",
                          output_dir=predictions, output_format="mmcif", boltz2=True))
            writer.write_on_batch_end(None, model, result, None, batch, 0, 0)
            report["outputs"][member] = {"tensor_leaves": len(leaves),
                                         "predict_args": dict(model.predict_args),
                                         "request_runtime": runtime_report}
    report["files"] = [str(p.relative_to(output)) for p in sorted(predictions.rglob("*")) if p.is_file()]
    (output / "compressme.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="One native Boltz YAML or FASTA file")
    parser.add_argument("--artifact", type=Path,
                        default=Path(__file__).resolve().parents[1] / "artifacts/boltz2-shared")
    parser.add_argument("--molecules", type=Path, required=True,
                        help="Upstream molecular reference directory, complete for the requested inputs")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--invariant-conditioning", action="store_true",
                        help="Opt into experimental request-invariant execution; benchmark on your workload")
    args = parser.parse_args()
    result = predict(args.input, artifact=args.artifact, molecules=args.molecules,
                     output=args.output, device=args.device, seed=args.seed,
                     invariant_conditioning=args.invariant_conditioning)
    print(json.dumps(result, indent=2))
