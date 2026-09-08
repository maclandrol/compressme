#!/usr/bin/env python3
"""Prepare genuine native Nesso-1 inputs, without executing Nesso-1 predictions.

Original work: https://github.com/recursionpharma/nesso and
https://doi.org/10.64898/2026.08.01.742196. Source fixtures are Apache-2.0.

This explicitly executes the reviewed local Nesso source and loads a
publisher-trusted, hash-checked CCD pickle. ESM-2 runs on CPU in float32 using
Nesso's unchanged final-layer extraction helper. No weights are downloaded.
Prepared tensors are serialized with safetensors; metadata uses tagged JSON.
The realized input tensors, not independently regenerated RDKit conformers,
are the repeatable inputs for the later model comparison and timing.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
from typing import Any

NESSO_REVISION = "6c72f66720d9d3447fd73c515cda963e39128b1f"
ESM_REPOSITORY = "facebook/esm2_t33_650M_UR50D"
ESM_REVISION = "08e4846e537177426273712802403f7ba8261b6c"
CASES = {
    "tiny20": {"protein_residues": 20, "ligand_heavy_atoms": 3,
               "scope": "synthetic peptide/ethanol execution smoke; no binding label"},
    "fragment130": {"protein_residues": 130, "ligand_heavy_atoms": 13,
                    "scope": "exact upstream tests/test_forward.py fragment; no binding label"},
    "tutorial384": {"protein_residues": 384, "ligand_heavy_atoms": 13,
                    "scope": "unchanged full upstream tutorial/smiles.yaml input; no binding label"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source(source: Path) -> dict[str, str]:
    """Verify the exact tracked Python source before adding it to the import path."""
    def git(*args: str) -> bytes:
        return subprocess.check_output(["git", "-C", str(source), *args], stderr=subprocess.PIPE)
    if git("rev-parse", "HEAD").decode().strip() != NESSO_REVISION:
        raise ValueError(f"Nesso checkout must be at {NESSO_REVISION}")
    names = [x for x in git("ls-tree", "-r", "--name-only", "HEAD", "nesso").decode().splitlines()
             if x.endswith(".py")]
    actual = {p.relative_to(source).as_posix() for p in (source / "nesso").rglob("*.py")}
    if actual != set(names):
        raise ValueError("Nesso Python file inventory differs from the pinned commit")
    result = {}
    for name in names:
        observed = (source / name).read_bytes()
        if observed != git("show", f"HEAD:{name}"):
            raise ValueError(f"Modified upstream source: {name}")
        result[name] = hashlib.sha256(observed).hexdigest()
    return result


def snapshot_files(snapshot: Path) -> dict[str, dict[str, Any]]:
    if snapshot.name != ESM_REVISION:
        raise ValueError(f"Use the local Hugging Face snapshot directory named {ESM_REVISION}")
    names = ["config.json", "model.safetensors", "special_tokens_map.json",
             "tokenizer_config.json", "vocab.txt"]
    missing = [n for n in names if not (snapshot / n).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete ESM safetensors snapshot: {missing}")
    # These optional tokenizer/generation files are consumed if present.
    names += [n for n in ("tokenizer.json", "added_tokens.json", "generation_config.json")
              if (snapshot / n).is_file()]
    result = {}
    for name in names:
        path = snapshot / name
        digest = sha256(path)
        result[name] = {"bytes": path.stat().st_size, "sha256": digest}
        if path.is_symlink():
            blob = path.resolve().name
            if len(blob) == 64 and digest != blob:
                raise ValueError(f"ESM LFS blob checksum mismatch: {name}")
            if len(blob) == 40:
                data = path.read_bytes()
                observed = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
                if observed != blob:
                    raise ValueError(f"ESM Git blob checksum mismatch: {name}")
    return result


def _encode(value: Any, tensors: dict, key: str = "batch") -> Any:
    import numpy as np
    import torch
    if isinstance(value, torch.Tensor):
        if not value.is_contiguous():
            raise ValueError(f"Unexpected non-contiguous native batched tensor: {key}")
        tensors[key] = value.detach().cpu()
        return {"kind": "tensor", "key": key}
    if hasattr(value, "to_dict") and type(value).__name__ == "Record":
        return {"kind": "record", "value": value.to_dict()}
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int, float)):
        return {"kind": "scalar", "value": value}
    if isinstance(value, (list, tuple)):
        return {"kind": "tuple" if isinstance(value, tuple) else "list",
                "items": [_encode(v, tensors, f"{key}.{i}") for i, v in enumerate(value)]}
    if isinstance(value, dict):
        return {"kind": "dict", "items": [[_encode(k, tensors, f"{key}.key{i}"),
                _encode(v, tensors, f"{key}.{k}")] for i, (k, v) in enumerate(value.items())]}
    raise TypeError(f"Unsupported native non-tensor metadata: {key}: {type(value)}")


def _decode(tree: dict, tensors: dict) -> Any:
    kind = tree["kind"]
    if kind == "tensor":
        return tensors[tree["key"]]
    if kind == "record":
        from nesso.data.types import Record
        return Record.from_dict(tree["value"])
    if kind == "scalar":
        return tree["value"]
    if kind in ("list", "tuple"):
        values = [_decode(v, tensors) for v in tree["items"]]
        return tuple(values) if kind == "tuple" else values
    if kind == "dict":
        return {_decode(k, tensors): _decode(v, tensors) for k, v in tree["items"]}
    raise ValueError(f"Unsupported fixture metadata kind: {kind}")


def load_batch(directory: str | Path, device: str = "cpu") -> dict:
    """Load exact prepared inputs; caller must have the pinned Nesso package installed.

    No CCD or tensor pickle is loaded. Record metadata uses Nesso's JSON constructor.
    ``device`` transfers tensor leaves without changing their dtype.
    """
    from safetensors.torch import load_file
    directory = Path(directory)
    metadata = json.loads((directory / "batch.json").read_text())
    if metadata.get("format") != "compressme.nesso.fixture.v1":
        raise ValueError("Unsupported fixture format")
    path = directory / "batch.safetensors"
    if sha256(path) != metadata["tensor_file"]["sha256"]:
        raise ValueError("Prepared batch tensor checksum mismatch")
    tensors = load_file(str(path), device="cpu")
    if str(device) != "cpu":
        tensors = {k: v.to(device) for k, v in tensors.items()}
    return _decode(metadata["batch_tree"], tensors)


def prepare(args: argparse.Namespace) -> dict:
    source = Path(args.nesso_source).expanduser().resolve()
    snapshot = Path(args.esm_snapshot).expanduser().absolute()
    ccd = Path(args.ccd).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output must be a new directory: {output}")
    source_hashes = verify_source(source)
    ccd_hash = sha256(ccd)
    if ccd_hash != args.ccd_sha256.lower():
        raise ValueError("CCD SHA256 mismatch; refusing publisher-trusted pickle load")
    esm_hashes = snapshot_files(snapshot)
    # A local snapshot plus offline flags prevents implicit weight/model downloads.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(source))
    import nesso
    if Path(nesso.__file__).resolve().parent != source / "nesso":
        raise RuntimeError("Another Nesso source was already imported in this process")
    import numpy as np
    import torch
    import yaml
    from rdkit import rdBase
    from safetensors.torch import save_file
    from nesso.data.esm import setup_esm_model, extract_esm_embedding
    from nesso.data.featurizer import NessoFeaturizer
    from nesso.data.inference import InferenceDataset, inference_collate
    from nesso.data.types import Manifest
    from nesso.data.yaml_input import load_ccd_mol_dict, parse_yaml

    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    def seed_all() -> None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.random.default_generator.manual_seed(args.seed)
        rdBase.SeedRandomNumberGenerator(args.seed)

    seed_all()
    started = time.perf_counter()
    esm_model, tokenizer = setup_esm_model(str(snapshot), torch.device("cpu"))
    load_seconds = time.perf_counter() - started
    ccd_dict = load_ccd_mol_dict(ccd)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-prepare-", dir=output.parent) as tmp:
        stage = Path(tmp) / "result"
        stage.mkdir()
        reports = []
        selected = list(CASES) if args.case == "all" else [args.case]
        for name in selected:
            case_dir = stage / name
            case_dir.mkdir()
            paths = {n: case_dir / "processed" / n for n in
                     ("structures", "records", "rdkit_conformers", "esm_embeddings")}
            for path in paths.values():
                path.mkdir(parents=True)
            fixture = Path(__file__).with_name(name + ".yaml")
            schema = yaml.safe_load(fixture.read_text())
            seq = schema["sequences"][0]["protein"]["sequence"]
            if len(seq) != CASES[name]["protein_residues"]:
                raise ValueError(f"Unexpected fixture sequence length: {name}")
            input_path = case_dir / (name + ".yaml")
            input_path.write_bytes(fixture.read_bytes())
            seed_all()
            start = time.perf_counter()
            structure, record, _, _ = parse_yaml(input_path, paths["rdkit_conformers"], ccd_dict=ccd_dict)
            structure.dump(paths["structures"] / f"{record.id}.npz")
            record.dump(paths["records"] / f"{record.id}.json")
            manifest = Manifest([record])
            manifest.dump(case_dir / "processed" / "manifest.json")
            parse_seconds = time.perf_counter() - start
            start = time.perf_counter()
            embedding = extract_esm_embedding(seq, esm_model, tokenizer)
            if tuple(embedding.shape) != (1, len(seq) + 2, 1280) or embedding.dtype != torch.float32:
                raise ValueError(f"Unexpected native ESM layout/dtype: {embedding.shape}, {embedding.dtype}")
            mid = hashlib.md5(seq.encode()).hexdigest()
            esm_file = paths["esm_embeddings"] / f"{mid}.safetensors"
            save_file({"embeddings": embedding.contiguous()}, str(esm_file))
            esm_seconds = time.perf_counter() - start
            seed_all()
            start = time.perf_counter()
            dataset = InferenceDataset(manifest=manifest, target_dir=case_dir / "processed",
                featurizer=NessoFeaturizer(paths["esm_embeddings"], esm_emb_dim=1280, esm_num_layers=33),
                ligand_dir=paths["rdkit_conformers"], ccd_pkl=ccd, use_esm_all_layers=False)
            sample = dataset[0]
            if sample.get("exception"):
                raise RuntimeError(f"Native preprocessing failed for {name}: {sample}")
            batch = inference_collate([sample])
            features_seconds = time.perf_counter() - start
            tensors: dict = {}
            tree = _encode(batch, tensors)
            tensor_path = case_dir / "batch.safetensors"
            save_file(tensors, str(tensor_path))
            metadata = {"format": "compressme.nesso.fixture.v1", "case": name,
                "fixture": CASES[name], "input_yaml_sha256": sha256(input_path),
                "sequence_sha256": hashlib.sha256(seq.encode()).hexdigest(),
                "record_id": record.id, "batch_tree": tree,
                "tensor_file": {"name": tensor_path.name, "bytes": tensor_path.stat().st_size,
                                "sha256": sha256(tensor_path)},
                "tensors": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in tensors.items()},
                "esm_embedding_sha256": sha256(esm_file),
                "token_count": int(batch["token_pad_mask"].sum().item()),
                "atom_count": int(batch["atom_pad_mask"].sum().item()),
                "preprocessing_seconds": {"parse": parse_seconds, "esm": esm_seconds,
                                          "features_and_collate": features_seconds}}
            (case_dir / "batch.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
            restored = load_batch(case_dir)
            restored_tensors: dict = {}
            restored_tree = _encode(restored, restored_tensors)
            if restored_tree != tree or set(restored_tensors) != set(tensors):
                raise AssertionError("Prepared batch metadata round-trip mismatch")
            for key, value in tensors.items():
                if value.dtype != restored_tensors[key].dtype or value.shape != restored_tensors[key].shape:
                    raise AssertionError(f"Prepared tensor layout mismatch: {key}")
                if not torch.equal(value.view(torch.uint8), restored_tensors[key].view(torch.uint8)):
                    raise AssertionError(f"Prepared tensor byte round-trip mismatch: {key}")
            reports.append({k: v for k, v in metadata.items() if k not in ("batch_tree", "tensors")})
            print(json.dumps({"prepared": name, "tokens": metadata["token_count"],
                              "atoms": metadata["atom_count"]}), flush=True)
        report = {"format": "compressme.nesso.preparation.v1", "cases": reports,
            "source": {"repository": "https://github.com/recursionpharma/nesso", "revision": NESSO_REVISION,
                       "python_sha256": source_hashes},
            "esm": {"repository": ESM_REPOSITORY, "revision": ESM_REVISION,
                    "device": "cpu", "dtype": "torch.float32", "files": esm_hashes,
                    "model_load_seconds": load_seconds},
            "ccd": {"repository": "recursionpharma/nesso",
                    "revision": "499ed12b0343918ab01b2519226390cf8eca038a",
                    "bytes": ccd.stat().st_size, "sha256": ccd_hash},
            "seed": args.seed, "threads": args.threads,
            "versions": {p: importlib.metadata.version(p) for p in
                         ("torch", "transformers", "rdkit", "numpy", "safetensors", "lightning")},
            "scope": "No Nesso prediction executed. Genuine CPU ESM features and unchanged native parsing/featurization.",
            "reproducibility": "Benchmarks reload exact saved batch bytes. Independent preprocessing can differ across RDKit/platform versions; ETKDG options remain unchanged."}
        (stage / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")
        if output.exists():
            raise FileExistsError(f"Output appeared during preparation: {output}")
        stage.rename(output)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-cases", action="store_true", help="List fixture sizes/scope without ML imports")
    parser.add_argument("--case", choices=["all", *CASES], default="all")
    parser.add_argument("--nesso-source", help="Unmodified local checkout at the pinned Nesso revision")
    parser.add_argument("--esm-snapshot", help="Complete local pinned ESM Hugging Face snapshot (safetensors)")
    parser.add_argument("--ccd", help="Publisher-trusted local CCD pickle; verified before loading")
    parser.add_argument("--ccd-sha256", help="Expected publisher/download-manifest SHA256 of CCD")
    parser.add_argument("--output", help="New output directory (contains one directory per selected case)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.list_cases:
        print(json.dumps(CASES, indent=2))
        return 0
    for name in ("nesso_source", "esm_snapshot", "ccd", "ccd_sha256", "output"):
        if not getattr(args, name):
            parser.error(f"--{name.replace('_', '-')} is required for preparation")
    if args.threads < 1 or not (0 <= args.seed <= 2**31 - 1):
        parser.error("--threads must be positive and --seed must be in0..2**31-1")
    try:
        prepare(args)
    except (OSError, ValueError, RuntimeError, ImportError, subprocess.CalledProcessError) as exc:
        print(f"Preparation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
