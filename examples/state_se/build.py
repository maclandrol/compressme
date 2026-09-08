"""Build the validated CPU State SE artifact with no original weights at reload."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from unittest.mock import patch
import torch
from compressme import CompressionResult
from compressme.finite_lookup import compile_finite_lookup, RowNormalize
from original import load_state_se as load_original
from adapter import StateSETokenAdapter, FinalBranchLookup
from audit import audit
from checks import output_fingerprints, benchmark
from compressme import load_state_se


def main(output, architecture, checkpoint, *, measure=False, tokenizer_source=None):
    digest = hashlib.sha256()
    with Path(checkpoint).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != "7ee823583ad10c6a12befba6df12dfcd61ac0972deab57be3b4db24a0045af86":
        raise ValueError("Expected the pinned SE-100M model.safetensors; use the general compiler for other weights")
    torch.set_num_threads(4)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(architecture, root/"architecture", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    tokenizer_source = Path(tokenizer_source) if tokenizer_source is not None else Path(__file__).resolve().parents[2]/"vendor/state_se_tokenizer"
    from compressme.state_anndata import pinned_loader
    pinned_loader(tokenizer_source)
    shutil.copytree(tokenizer_source, root/"tokenizer_source", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    original = load_original(checkpoint, config=Path(architecture)/"config.json", source_root=architecture)
    raw = compile_finite_lookup(torch.nn.Sequential(original.pe_embedding, original.encoder).eval(),
                               input_contract="token_indices_only", absolute_tolerance=1e-5, relative_tolerance=1e-5)
    normalized = compile_finite_lookup(torch.nn.Sequential(original.pe_embedding, RowNormalize(), original.encoder).eval(),
                                      input_contract="token_indices_only", absolute_tolerance=1e-5, relative_tolerance=1e-5)
    assert raw.report["status"] == normalized.report["status"] == "accepted_on_full_domain_validation"
    model = StateSETokenAdapter(original, FinalBranchLookup(raw.model, normalized.model))
    model.post_projection = torch.nn.Identity().eval()
    model.constant_mode = "source_shape"
    model.set_gene_vocabulary(json.loads((root/"architecture/vocabulary.json").read_text()))
    report = audit(original, model, {"method": "two_generic_finite_lookups", "raw": raw.report,
                                    "normalized": normalized.report}, extended=True)
    report.update({"target": "State SE-100M", "validated_devices": ["cpu"],
                   "validated_dtype": "float32", "source_commit": "9bbfe78a434a55205e4de834e1ea99f85f7a3add",
                   "checkpoint_repository": "arcinstitute/SE-100M",
                   "checkpoint_revision": "bc72702320639217128df42673e94a9658f67d24",
                   "checkpoint_sha256": "7ee823583ad10c6a12befba6df12dfcd61ac0972deab57be3b4db24a0045af86",
                   "source_unique_parameter_bytes": sum(p.numel()*p.element_size() for p in original.parameters()),
                   "compiled_unique_parameter_bytes": sum(p.numel()*p.element_size() for p in model.parameters()),
                   "contract": "Original numerical gene-ID batch API and arbitrary raw-vector forward/raw gene_embedding_layer retained; inference only",
                   "gene_names": "19790 ordered known gene names; hash-verified tokenizer included, AnnData checks recorded separately",
                   "raw_protein_dictionary": "Optional external dictionary required for original model name helper and its raw sum()==0 guard; not stored in this artifact",
                   "mps_status": "Rejected at unchanged local or complete-output numerical gates; token route rejects MPS even after model.to('mps')",
                   "constant_folding": "Bounded shape-matched cache stores only parameter-derived CLS/dataset rows; no cell inputs or outputs cached"})
    fingerprints = output_fingerprints(model, extended=True)
    (root/"reload-fingerprints.json").write_text(json.dumps(fingerprints, indent=2)+"\n")
    if measure:
        report["benchmark"] = benchmark(original, model, rounds=5)
    CompressionResult(model, report).save(root, packing=True)
    # Free both heavy models before exercising the independent local factory.
    del original, model, raw, normalized
    with patch("torch.load", side_effect=AssertionError("Original checkpoint access forbidden")):
        reloaded = load_state_se(root)
    actual = output_fingerprints(reloaded, extended=True)
    assert actual == fingerprints, "Portable reload changed outputs"
    assert len(reloaded.gene_names) == 19790 and reloaded.gene_to_index[reloaded.gene_names[0]] == 0
    try:
        reloaded.gene_to_index["new_gene"] = 0
        raise AssertionError("Vocabulary map should be read-only")
    except TypeError:
        pass
    try:
        load_state_se(root, device="mps")
        raise AssertionError("MPS loader should reject this artifact")
    except ValueError:
        pass
    report.update(packed_weights_bytes=(root/"model.cmppack").stat().st_size,
                  portable_directory_bytes=sum(p.stat().st_size for p in root.rglob("*") if p.is_file()),
                  reload_fingerprints_bitwise_equal=True, reload_forbids_torch_load=True,
                  output_comparisons=105, known_gene_names=19790)
    (root/"build-result.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({key: report[key] for key in ("parameters_before", "parameters_after", "packed_weights_bytes",
                      "reload_fingerprints_bitwise_equal", "output_comparisons", "known_gene_names")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(Path(__file__).resolve().parents[2]/"artifacts/state-se-100m-cpu"))
    parser.add_argument("--architecture", default=str(Path(__file__).resolve().parents[2]/"vendor/state_se"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--tokenizer-source", help="Pinned tokenizer source; defaults to vendor/state_se_tokenizer")
    args = parser.parse_args()
    main(args.output, args.architecture, args.checkpoint, measure=args.benchmark, tokenizer_source=args.tokenizer_source)
