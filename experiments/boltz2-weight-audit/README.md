# Boltz-2 exact compression audit

The pinned source and official checkpoint identities are recorded in `source-inventory.json` and `range-summary.json`. This archive separates exact tensor-storage evidence, local real-weight bank tests and native runtime evidence. The original Python model has now run successfully on CPU and MPS; complete weight-sharing API gates are maintained separately in the runtime audit. No checkpoint pickle was unpickled: the allowlisted opcode interpreter creates inert metadata records; tensor payloads are read from ZIP entries.

## Actual-weight component result

Both full publisher SHA256 values were verified. All 5,019 common named tensors are byte-identical, including the entire shared pairformer, structure and confidence modules. Exact byte comparisons confirm every reused hash. The two independently byte-deduplicated inference states total 1,021,780,358 float32 values; a joint state stores 515,466,951, saving 49.5521% (2,025,253,628 tensor bytes). The shared safetensors file is 2,062,620,108 bytes including its header. This preserves both distinct models and all heads as a joint distribution; it does not reduce either model’s arithmetic. See `export-report.json` in the artifact and `cross-model-modules.json` here.

The 24 token pair-bias branches apply LayerNorm 128 then Linear 128→16 to the same input. Six atom branches share 16-wide pair input. Their LayerNorm affine parameters differ. With `u=LN_nonaffine(x)`, rewrite each branch as `W_i LN_i(x)=(W_i diag(gamma_i))u+W_i beta_i`. This shares the normalization and concatenates the folded projections, saving 5,928 float32 values (23,712 bytes). A statistics-only variant keeps each original affine and linear operation.

Actual conf and affinity weights passed 84 CPU checks (float32/64) and 42 MPS float32 checks, including ordinary, large-scale, large-offset, nearly constant, constant, empty and singleton inputs. Float32 folded max absolute difference was 8.5831e-6 on both devices. Every check passed `abs(error)<=1e-5+1e-5*abs(reference)`. Statistics-only outputs were bitwise identical. Float64 folded max was 1.7764e-14. These are local observations, not whole-model coordinate guarantees.

The source computes these biases once before sampling and reuses them through denoising. This does not remove 24 normalizations on every diffusion step. A separate 48-condition-normalization fanout in the 24-layer score transformer remains a source-only candidate.

## Storage and output boundaries

The confidence file is 2,286,561,469 bytes, with 521,047,871 named float32 state values and 50,052,584 extra optimizer/callback values (~200.21 MB). Source gate aliases contribute 14,322,816 repeated named values (~57.29 MB); all 66 gate aliases are byte-identical in the full comparison, and one actual gate plus strict-load precedence passed. Removing training state preserves inference scope, not training resumption.

The affinity file is 2,062,139,170 bytes, with 529,378,245 named values but 515,055,429 unique storage values. It already deduplicates 66 gate alias pairs. Keep both distinct affinity heads, confidence used to select the highest-IPTM structure, all member outputs and probability averaging. Native affinity inference reruns trunk/structure; sharing equal parameters does not justify reusing activations.

Actual SiLU/SwiGLU transitions, sigmoid condition gates and continuous geometry block simple affine compression. Token/atom heads have 48/32 dimensions; Mol-JEPA's 512-wide-head opportunity does not transfer. Categorical atom/relative-position columns can become gathers while retaining continuous coordinates/charges; these principally save allocations/arithmetic, not weight count.

## Reproduction

Use Python 3.10–3.12. The small actual-weight bank and alias fixtures are included with their MIT licence and tensor hashes, so local probes need no download. Create an isolated environment and install requirements-probes.txt. Metadata/download/export scripts otherwise use the standard library. Optional full reproduction downloads 4.35 GB of official checkpoints; shared export needs about 2.1 GB of additional disk space. No full checkpoints, pickle payloads, ZIP prefixes, range caches, molecular assets or credentials are included here.

```sh
python -m venv /path/to/probe-env
/path/to/probe-env/bin/python -m pip install -r requirements-probes.txt
/path/to/probe-env/bin/python verify_archive.py
/path/to/probe-env/bin/python probe_bank.py --directory . --device cpu --output /path/to/new-bank-cpu.json
/path/to/probe-env/bin/python probe_bank.py --directory . --device mps --output /path/to/new-bank-mps.json
/path/to/probe-env/bin/python probe_alias.py --directory . --output /path/to/new-alias.json

# Optional: reproduce the full exact-byte comparison and joint pure tensor export.
python download_checkpoints.py --directory /path/to/originals
python export_shared_tensors.py --checkpoints /path/to/originals --output /path/to/shared-state
```

Download and export both verify complete publisher SHA256 values. The exporter interprets metadata inertly, refuses layouts other than full contiguous float32 storage, and stores each unique byte string once. Hash matches are verified by exact byte comparison. `safe_state.load_tensor_state(directory, checkpoint)` reconstructs original names/shapes and JSON configuration from pure safetensors. Instantiation and strict loading belong to the separate pinned Boltz runtime baseline.

The selected manifests retain their historical range-only validation labels. The later full-file checks and fresh safe-state reload establish stronger provenance, recorded in download-verification.json and fresh-safe-state-load.json. source-inventory.json lists source lines and consumer hazards. SHA256.json enumerates every included file except itself; verify_archive.py checks the inventory and hashes. New probe reports require explicit output paths and refuse existing files, preserving the bundled evidence. The hash manifest is an integrity record, not a cryptographic signature.

## Native runtime limits

The original Python Boltz2 model, native preprocessing and writers succeeded on one tiny protein–ligand fixture on both CPU and MPS using float32 and the published sampling settings; runtime-status.json records the scope. This does not establish cross-device bitwise equality, biological accuracy, or speedup. Complete compression/weight-sharing parity gates belong to the separate runtime audit.

The CLI lists gpu/cpu/tpu rather than a literal mps option, but Lightning 2.5.0 explicitly selects MPS for the generic gpu accelerator when MPS is available. The isolated resolver probe confirmed this on the tested Mac. Therefore that CLI spelling alone is not a Mac blocker. lightning-accelerator-result.json records the separate resolver check; it is not a complete CLI invocation. To repeat it in a Lightning 2.5.0 environment, run probe_lightning_accelerator.py with an explicit --output path.

Disable optional CUDA kernels for CPU/MPS. Triangle kernels read child weight attributes directly, and Transition's chunked path slices its linear weights; child-forward-only replacements are bypassed. Preserve masks, recycling, sampling, confidence and both distinct affinity heads when testing any rewrite.

Pinned source: https://github.com/jwohlwend/boltz/tree/b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc
Official weights: https://huggingface.co/boltz-community/boltz-2/tree/6fdef46d763fee7fbb83ca5501ccceff43b85607
