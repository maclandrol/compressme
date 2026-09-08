# NovoMolGen: reproduce the complete token-input experiments

This directory records native NovoMolGen 32M AtomWise experiments against the
published float32 checkpoint. It is research evidence and code, not a production
compressed model or preservation of arbitrary `inputs_embeds` calls.

The main result is bitwise agreement on the recorded CPU and MPS output checks
after precomputing the first token normalization and Q/K/V projections at several
numerical execution shapes. All original residual embeddings remain. CPU stores
two tables and saves 1.676% of parameters; MPS stores three and saves 1.267%.
The original manual-layout timing is essentially neutral. See
[the detailed evidence](manual/CANDIDATE_EVIDENCE.md) and
[project explanation](../../docs/novomolgen.md).

## Files and scope

- `original_native/` contains a portable, hash-checked original-model smoke with
  twelve API stages. Its historical CPU/MPS × eager/SDPA reports remain unchanged.
- [`generic/`](generic/README.md) contains the reusable compiler's complete-model
  probes after lookup save/reload, separate explicit byte checks, timing samples
  and a retained noisy-case recheck. Its scripts use the installed package and
  verify the audited compiler source hash. Seventeen files are indexed by hash.
- `manual/` preserves the initial failed single-table experiment, failed
  cross-device and two-table MPS proposals, successful manual CPU/MPS prototypes,
  boundary checks and every timing sample. `EVIDENCE_FILES.json` verifies the
  thirty archived files. Its original README retains the experiment's machine
  paths as historical context; use the explicit-input commands below elsewhere.
  Its [terminology clarification](manual/VALIDATION_NOTE.md) distinguishes the
  original zero-error comparisons from the later explicit byte checks.
- The generic compiler itself is in `src/compressme/finite_fanout.py` at project
  root. It contains no NovoMolGen-specific module names, token IDs or thresholds.

No model-weight payload is copied into the experiment archive. Local lookup
artifacts generated during a replay contain derived trained coefficients and
retain the applicable upstream model terms.

## Environment and inputs

Use a separate Python 3.12 environment with the recorded dependencies in
`original_native/requirements.txt`, plus an installation of this project. The
reference requires Transformers 4.46.2, tokenizers 0.20.3 and the pinned native
Llama source. Avoid installing these pinned dependencies into the separate State
environment, whose model requires another Transformers version.

Supply `model.safetensors`, `config.json` and the tokenizer/generation JSON files
from the native `hf-checkpoint` revision of
[`chandar-lab/NovoMolGen_32M_SMILES_AtomWise`](https://huggingface.co/chandar-lab/NovoMolGen_32M_SMILES_AtomWise/tree/dcd3f59261bebf84142c13617d4e129b4b0d0fdc).
That is a revision name, not a subdirectory within that revision. The project
also retains the small config/tokenizer inputs in its pinned audit directory.

First check the portable original model; all input, tokenizer and runtime hashes
are verified before constructing it:

```bash
python original_native/native_smoke.py \
  --checkpoint /path/to/model.safetensors \
  --config /path/to/config.json \
  --tokenizer-source /path/to/native-config-and-tokenizer-files \
  --device cpu --attention eager \
  --output /path/to/new-original-cpu.json
```

For the manual CPU compressed experiment, run from this directory with an
explicit private cache directory and local-only Hugging Face settings:

```bash
HF_HOME=/path/to/private-experiment-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python manual/singleton/probe.py \
  --checkpoint /path/to/model.safetensors \
  --config /path/to/config.json \
  --device cpu --backend eager --table-device cpu \
  --output /path/to/new-compressed-cpu.json
```

The tokenizer files must be beside the supplied config. Use
`manual/regimes/probe.py --device mps --table-device mps` for the recorded MPS
policy. MPS requires actual Apple GPU access. The matching `boundary_probe.py`
checks the table-selection boundaries and strided/composite inputs.

Main compression probes use eager attention to compare requested attention maps.
The original model's broader SDPA smoke is separate. Passing outputs on one
device does not qualify tables for another device, and matching generation
alone does not establish agreement of hidden states or caches.

For the reusable compiler, replace `manual/singleton/probe.py` in the command
above with `generic/probe.py` and add
`--lookup-artifact /path/to/new-lookup-artifact`. Use `--device mps --table-device
mps` for its MPS counterpart. Keep an explicit private `HF_HOME` and local-only
flags; the standalone original-model smoke creates its own private cache.
The optional RTK wrapper shown in some archived command examples is unnecessary
when running the same Python executable directly.
