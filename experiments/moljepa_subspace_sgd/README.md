# Reproduce the archived component experiment

This directory contains an isolated Mol-JEPA first-attention SGD experiment.
`results.json` is the archived evidence; new runs default to `replay-results.json`.
See `note.md` for the mathematical contract, storage counts and failed float32
raw-logit stress gate. This is not a production model loader or training adapter.

Supply local files from `Flogrammer/Mol-JEPA` at revision
`4c912b450175f31b5ba913a5dc921c03b27b985a`:

```bash
python probe.py \
  --checkpoint /path/to/model.safetensors \
  --source-file /path/to/modeling_moljepa.py \
  --check-inputs-only

python probe.py \
  --checkpoint /path/to/model.safetensors \
  --source-file /path/to/modeling_moljepa.py \
  --output /path/to/new-results.json
```

Both files are checked against pinned SHA256 hashes **before** the extracted
featurizer source executes. A deliberately different revision requires explicit
`--expected-checkpoint-sha256` and `--expected-source-sha256` values and produces
a separate experiment. `--steps` defaults to20 and `--threads` to4. Existing
outputs are refused unless `--overwrite` is explicitly supplied.

The tested environment used Python3.12, PyTorch2.14.0 and
torch-geometric2.8.0.post1, plus safetensors, molfeat and RDKit. The source
featurizer imports its ordinary molfeat dependencies. No original full-model
Python import, remote-code download, original optimizer checkpoint, GPU, or
`torch.load` is needed. The checkpoint and source carry upstream
CC-BY-NC-4.0 terms; they are not copied into this experiment directory.

`provenance.json` records the exact input identities. The full evidence is not
overwritten during hash-only verification. The script contains no hardcoded
machine, environment, checkpoint or source paths.
