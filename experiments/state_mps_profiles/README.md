# State SE MPS profile diagnostics — stopped before full-domain work

This bounded exploration used the original pinned SE-100M checkpoint and
unmodified encoder on MPS. It does not supply a compressed GPU model or change
the production adapter's CPU-only guard. Work stopped when Boltz-2 became the
user's priority.

The earlier native final raw lookup missed the local maximum-absolute gate by
reaching 1.0729e-5; CPU-built final tables also failed complete MPS outputs.
Simply transferring NovoMolGen's profile thresholds was therefore not assumed.

The new diagnostics inspect 16 fixed real gene rows over every count 1–128,
plus 257 and 1,024. They compare actual tensor bytes, separately from numerical
errors. Repeated identical genes at different positions and mixed-token chunks
showed no byte differences from their matching templates on these probes.

Raw final encoder outputs have five sampled classes: counts 1–9, 10–12,
13–24, 25–49 and 50+. Normalized outputs have the same classes plus a distinct
singleton normalization case. Normalized input vectors themselves differ only
between singleton and multiple-row evaluation on these probes. This is a
measured shape classification, not a theorem or full-vocabulary validation.

Source sequencing matters: normalization sees B×T genes, whereas the encoder
sees B×(T+1) after dataset-token insertion and CLS replacement. Raw task genes
see B×Q. A profile selected solely from the unmodified gene-ID count would not
represent that sequence of operations.

The arbitrary raw 5,120-vector encoder must remain. Original model parameters
number 212,038,824, including a 101,324,800-value raw gene table. Each complete
final profile table would hold 19,790×1,024 = 20,264,960 FP32 values. Even five
raw plus five normalized dense tables would grow the complete representation
to 313,363,624 values (+47.8%), before additional metadata. Including the sixth
normalized table would grow it further. This rejects naive dense profile
storage as a compression result; lossless profile differences were not tested
on the complete vocabulary here.

Files:

- `local_profile_probe.py` / `local-mps.json`: component stages and realistic
  raw/normalized insertion layouts, including strided IDs and length 2,048.
- `regime_catalog.py` / `catalog-mps.json`: every count 1–128 and larger probes,
  deduplicated byte signatures and repeated/mixed-position checks.
- `sampled-profile-tables.safetensors`: only the 16 sampled gene IDs and their
  representative final outputs, for studying lossless storage ideas.

Both scripts accept checkpoint, architecture, original-loader, device and output
paths. They verify the checkpoint and vendored architecture hashes using the
existing State source manifest. They were run with Python 3.12.14, PyTorch 2.14.0
and the project's `.venv-state` runtime, with native MPS access. They do not
benchmark speed, enumerate all 19,790 genes, or validate a whole-model candidate.

