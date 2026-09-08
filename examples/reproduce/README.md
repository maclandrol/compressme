# Reproduction entry points

Start at the [Boltz-2 tutorial](../../docs/tutorials/boltz2.md) or [Mol-JEPA tutorial](../../docs/tutorials/moljepa.md). These scripts do not require committed artifacts, a `vendor/` directory, downloaded-code caches, or historical processed batches.

- `fetch.py`: explicitly download and hash-check pinned Mol-JEPA source/weights, Boltz checkpoints or molecular assets. Downloaded Python is not imported here.
- `export_shared_tensors.py`: convert the two exact Boltz checkpoint ZIPs through `static_metadata.py`'s inert symbolic opcode reader. It verifies full publisher hashes before interpreting metadata, never executes pickle globals, and refuses other layouts.
- `safe_state.py`: reconstruct original named tensors from that pure safetensors/JSON state.
- `boltz2.py`: build or verify both complete native models with a freshly parsed fixture, full schedules, original self-repeat and complete output byte comparisons.
- `moljepa.py`: build a SMILES-only artifact with all outputs retained; verify a fresh reload; reproduce the recorded timing protocol or run an explicitly custom protocol.

The molecule parser still uses trusted upstream RDKit pickle assets after their archive and file hashes have been verified. The tutorials explain that separate trust boundary. Model construction and artifact loading execute the pinned local architecture code. This is not a sandbox for arbitrary third-party Python or a general pickle loader.

`moljepa-benchmark.json` records the exact historical molecules, variants, call settings, warmups and shuffling protocol. The `*-macos-tested.txt` files record successful dependency versions; they do not promise identical timing or future wheel availability. Source/weight licences remain upstream licences. The Mol-JEPA licence text and attribution are retained for the generated artifact.

Run the lightweight helper checks without downloading any weights:

```sh
python -m pip install -e '.[torch,packing,test]'
python -m pytest -q examples/reproduce/test_helpers.py
```

All source/cache/artifact/output locations are command-line arguments or paths relative to this checked-out repository. New output directories and reports are required. Full reproduction writes several gigabytes; keep the generated `work/` tree out of version control.

[reproduction-smoke.json](reproduction-smoke.json) records the fresh actual-weight CPU conversion/build/reload results and short benchmark-command smoke. Its one timing round is an execution check, not performance evidence. [SHA256.json](SHA256.json) enumerates the included helper/fixture files.
