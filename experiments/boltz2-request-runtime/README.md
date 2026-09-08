# Boltz-2 compute experiment evidence

`lean/README.md` contains the final complete-output gates, timing result and reproducible commands. `guarded/README.md` preserves the earlier guarded prototype and its inconclusive timing, including an explicitly interrupted CPU trial. No failed or negative experiment was removed.

The final result is byte-preserving compute reuse, with no useful MPS gain on the tested small complex and a small CPU affinity improvement over three pairs. It remains optional research, not evidence of general acceleration. The separate Boltz shared-weight artifact and its storage reduction are not changed by this experiment.

The three root support scripts are copied from the native Boltz runtime evidence. Supply that package via `--native-dir` and the compiled bundle via `--artifact`. Native inputs, model weights and environment are deliberately not duplicated here. The original source SHA and every output leaf are recorded in the detailed JSON reports. `FILES.json` records exact archived file hashes, including this README.
