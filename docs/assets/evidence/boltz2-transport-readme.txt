# Lossless transport copy of the Boltz-2 shared bundle

This directory preserves the original manifest and JSON architecture, with its
tensor file stored as CMPRPACK v1. It is not directly loadable until restored.
The project also includes the ready-to-load plain bundle at `../boltz2-shared`.
Keeping both copies uses more disk; this transport copy is for transfer/archive.

Restore using the public file codec with its optional packing dependencies:

```python
from compressme import unpack_file, load_boltz2

unpack_file(
    "artifacts/boltz2-shared-transport/model.cmprpack",
    "artifacts/boltz2-shared-transport/model.safetensors",
    max_output_bytes=3 * 1024**3,
)
bundle = load_boltz2("artifacts/boltz2-shared-transport", device="mps")
```

Use an environment with both the pinned Boltz dependencies and
`compressme[packing]`, or restore with the project's primary `.venv` first and
load with `.venv-boltz`. Unpacking refuses an existing output unless explicitly
authorized by `overwrite=True`. It verifies the complete restored checksum.

Packed tensor file: 1,759,531,547 bytes.
Packed SHA256: `fe19430c29b952aa39c131ff70f2b194cacf62e206c8e1b1dc50dc7ef39a1674`.
Restored tensor file: 2,062,669,224 bytes.
Restored SHA256: `1e6904266eccc8826225e828159eaea252888cdd6ea787fac79611f4e013d434`.

Every restored byte was compared against the complete source. This is a 14.696%
file-size reduction from the already shared weights. Resident model storage,
all values and computation are unchanged by packing. The manifest and
architecture are additional files; inspect `FILES.json` for their exact sizes.
Full numerical scope and source attribution are in `../../docs/boltz2.md`.
