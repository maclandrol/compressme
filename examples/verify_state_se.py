"""Fresh-process reload using only the portable artifact and compressme."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch
import torch
from compressme import load_state_se
from _state_se_checks import output_fingerprints

parser = argparse.ArgumentParser()
parser.add_argument("--artifact", default=str(Path(__file__).resolve().parents[1]/"artifacts/state-se-100m-cpu"))
parser.add_argument("--output", default=str(Path(__file__).resolve().parents[1]/"benchmarks/state_se_reload.json"))
args = parser.parse_args()
torch.set_num_threads(4)
root = Path(args.artifact)
with patch("torch.load", side_effect=AssertionError("Original checkpoint/dictionary access forbidden")):
    model = load_state_se(root)
expected = json.loads((root/"reload-fingerprints.json").read_text())
actual = output_fingerprints(model, extended=True)
assert expected == actual
assert len(model.gene_names) == 19790
assert model.gene_to_index[model.gene_names[-1]] == 19789
try:
    model.get_gene_embedding([model.gene_names[0]])
    raise AssertionError("Name helper must require the original dictionary")
except ValueError as error:
    assert "dictionary" in str(error)
try:
    load_state_se(root, device="mps")
    raise AssertionError("CPU-only loader must reject MPS")
except ValueError:
    pass
model.supported_token_devices = ()
try:
    model.get_gene_embedding_by_id(torch.tensor([[0]]))
    raise AssertionError("Runtime token device guard was bypassed")
except ValueError:
    pass
result = {"fresh_process": True, "torch_load_forbidden": True, "complete_output_fingerprints_equal": True,
          "cases": len(actual), "output_comparisons": sum(len(x["outputs"]) for x in actual),
          "parameters": sum(p.numel() for p in model.parameters()), "known_gene_names": len(model.gene_names),
          "mps_loader_rejected": True, "runtime_token_device_guard_checked": True}
Path(args.output).write_text(json.dumps(result, indent=2)+"\n")
print(json.dumps(result, indent=2))
