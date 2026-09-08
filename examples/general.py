"""A complete compression/export example with no molecular dependencies."""
from pathlib import Path
import tempfile
import torch
from torch import nn
from compressme import Example, compile_affine, load, validate


def factory():
    return nn.Sequential(nn.Linear(82,512),nn.LayerNorm(512),nn.Linear(512,512)).eval()


def main():
    torch.set_num_threads(4)
    torch.manual_seed(54)
    model = factory()
    # Constructed example: these are not molecular checkpoint measurements.
    inputs = [Example((torch.randn(32,82),))]
    result = compile_affine(model,validation=inputs)
    assert result.report["validation"]["accepted"]
    with tempfile.TemporaryDirectory() as folder:
        result.save(Path(folder))
        restored = load(factory,folder).model
        assert validate(result.model,restored,inputs,relative_tolerance=0,absolute_tolerance=0)["accepted"]
    print({"parameters_before":result.report["parameters_before"],
           "parameters_after":result.report["parameters_after"],
           "original_output_agreement":result.report["validation"]["accepted"]})


if __name__ == "__main__": main()
