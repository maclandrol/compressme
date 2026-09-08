import copy
import os
import sys
from pathlib import Path
import pytest
import torch

pytest.importorskip("molfeat")
from compressme.moljepa_io import moljepa_factory
from compressme.sparse_graph import SparseMolJEPAFeaturizer, smiles_to_batch


@pytest.fixture(scope="module")
def source():
    architecture = Path(os.environ.get(
        "COMPRESSME_MOLJEPA_SOURCE",
        str(Path(__file__).resolve().parents[1]/"vendor"/"moljepa"),
    ))
    if not architecture.is_dir():
        pytest.skip("Optional pinned Mol-JEPA source is absent; set COMPRESSME_MOLJEPA_SOURCE after the tutorial download")
    model = moljepa_factory(architecture)
    return sys.modules[type(model).__module__], model


def test_sparse_features_and_batch_reconstruction_match_upstream(source):
    module, model = source
    smiles = ["CCO", "[He]", "[Na+].[Cl-]", "N->[Cu+2]<-N", "C~C", "C1CC2CCC1C2", "[2H]O[2H]"]
    fast = SparseMolJEPAFeaturizer(module.GraphFeaturizer)
    original = model.smiles_to_batch(smiles)
    actual = smiles_to_batch(smiles,featurizer=fast,data_class=module.MultimodalData)
    assert set(original.keys()) == set(actual.keys())
    for key in original.keys():
        assert original[key].dtype == actual[key].dtype
        assert torch.equal(original[key],actual[key])
    for a,b in zip(original.to_data_list(),actual.to_data_list()):
        for key in a.keys():
            assert torch.equal(a[key],b[key])
    assert fast.fallback_count == 0


@pytest.mark.parametrize("smiles",["", "C1CC", None, 123])
def test_invalid_input_retains_original_exception(source,smiles):
    module,_ = source
    fast = SparseMolJEPAFeaturizer(module.GraphFeaturizer)
    with pytest.raises(Exception) as original:
        module.GraphFeaturizer().encode(smiles)
    with pytest.raises(type(original.value)) as actual:
        fast.encode(smiles)
    assert str(original.value) == str(actual.value)


def test_descriptor_functions_do_not_prevent_model_copy(source):
    module,_ = source
    fast = SparseMolJEPAFeaturizer(module.GraphFeaturizer)
    clone = copy.deepcopy(fast)
    for key,value in fast.encode("c1ccccc1").items():
        assert torch.equal(value,clone.encode("c1ccccc1")[key])
