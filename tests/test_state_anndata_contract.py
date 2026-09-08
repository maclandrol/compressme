import random

import pytest
import torch

np = pytest.importorskip("numpy")
from compressme.state_anndata import VocabularyKeys, _seeded


def test_vocabulary_contains_ordered_names_without_fabricating_vectors():
    names = VocabularyKeys(["TP53", "KRAS"])
    assert tuple(names.keys()) == ("TP53", "KRAS")
    assert len(names) == 2
    with pytest.raises(TypeError, match="metadata only"):
        names["TP53"]
    with pytest.raises(ValueError):
        VocabularyKeys(["TP53", "TP53"])


def test_explicit_cpu_seed_repeats_and_restores_rng_even_after_exception(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU encoding must not change GPU random state")

    monkeypatch.setattr(torch.cuda, "manual_seed_all", forbidden)
    monkeypatch.setattr(torch.mps, "manual_seed", forbidden)
    before = (random.getstate(), np.random.get_state(), torch.random.get_rng_state())
    with _seeded(97):
        first = (random.random(), np.random.random(3), torch.rand(3))
    with pytest.raises(RuntimeError, match="test exception"):
        with _seeded(97):
            second = (random.random(), np.random.random(3), torch.rand(3))
            raise RuntimeError("test exception")
    assert first[0] == second[0]
    assert np.array_equal(first[1], second[1])
    assert torch.equal(first[2], second[2])
    assert before[0] == random.getstate()
    current = np.random.get_state()
    assert before[1][0] == current[0]
    assert np.array_equal(before[1][1], current[1])
    assert before[1][2:] == current[2:]
    assert torch.equal(before[2], torch.random.get_rng_state())
