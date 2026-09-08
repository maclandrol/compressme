import copy
import json
from unittest.mock import patch
import pytest
import torch
from torch import nn
from compressme.packed_embedding import PackedFrozenEmbedding
from compressme.state_se_io import _GeneIndex, _PackedProteinRows, load_state_se


def test_packed_name_mapping_and_deepcopy():
    model = nn.Module()
    source = nn.Embedding(10, 8).eval().requires_grad_(False)
    model.pe_embedding = PackedFrozenEmbedding(source, block_rows=4)
    names = tuple(f'gene{i}' for i in range(10))
    model.gene_to_index = _GeneIndex(names)
    model.protein_embeds = _PackedProteinRows(model.pe_embedding, model.gene_to_index)
    assert list(model.protein_embeds) == list(names)
    assert 'missing' not in model.protein_embeds
    with pytest.raises(KeyError): model.protein_embeds['missing']
    with pytest.raises(TypeError): model.gene_to_index['gene1'] = 3
    for i, name in enumerate(names):
        assert torch.equal(model.protein_embeds[name], source.weight[i])
    cloned = copy.deepcopy(model)
    assert cloned.protein_embeds._embedding is cloned.pe_embedding
    assert cloned.pe_embedding.payload.data_ptr() != model.pe_embedding.payload.data_ptr()
    assert torch.equal(cloned.protein_embeds['gene3'], source.weight[3])


def test_old_finite_artifact_still_rejects_mps(tmp_path):
    (tmp_path/'manifest.json').write_text(json.dumps({'report':{'validated_devices':['cpu']}}))
    with patch('compressme.serialization.load', side_effect=AssertionError('Must reject before loading')):
        with pytest.raises(ValueError, match='did not pass'):
            load_state_se(tmp_path, device='mps')
