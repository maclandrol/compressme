"""Small CPU counterexamples to claims beyond the declared frozen scope."""
import json
import sys
from pathlib import Path
import torch
from torch import nn

sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
from request_constants import PreparedSingleConditioningPrefix


class Single(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_single = nn.LayerNorm(4)
        self.single_embed = nn.Linear(4, 4)
        self.disable_times = True
        self.transitions = nn.ModuleList([])

    def forward(self, times, trunk, inputs):
        result = self.single_embed(self.norm_single(torch.cat((trunk, inputs), -1)))
        for transition in self.transitions:
            result = transition(result) + result
        return result, None


torch.manual_seed(9)
trunk, inputs = torch.randn(2, 3, 2), torch.randn(2, 3, 2)
model = Single().eval().requires_grad_(False)
report = {}
with torch.no_grad():
    prepared = PreparedSingleConditioningPrefix(model, trunk, inputs)
    trunk.add_(torch.tensor([0.5, 0.0]))
    report['same_layout_input_mutation_error'] = float((model(None, trunk, inputs)[0] - prepared(None, trunk, inputs)[0]).abs().max())
    prepared = PreparedSingleConditioningPrefix(model, trunk, inputs)
    model.single_embed.bias.add_(1)
    report['source_weight_mutation_error'] = float((model(None, trunk, inputs)[0] - prepared(None, trunk, inputs)[0]).abs().max())
    prepared = PreparedSingleConditioningPrefix(model, trunk, inputs)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        report['autocast_source_dtype'] = str(model(None, trunk, inputs)[0].dtype)
        report['autocast_prepared_dtype'] = str(prepared(None, trunk, inputs)[0].dtype)

model.norm_single.train(True)
report['source_descendant_training_before_constructor'] = model.norm_single.training
PreparedSingleConditioningPrefix(model, trunk, inputs)
report['source_descendant_training_after_constructor'] = model.norm_single.training

trunk = trunk.clone().requires_grad_(True)
prepared = PreparedSingleConditioningPrefix(model, trunk, inputs)
prepared(None, trunk, inputs)[0].sum().backward()
try:
    prepared(None, trunk, inputs)[0].sum().backward()
    report['second_backward_error'] = None
except RuntimeError as error:
    report['second_backward_error'] = str(error).split('.')[0]
print(json.dumps(report, indent=2))
