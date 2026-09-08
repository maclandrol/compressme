"""Small routing/cleanup tests, distinct from real-weight numerical gates."""
import unittest
from unittest.mock import patch
import torch
from torch import nn
from boltz.model.modules.diffusionv2 import AtomDiffusion,DiffusionModule
from boltz.model.modules.encodersv2 import SingleConditioning
from boltz.model.modules.transformersv2 import AtomTransformer
from request_constants import constant_conditioning_sampling_scope,PreparedSingleConditioningPrefix

def fixture():
    score=DiffusionModule.__new__(DiffusionModule);nn.Module.__init__(score)
    score.single_conditioner=SingleConditioning(sigma_data=16,token_s=2,dim_fourier=4,num_transitions=0)
    for name,child in [('atom_attention_encoder','atom_encoder'),('atom_attention_decoder','atom_decoder')]:
        outer=nn.Module();setattr(outer,child,AtomTransformer(attn_window_queries=2,attn_window_keys=4,dim=4,dim_single_cond=4,depth=1,heads=1));setattr(score,name,outer)
    diffusion=AtomDiffusion.__new__(AtomDiffusion);nn.Module.__init__(diffusion);diffusion.score_model=score;diffusion.eval().requires_grad_(False)
    kwargs={'s_trunk':torch.randn(1,2,2),'s_inputs':torch.randn(1,2,2),'diffusion_conditioning':{'c':torch.randn(1,4,4)}}
    return diffusion,kwargs

def local_forward(self,s_trunk,s_inputs,diffusion_conditioning,times,multiplicity=1):
    return self.single_conditioner(times,s_trunk.repeat_interleave(multiplicity,0),s_inputs.repeat_interleave(multiplicity,0))[0]

class Guards(unittest.TestCase):
    def run_case(self,mutation):
        with torch.no_grad():
            model,kwargs=fixture();score=model.score_model;original_modules={n:id(m) for n,m in score.named_modules()}
            def local_sample(self,**inputs):
                call=dict(inputs,times=torch.zeros(1),multiplicity=1)
                first=self.score_model(**call);mutation(self.score_model,inputs)
                second=self.score_model(**call)
                return first,second
            with patch.object(DiffusionModule,'forward',local_forward),patch.object(AtomDiffusion,'sample',local_sample):
                with constant_conditioning_sampling_scope(model) as stats:outputs=model.sample(**kwargs)
                expected=local_forward(score,**kwargs,times=torch.zeros(1),multiplicity=1)
            self.assertTrue(torch.equal(outputs[1],expected));self.assertEqual(stats['requests'][0]['optimized_score_calls'],1);self.assertEqual(stats['requests'][0]['fallback_score_calls'],1);self.assertTrue(stats['requests'][0]['cache_cleared']);self.assertNotIn('forward',score.__dict__);self.assertNotIn('sample',model.__dict__)
            self.assertEqual(original_modules,{n:id(m) for n,m in score.named_modules()})
    def test_rebound_frozen_bias(self):
        self.run_case(lambda score,inputs:setattr(score.single_conditioner.single_embed,'bias',nn.Parameter(score.single_conditioner.single_embed.bias+1,requires_grad=False)))
    def test_rebound_parameter_storage(self):
        def mutate(score,inputs):score.single_conditioner.single_embed.bias.data=(score.single_conditioner.single_embed.bias+1).clone()
        self.run_case(mutate)
    def test_changed_input_version(self):
        self.run_case(lambda score,inputs:inputs['s_trunk'].add_(1))
    def test_exception_cleanup(self):
        with torch.no_grad():
            model,kwargs=fixture();score=model.score_model;identities={n:id(m) for n,m in score.named_modules()}
            def fail(self,**inputs):
                self.score_model(**inputs,times=torch.zeros(1));raise RuntimeError('planned')
            with patch.object(DiffusionModule,'forward',local_forward),patch.object(AtomDiffusion,'sample',fail):
                with self.assertRaisesRegex(RuntimeError,'planned'):
                    with constant_conditioning_sampling_scope(model) as stats:model.sample(**kwargs)
            self.assertEqual(identities,{n:id(m) for n,m in score.named_modules()});self.assertTrue(stats['requests'][0]['cache_cleared']);self.assertNotIn('forward',score.__dict__);self.assertNotIn('sample',model.__dict__)
    def test_prepared_forward_rejects_grad_and_amp_change(self):
        with torch.no_grad():
            model,inputs=fixture();source=model.score_model.single_conditioner;prepared=PreparedSingleConditioningPrefix(source,inputs['s_trunk'],inputs['s_inputs'])
            with torch.autocast('cpu',dtype=torch.bfloat16):
                with self.assertRaisesRegex(ValueError,'Autocast'):prepared(torch.zeros(1),inputs['s_trunk'],inputs['s_inputs'])
        with self.assertRaisesRegex(ValueError,'no-grad'):prepared(torch.zeros(1),inputs['s_trunk'],inputs['s_inputs'])
    def test_training_descendant_refused_without_mutation(self):
        with torch.no_grad():
            model,inputs=fixture();source=model.score_model.single_conditioner;source.norm_single.train()
            with self.assertRaisesRegex(ValueError,'descendant'):PreparedSingleConditioningPrefix(source,inputs['s_trunk'],inputs['s_inputs'])
            self.assertTrue(source.norm_single.training)
if __name__=='__main__':unittest.main()
