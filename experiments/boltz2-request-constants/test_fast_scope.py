"""Routing/cleanup only: model forward/sample and source proof check are fixtures."""
import unittest
from unittest.mock import patch
import torch
from torch import nn
import boltz2_runtime as candidate
from test_scope_guards import fixture,local_forward,DiffusionModule,AtomDiffusion

class FastScope(unittest.TestCase):
    def run_case(self,body,expect_exception=None):
        with torch.no_grad():
            model,inputs=fixture();ids={n:id(m) for n,m in model.named_modules()}
            with patch.object(candidate,'_verify_source_and_hierarchy',lambda d:d.score_model),patch.object(DiffusionModule,'forward',local_forward),patch.object(AtomDiffusion,'sample',body):
                if expect_exception:
                    with self.assertRaisesRegex(RuntimeError,expect_exception):
                        with candidate.boltz2_invariant_sampling(model,immutable_request=True) as stats:model.sample(**inputs)
                else:
                    with candidate.boltz2_invariant_sampling(model,immutable_request=True) as stats:model.sample(**inputs)
            self.assertEqual(ids,{n:id(m) for n,m in model.named_modules()})
            self.assertNotIn('sample',model.__dict__);self.assertNotIn('forward',model.score_model.__dict__)
            self.assertTrue(stats['requests'][0]['cache_cleared'])
            return stats['requests'][0]
    def test_persistent_wrappers(self):
        def body(d,**inputs):
            wrappers=[]
            for i in range(5):
                d.score_model(**inputs,times=torch.zeros(1));wrappers.append(id(d.score_model.single_conditioner))
            self.assertEqual(len(set(wrappers)),1)
        stats=self.run_case(body);self.assertEqual(stats['wrapper_installations'],1);self.assertEqual(stats['optimized_score_calls'],5)
    def test_changed_multiplicity_reinstalls_and_reuses(self):
        def body(d,**inputs):
            for m in (1,1,3,3,1):d.score_model(**inputs,times=torch.zeros(m),multiplicity=m)
        stats=self.run_case(body);self.assertEqual(stats['wrapper_installations'],3);self.assertEqual(len(stats['prepared_layouts']),2)
    def test_autocast_fallback_restores(self):
        def body(d,**inputs):
            d.score_model(**inputs,times=torch.zeros(1))
            with torch.autocast('cpu',dtype=torch.bfloat16):
                d.score_model(**inputs,times=torch.zeros(1))
                self.assertNotIsInstance(d.score_model.single_conditioner,candidate.PreparedSingleConditioningPrefix)
            d.score_model(**inputs,times=torch.zeros(1))
        stats=self.run_case(body);self.assertEqual(stats['optimized_score_calls'],2);self.assertEqual(stats['fallback_score_calls'],1)
    def test_changed_conditioning_identity_restores(self):
        def body(d,**inputs):
            d.score_model(**inputs,times=torch.zeros(1))
            other=dict(inputs,s_trunk=inputs['s_trunk']+1)
            d.score_model(**other,times=torch.zeros(1))
            self.assertNotIsInstance(d.score_model.single_conditioner,candidate.PreparedSingleConditioningPrefix)
        stats=self.run_case(body);self.assertEqual(stats['fallback_score_calls'],1)
    def test_changed_input_rejects(self):
        def body(d,**inputs):
            d.score_model(**inputs,times=torch.zeros(1));inputs['s_trunk'].add_(1);d.score_model(**inputs,times=torch.zeros(1))
        self.run_case(body,'conditioning changed')
    def test_rebound_parameter_rejects_at_boundary(self):
        def body(d,**inputs):
            d.score_model(**inputs,times=torch.zeros(1));source=d.score_model.single_conditioner.source
            source.single_embed.bias=nn.Parameter(source.single_embed.bias+1,requires_grad=False)
            d.score_model(**inputs,times=torch.zeros(1))
        stats=self.run_case(body,'model state changed');self.assertFalse(stats['boundary_state_unchanged'])
    def test_changed_parameter_storage_rejects_at_boundary(self):
        def body(d,**inputs):
            d.score_model(**inputs,times=torch.zeros(1));source=d.score_model.single_conditioner.source
            source.single_embed.bias.data=(source.single_embed.bias+1).clone()
        self.run_case(body,'model state changed')
    def test_changed_sampler_setting_rejects(self):
        def body(d,**inputs):
            d.score_model(**inputs,times=torch.zeros(1));d.new_setting=42
        self.run_case(body,'model state changed')
    def test_exception_cleanup(self):
        def body(d,**inputs):d.score_model(**inputs,times=torch.zeros(1));raise RuntimeError('planned')
        self.run_case(body,'planned')
    def test_between_request_model_edit_refused(self):
        def body(d,**inputs):d.score_model(**inputs,times=torch.zeros(1))
        with torch.no_grad():
            model,inputs=fixture()
            with patch.object(candidate,'_verify_source_and_hierarchy',lambda d:d.score_model),patch.object(DiffusionModule,'forward',local_forward),patch.object(AtomDiffusion,'sample',body):
                with candidate.boltz2_invariant_sampling(model,immutable_request=True):
                    model.sample(**inputs)
                    model.score_model.single_conditioner.single_embed.bias.add_(1)
                    with self.assertRaisesRegex(ValueError,'outer scope'):model.sample(**inputs)
            self.assertNotIn('sample',model.__dict__)
            self.assertNotIn('forward',model.score_model.__dict__)
    def test_native_source_hierarchy_guard(self):
        arguments=dict(token_s=4,atom_s=4,atoms_per_window_queries=2,atoms_per_window_keys=4,dim_fourier=4,atom_encoder_depth=1,atom_encoder_heads=1,token_transformer_depth=1,token_transformer_heads=1,atom_decoder_depth=1,atom_decoder_heads=1,conditioning_transition_layers=0)
        with torch.no_grad():
            model=AtomDiffusion(score_model_args=arguments).eval().requires_grad_(False)
            with candidate.boltz2_invariant_sampling(model,immutable_request=True):pass
            original=model.score_model.atom_attention_encoder.atom_encoder.diffusion_transformer.layers[0].adaln
            class AlteredAdaLN(type(original)):
                def forward(self,a,s):return super().forward(a,s)+1
            original.__class__=AlteredAdaLN
            with self.assertRaisesRegex(ValueError,'exact AdaLN'):
                with candidate.boltz2_invariant_sampling(model,immutable_request=True):pass
    def test_sparse_static_input_falls_back_and_next_request_works(self):
        def body(d,**inputs):return d.score_model(**inputs,times=torch.zeros(1))
        with torch.no_grad():
            model,inputs=fixture()
            with patch.object(candidate,'_verify_source_and_hierarchy',lambda d:d.score_model),patch.object(DiffusionModule,'forward',local_forward),patch.object(AtomDiffusion,'sample',body):
                with candidate.boltz2_invariant_sampling(model,immutable_request=True) as stats:
                    sparse=dict(inputs,diffusion_conditioning={'c':inputs['diffusion_conditioning']['c'].to_sparse()})
                    model.sample(**sparse)
                    model.sample(**inputs)
            self.assertEqual(stats['requests'][0]['fallback_score_calls'],1)
            self.assertEqual(stats['requests'][1]['optimized_score_calls'],1)
            self.assertTrue(all(r['cache_cleared'] for r in stats['requests']))
    def test_boundary_audit_error_releases_request_ownership(self):
        def body(d,**inputs):return d.score_model(**inputs,times=torch.zeros(1))
        actual_signature=candidate._boundary_signature
        calls=[0]
        def signature(model):
            calls[0]+=1
            if calls[0]==3:raise RuntimeError('planned audit failure')
            return actual_signature(model)
        with torch.no_grad():
            model,inputs=fixture()
            with patch.object(candidate,'_verify_source_and_hierarchy',lambda d:d.score_model),patch.object(DiffusionModule,'forward',local_forward),patch.object(AtomDiffusion,'sample',body),patch.object(candidate,'_boundary_signature',signature):
                with candidate.boltz2_invariant_sampling(model,immutable_request=True) as stats:
                    with self.assertRaisesRegex(RuntimeError,'planned audit failure'):model.sample(**inputs)
                    model.sample(**inputs)
            self.assertEqual(len(stats['requests']),2)
            self.assertTrue(all(r['cache_cleared'] for r in stats['requests']))
            self.assertTrue(stats['requests'][1]['boundary_state_unchanged'])
    def test_restore_error_releases_ownership_and_preserves_primary_error(self):
        calls=[0]
        def body(d,**inputs):
            result=d.score_model(**inputs,times=torch.zeros(1));calls[0]+=1
            if calls[0]==1:raise RuntimeError('primary sampling failure')
            return result
        original_close=candidate.ExitStack.close
        injected=[False]
        def close(stack):
            has_callbacks=bool(stack._exit_callbacks)
            original_close(stack)
            if has_callbacks and not injected[0]:
                injected[0]=True
                raise ValueError('planned restore failure')
        with torch.no_grad():
            model,inputs=fixture();identities={n:id(m) for n,m in model.named_modules()}
            with patch.object(candidate,'_verify_source_and_hierarchy',lambda d:d.score_model),patch.object(DiffusionModule,'forward',local_forward),patch.object(AtomDiffusion,'sample',body),patch.object(candidate.ExitStack,'close',close):
                with candidate.boltz2_invariant_sampling(model,immutable_request=True) as stats:
                    with self.assertRaisesRegex(RuntimeError,'primary sampling failure'):model.sample(**inputs)
                    model.sample(**inputs)
            self.assertEqual(identities,{n:id(m) for n,m in model.named_modules()})
            self.assertTrue(all(r['memo_cleared'] for r in stats['requests']))
            self.assertFalse(stats['requests'][0]['cache_cleared'])
            self.assertEqual(stats['requests'][0]['cleanup_error_type'],'ValueError')
            self.assertTrue(stats['requests'][1]['boundary_state_unchanged'])
    def test_owner_contract_explicit(self):
        model,_=fixture()
        with self.assertRaises(TypeError):
            with candidate.boltz2_invariant_sampling(model):pass
        with self.assertRaisesRegex(ValueError,'explicit'):
            with candidate.boltz2_invariant_sampling(model,immutable_request=False):pass

if __name__=='__main__':unittest.main()
