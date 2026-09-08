"""Create actual request-conditioning inputs without running the full trunk again."""
import argparse,hashlib,json,sys
from pathlib import Path
import torch
from safetensors.torch import load_file,save_file

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--shared-state',type=Path,required=True);ap.add_argument('--safe-loader-dir',type=Path,required=True);ap.add_argument('--native-runtime-dir',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    args.output.mkdir(exist_ok=True);torch.set_num_threads(2);torch.set_grad_enabled(False);torch.manual_seed(1729)
    sys.path.insert(0,str(args.safe_loader_dir));sys.path.insert(0,str(args.native_runtime_dir))
    from safe_state import load_tensor_state
    from sharing_probe import batch_from_existing
    from boltz.model.modules.trunkv2 import InputEmbedder
    from boltz.model.modules.encodersv2 import RelativePositionEncoder
    from boltz.model.modules.diffusion_conditioning import DiffusionConditioning
    _,feats=batch_from_existing(args.native_runtime_dir/'mps-fp32-default-sampling',args.native_runtime_dir/'mols')
    native=args.native_runtime_dir/'mps-fp32-default-sampling/output-tensors.safetensors';outputs=load_file(str(native));s_trunk=outputs['output/s'];z_trunk=outputs['output/z']
    state,h=load_tensor_state(args.shared_state,'boltz2_conf.ckpt');score=h['score_model_args']
    common={k:h[k] for k in ['atom_s','atom_z','token_s','token_z','atoms_per_window_queries','atoms_per_window_keys','atom_feature_dim','use_no_atom_char','use_atom_backbone_feat','use_residue_feats_atoms']}
    inp=InputEmbedder(**common,**h['embedder_args']).eval().requires_grad_(False)
    rel=RelativePositionEncoder(h['token_z'],fix_sym_check=h['fix_sym_check'],cyclic_pos_enc=h['cyclic_pos_enc']).eval().requires_grad_(False)
    diff=DiffusionConditioning(**common,**{k:score[k] for k in ['atom_encoder_depth','atom_encoder_heads','token_transformer_depth','token_transformer_heads','atom_decoder_depth','atom_decoder_heads','conditioning_transition_layers']}).eval().requires_grad_(False)
    for module,prefix in [(inp,'input_embedder.'),(rel,'rel_pos.'),(diff,'diffusion_conditioning.')]:module.load_state_dict({k.removeprefix(prefix):v for k,v in state.items() if k.startswith(prefix)},strict=True)
    del state
    s_inputs=inp(feats);relative=rel(feats);q,c,to_keys,enc,dec,token=diff(s_trunk,z_trunk,relative,feats)
    save_file({'c':c.contiguous(),'q':q.contiguous(),'atom_enc_bias':enc.contiguous(),'atom_dec_bias':dec.contiguous(),'atom_mask':feats['atom_pad_mask'].contiguous(),'s_inputs':s_inputs.contiguous(),'s_trunk':s_trunk.contiguous()},str(args.output/'conditioning-inputs.safetensors'))
    report={'scope':'Actual 20-residue protein plus ethanol native input. Saved native MPS trunk s/z are input values; small original embedder/conditioning modules evaluated on CPU to produce local request-conditioning fixtures. This is not an end-to-end regenerated prediction.','native_trunk_sha256':hashlib.sha256(native.read_bytes()).hexdigest(),'shapes':{'c':list(c.shape),'s_inputs':list(s_inputs.shape),'s_trunk':list(s_trunk.shape)},'all_finite':all(bool(torch.isfinite(t).all()) for t in (c,s_inputs,s_trunk))}
    (args.output/'fixture-provenance.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
