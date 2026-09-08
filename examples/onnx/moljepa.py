"""Bounded FP32 CPU ONNX Runtime comparison for the Mol-JEPA SMILES tensor core.

The exported inputs are graph tensors, not SMILES strings. Every prediction,
CLS, latent embedding and requested transformer attention tensor is retained.
The exported graphs have fixed shapes and a fixed batch of four molecules.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import random
import statistics
import sys
import time
import traceback
import warnings

INPUTS = ('graph_x', 'graph_edge_index', 'graph_edge_attr', 'graph_x_batch', 'graph_x_ptr')
OUTPUTS = ('predictions', 'cls', 'embeddings', 'attention_0', 'attention_1')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def as_tuple(out):
    if len(out.attentions or []) != 2:
        raise ValueError('This probe requires both transformer attention arrays')
    return (out.predictions, out.cls, out.embeddings, *out.attentions)


def build_core(model, fast=False, explicit_mask_promotion=False):
    import torch
    from torch import nn
    class TensorCore(nn.Module):
        def __init__(self):
            super().__init__()
            self.owner = model
            self.fast = fast
            backbone = getattr(model, 'backbone', model)
            self.capture = sys.modules[type(backbone).__module__]._capture_attention
            self.modalities = [f"{m['name']}_x_ptr" if m['name'] == 'graph' else None
                               for m in backbone.model.modalities_spec]

        def forward(self, graph_x, graph_edge_index, graph_edge_attr, graph_x_batch, graph_x_ptr):
            batch = dict(zip(INPUTS, (graph_x, graph_edge_index, graph_edge_attr, graph_x_batch, graph_x_ptr)))
            B = graph_x_ptr.shape[0] - 1
            if self.fast:
                return as_tuple(self.owner.tensor_forward(batch, B, return_attn=True))
            core = getattr(self.owner, 'backbone', self.owner).model
            active_per_mod = [graph_x_ptr[1:] - graph_x_ptr[:-1] if m is not None
                              else torch.zeros(B, device=graph_x.device) for m in self.modalities]
            if explicit_mask_promotion:
                # Native torch.stack implicitly promotes the long graph counts
                # and absent-modality float zeros to float32. State that existing
                # promotion explicitly because this exporter omitted its Cast.
                active_per_mod = [value.to(dtype=torch.float32) for value in active_per_mod]
            active_mask = (torch.stack(active_per_mod) > 0).t()
            targets = core.encode(batch, self.modalities, active_mask)
            with self.capture(core.transformer_head.transformer) as captured:
                predictions, cls, embeddings = core.predict(targets, active_mask.clone(), active_mask)
            return (predictions, cls, embeddings, *captured)
    return TensorCore().eval()


def gate(left, right):
    import torch
    if len(left) != len(OUTPUTS) or len(right) != len(OUTPUTS): raise ValueError('Output count changed')
    metrics = {}
    for name, a, b in zip(OUTPUTS, left, right):
        a = a.detach().cpu(); b = torch.as_tensor(b).detach().cpu()
        if a.shape != b.shape or a.dtype != b.dtype: raise ValueError(f'Shape/dtype changed: {name}')
        if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()): raise ValueError(f'Nonfinite: {name}')
        delta = (a.double()-b.double())
        absolute = float(delta.abs().max()) if delta.numel() else 0.
        relative = float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(a.double()).clamp_min(1e-12))
        bitwise = torch.equal(a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8))
        metrics[name] = {'shape':list(a.shape), 'dtype':str(a.dtype), 'max_abs':absolute,
                         'relative_l2':relative, 'bitwise':bitwise,
                         'accepted':absolute <= 1e-5 and relative <= 1e-5}
    return {'accepted':all(m['accepted'] for m in metrics.values()),
            'bitwise':all(m['bitwise'] for m in metrics.values()), 'metrics':metrics}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--verification-smiles', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--rounds', type=int, default=10)
    parser.add_argument('--ort-spinning', choices=['off','on'], default='off',
                        help="Keep idle ORT sessions from consuming the other arms CPU budget")
    parser.add_argument('--explicit-mask-promotion', action='store_true',
                        help='Make native stack float32 promotion explicit for ONNX export')
    parser.add_argument('--exporters', nargs='+', choices=['dynamo','legacy'], default=['dynamo','legacy'])
    args=parser.parse_args()
    if args.output.exists() or args.work.exists(): parser.error('Use fresh work and report paths')
    if args.rounds < 1 or args.warmup < 0 or args.threads < 1: parser.error('Invalid timing/thread counts')
    import torch
    import onnx
    import onnxruntime as ort
    from safetensors.torch import load_file
    from compressme import load_moljepa
    from compressme.moljepa_io import moljepa_factory
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    ort.disable_telemetry_events()
    args.work.mkdir(parents=True); args.output.parent.mkdir(parents=True, exist_ok=True)
    smiles_data=json.loads(args.verification_smiles.read_text())
    smiles=smiles_data['smiles'] if isinstance(smiles_data, dict) else smiles_data
    selected=[smiles[i] for i in [5,28,46,56]]
    report={'status':'running','accepted':False,'model':'Mol-JEPA','device':'cpu',
            'smiles':selected,'input_contract':'SMILES only; embeddings_data=None; return_attn=True; fixed four-graph tensor shapes',
            'excluded_from_onnx':['SMILES strings, RDKit and molfeat preprocessing','optional modality inputs','other tensor shapes, batching and graph sizes'],
            'all_declared_outputs':list(OUTPUTS),'tolerance':{'max_abs':1e-5,'relative_l2':1e-5,'bitwise_required':False},
            'platform':platform.platform(),'python':platform.python_version(),
            'versions':{n:importlib.metadata.version(n) for n in ['torch','onnx','onnxruntime','onnxscript','onnx_ir','torch-geometric','transformers','rdkit','molfeat','numpy']},
            'threads':args.threads,'interop_threads':1,'rounds':args.rounds,'warmup':args.warmup,
            'torch_mha_fastpath_enabled':torch.backends.mha.get_fastpath_enabled(),
            'torch_float32_matmul_precision':torch.get_float32_matmul_precision(),
            'torch_deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
            'available_ort_providers':ort.get_available_providers(),'selected_ort_providers':['CPUExecutionProvider'],
            'ort_session_options':{'intra_op_num_threads':args.threads,'inter_op_num_threads':1,
                                   'execution_mode':'ORT_SEQUENTIAL','graph_optimization_level':'ORT_ENABLE_ALL',
                                   'session.intra_op.allow_spinning':'0' if args.ort_spinning=='off' else '1',
                                   'session.inter_op.allow_spinning':'0' if args.ort_spinning=='off' else '1'},
            'onnx_opset':18,'exporters_requested':args.exporters,
            'explicit_mask_promotion':args.explicit_mask_promotion,
            'compatibility_change':'Explicit native stack float32 promotion only' if args.explicit_mask_promotion else None,
            'checkpoint_sha256':sha256(args.checkpoint),'artifact_manifest_sha256':sha256(args.artifact/'manifest.json'),
            'script_sha256':sha256(__file__),'source_manifest':json.loads((args.artifact/'architecture/SOURCE.json').read_text()),
            'core_modules_sha256':{},'exports':{},'gates':{},'timing':{}}
    import compressme
    for name in ['moljepa.py','moljepa_io.py','graph_attention.py','smiles_runtime.py','readouts.py','sparse_graph.py']:
        path=Path(compressme.__file__).parent/name
        if path.exists():report['core_modules_sha256'][name]=sha256(path)
    def save(): args.output.write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        original=moljepa_factory(args.artifact/'architecture')
        original.load_state_dict(load_file(str(args.checkpoint)),strict=True)
        original.eval().requires_grad_(False)
        compressed=load_moljepa(args.artifact,device='cpu',accelerate=False).eval().requires_grad_(False)
        fast=load_moljepa(args.artifact,device='cpu').eval().requires_grad_(False)
        models={'original':original,'compressed':compressed,'final_runtime':fast}
        cores={k:build_core(m,fast=(k=='final_runtime'),explicit_mask_promotion=args.explicit_mask_promotion) for k,m in models.items()}
        report['pytorch_registered_parameters']={k:sum(p.numel() for p in m.parameters()) for k,m in models.items()}
        def preprocess():
            batch=original.smiles_to_batch(selected).to('cpu')
            return tuple(batch[name] for name in INPUTS)
        inputs=preprocess()
        report['tensor_inputs']={n:{'shape':list(x.shape),'dtype':str(x.dtype),'stride':list(x.stride()),
                                   'sha256':hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
                                 for n,x in zip(INPUTS,inputs)}
        with torch.inference_mode():
            native=as_tuple(original(selected,return_attn=True))
            for name, core in cores.items():
                expected=as_tuple(models[name](selected,return_attn=True))
                actual=core(*inputs)
                report['gates'][name+'_native_vs_core']=gate(expected,actual)
                report['gates'][name+'_vs_original_native']=gate(native,actual)
                report['gates'][name+'_self_repeat']=gate(actual,core(*inputs))
                if not all(report['gates'][k]['accepted'] for k in list(report['gates'])[-3:]):
                    raise ValueError(f'PyTorch core output gate failed: {name}')
        save()
        print('Original, compressed and final CPU tensor-core boundaries passed', flush=True)
        arms={name+'_pytorch':(lambda values,c=core:c(*values)) for name,core in cores.items()}
        # Export only the original and portable compressed core; the final runtime
        # remains a separate PyTorch arm, without changing it for export.
        for name in ['original','compressed']:
            entry={'attempts':[]};report['exports'][name]=entry
            for exporter in args.exporters:
                folder=args.work/(name+'_'+exporter);folder.mkdir()
                file=folder/'model.onnx'; log=folder/'export.txt'
                attempt={'exporter':exporter,'status':'running','stage':'graph_export'};entry['attempts'].append(attempt);save()
                start=time.perf_counter()
                try:
                    with log.open('w') as stream, contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream), warnings.catch_warnings(record=True) as caught, torch.inference_mode():
                        warnings.simplefilter('always')
                        torch.onnx.export(cores[name],inputs,str(file),input_names=list(INPUTS),output_names=list(OUTPUTS),
                                          opset_version=18,dynamo=exporter=='dynamo',external_data=True,
                                          report=exporter=='dynamo',artifacts_dir=str(folder))
                    attempt['warnings']=[str(w.message) for w in caught]
                    attempt['export_seconds']=time.perf_counter()-start
                    attempt['graph_export']='passed';attempt['stage']='onnx_checker'
                    onnx.checker.check_model(str(file))
                    attempt['onnx_checker']='passed'
                    proto=onnx.load(str(file))
                    initializers=list(proto.graph.initializer)
                    entry['onnx']={'initializer_values':sum(__import__('math').prod(t.dims) for t in initializers),
                                   'initializer_count':len(initializers),
                                   'float_initializer_types':sorted(set(onnx.TensorProto.DataType.Name(t.data_type) for t in initializers)),
                                   'operator_types':sorted(set(n.op_type for n in proto.graph.node)),
                                   'nodes':len(proto.graph.node),
                                   'files':{f.name:{'bytes':f.stat().st_size,'sha256':sha256(f)} for f in folder.iterdir() if f.suffix in ['.onnx','.data']}}
                    attempt['onnx']=entry['onnx']
                    del proto, initializers
                    attempt['stage']='ort_session_load'
                    options=ort.SessionOptions();options.intra_op_num_threads=args.threads;options.inter_op_num_threads=1
                    spinning='0' if args.ort_spinning=='off' else '1'
                    options.add_session_config_entry('session.intra_op.allow_spinning',spinning)
                    options.add_session_config_entry('session.inter_op.allow_spinning',spinning)
                    options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
                    options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    session=ort.InferenceSession(str(file),sess_options=options,providers=['CPUExecutionProvider'])
                    attempt['ort_session_load']='passed'
                    session.disable_fallback()
                    attempt['stage']='runtime_output_gate'
                    actual_names=[x.name for x in session.get_inputs()]
                    output_names=[x.name for x in session.get_outputs()]
                    if output_names != list(OUTPUTS):raise ValueError(f'Export omitted outputs: {output_names}')
                    entry['onnx']['inputs']=actual_names;entry['onnx']['outputs']=output_names
                    def invoke(values,session=session,names=actual_names):
                        feed={n:x.detach().cpu().numpy() for n,x in zip(INPUTS,values) if n in names}
                        return tuple(torch.from_numpy(x) for x in session.run(list(OUTPUTS),feed))
                    with torch.inference_mode():
                        check=gate(cores[name](*inputs),invoke(inputs))
                        entry['same_core_gate']=check
                        entry['vs_original_native']=gate(native,invoke(inputs))
                        entry['source_self_repeat']=gate(cores[name](*inputs),cores[name](*inputs))
                        entry['ort_self_repeat']=gate(invoke(inputs),invoke(inputs))
                    if not check['accepted'] or not entry['vs_original_native']['accepted']:
                        raise ValueError('ONNX Runtime output gate failed')
                    attempt['status']='accepted';entry['accepted_exporter']=exporter
                    arms[name+'_onnxruntime']=invoke
                    print(name,exporter,'ONNX Runtime gate passed',flush=True)
                    break
                except Exception as exc:
                    attempt.update(status='failed',error_type=type(exc).__name__,error=str(exc),
                                   elapsed_seconds=time.perf_counter()-start,traceback=traceback.format_exc())
                    print(name,exporter,'failed:',type(exc).__name__,str(exc)[:180],flush=True)
                finally:
                    if log.exists(): attempt['export_log_sha256']=sha256(log)
                    save()
        # Verify that export did not silently freeze changing tensor values or
        # graph indices. These probes are shape-preserving, not biological samples.
        generator=torch.Generator().manual_seed(4379)
        value_probe=tuple(x.clone() for x in inputs)
        value_probe[0].add_(torch.randn(value_probe[0].shape,generator=generator)*0.01)
        edge_order=torch.randperm(inputs[1].shape[1],generator=generator)
        edge_probe=list(inputs);edge_probe[1]=inputs[1][:,edge_order];edge_probe[2]=inputs[2][edge_order]
        permutation=torch.randperm(inputs[0].shape[0],generator=generator)
        inverse=torch.argsort(permutation)
        node_probe=list(inputs);node_probe[0]=inputs[0][permutation];node_probe[1]=inverse[inputs[1]];node_probe[3]=inputs[3][permutation]
        report['changed_tensor_gates']={}
        with torch.inference_mode():
            for case,values in [('features',value_probe),('edge_order',tuple(edge_probe)),('node_relabel',tuple(node_probe))]:
                reference=cores['original'](*values)
                report['changed_tensor_gates'][case]={name:gate(reference,fn(values)) for name,fn in arms.items()}
                if not all(g['accepted'] for g in report['changed_tensor_gates'][case].values()):
                    raise ValueError(f'Changed-input gate failed: {case}')
        save()
        if not all(name+'_onnxruntime' in arms for name in ['original','compressed']):
            report['status']='export_blocked';save();return 2
        report['timing_scope']={
            'tensor_core':'Precomputed graph tensors; includes input NumPy views and ORT-to-PyTorch output views. All five output arrays materialized.',
            'common_preprocessing_complete_call':'Fresh original SMILES/RDKit/molfeat/PyG preprocessing in every arm plus its tensor core. Final-runtime arm deliberately uses common original preprocessing, not its faster sparse parser.',
            'excluded':'Checkpoint load, export, session compilation, warmup, hashes and output validation. No input/output cache.',
            'order':'All five arms in rotated/reversed order each round; median per-round ratio uses the original PyTorch call from that round.'}
        with torch.inference_mode():
            for scope in ['tensor_core','common_preprocessing_complete_call']:
                for _ in range(args.warmup):
                    for fn in arms.values():fn(inputs if scope=='tensor_core' else preprocess())
                rounds=[]
                keys=list(arms)
                for r in range(args.rounds):
                    order=keys[r%len(keys):]+keys[:r%len(keys)]
                    if r%2:order=order[::-1]
                    measured={}
                    for name in order:
                        start=time.perf_counter()
                        actual=arms[name](inputs if scope=='tensor_core' else preprocess())
                        elapsed=time.perf_counter()-start
                        checked=gate(native,actual)
                        if not checked['accepted']:raise ValueError(f'Timed output gate failed: {scope} {name}')
                        measured[name]={'milliseconds':elapsed*1000,'output_gate':checked}
                    rounds.append({'order':order,'arms':measured})
                summary={name:{'median_ms':statistics.median(r['arms'][name]['milliseconds'] for r in rounds),
                                'min_ms':min(r['arms'][name]['milliseconds'] for r in rounds),
                                'max_ms':max(r['arms'][name]['milliseconds'] for r in rounds),
                                'median_paired_speedup_vs_original_pytorch':statistics.median(r['arms']['original_pytorch']['milliseconds']/r['arms'][name]['milliseconds'] for r in rounds)} for name in arms}
                report['timing'][scope]={'rounds':rounds,'summary':summary};save()
                print(scope,json.dumps(summary),flush=True)
        report['status']='accepted';report['accepted']=True;save();return 0
    except Exception as exc:
        report.update(status='failed',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
        save();raise


if __name__ == '__main__':
    raise SystemExit(main())
