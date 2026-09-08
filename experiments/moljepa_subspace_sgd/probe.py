#!/usr/bin/env python3
"""Actual-weight Mol-JEPA first attention component, isolated SGD research.

Contract: raw82 node features; original input projection and shared edge
projection frozen; only query/key weights and biases train with plain SGD.
All 512 value coordinates, edge-value messages, head averaging and skip are
retained. This is not a production adapter or whole-model training test.
"""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import torch
import torch_geometric
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import TransformerConv
from torch_geometric.utils import dense_to_sparse, softmax
from safetensors import safe_open

EXPECTED_CHECKPOINT_SHA256='a443592193075334b55483501f1e04cdf4ef9c461db103f687ba4335b036cf14'
EXPECTED_SOURCE_SHA256='89eed5f7ebd458b2d4e6a1e4ad90aed275b7b35bfaa88f4a85a59833e57a0d4d'


def error(a,b):
    d=a.detach().double()-b.detach().double()
    return {'max_abs':float(d.abs().max()) if d.numel() else 0.,
            'relative_l2':float(d.norm()/a.detach().double().norm().clamp_min(1e-30)),
            'bitwise':torch.equal(a,b)}


def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def qk_aug(conv,name):
    layer=getattr(conv,'lin_'+name)
    return torch.cat((layer.weight.reshape(8,512,512),layer.bias.reshape(8,512,1)),dim=-1)


class Source(nn.Module):
    def __init__(self,checkpoint,dtype):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            self.input_proj=nn.Linear(82,512)
            self.conv=TransformerConv(512,512,heads=8,concat=False,edge_dim=17)
        with safe_open(str(checkpoint),framework='pt',device='cpu') as f:
            self.input_proj.load_state_dict({k:f.get_tensor('model.encoders.graph.input_proj.'+k) for k in ('weight','bias')})
            prefix='model.encoders.graph.convs.0.'
            self.conv.load_state_dict({k[len(prefix):]:f.get_tensor(k) for k in f.keys() if k.startswith(prefix)},strict=True)
        self.to(dtype=dtype).eval().requires_grad_(False)
        self.conv.lin_query.requires_grad_(True)
        self.conv.lin_key.requires_grad_(True)

    def forward(self,raw,edge_index,edge_attr):
        x=self.input_proj(raw)
        out,(_,alpha)=self.conv(x,edge_index,edge_attr,return_attention_weights=True)
        q=self.conv.lin_query(x).reshape(-1,8,512)
        k=self.conv.lin_key(x).reshape(-1,8,512)
        e=self.conv.lin_edge(edge_attr).reshape(-1,8,512)
        src,dst=edge_index
        logits=(q[dst]*(k[src]+e)).sum(-1)/math.sqrt(512)
        return out,alpha,logits


class Reduced(nn.Module):
    def __init__(self,source):
        super().__init__()
        self.input_proj=copy.deepcopy(source.input_proj)
        self.lin_value=copy.deepcopy(source.conv.lin_value)
        self.lin_edge=copy.deepcopy(source.conv.lin_edge)
        self.lin_skip=copy.deepcopy(source.conv.lin_skip)
        dtype=source.input_proj.weight.dtype
        with torch.no_grad():
            # Augmented node input is F [raw;1], and U is orthonormal. Keep
            # every reduced-QR column: no numerical rank truncation.
            transform=torch.zeros(513,83,dtype=torch.float64)
            transform[:512,:82]=source.input_proj.weight.double()
            transform[:512,82]=source.input_proj.bias.double()
            transform[512,82]=1
            U=torch.linalg.qr(transform,mode='reduced').Q
            Q=qk_aug(source.conv,'query').double()
            K=qk_aug(source.conv,'key').double()
            E=source.conv.lin_edge.weight.double().reshape(8,512,17)
            R=torch.stack([torch.linalg.qr(torch.cat((Q[h]@U,K[h]@U,E[h]),dim=1),mode='reduced').Q for h in range(8)])
            self.query=nn.Parameter(torch.stack([R[h].T@Q[h]@U for h in range(8)]).to(dtype))
            self.key=nn.Parameter(torch.stack([R[h].T@K[h]@U for h in range(8)]).to(dtype))
            self.register_buffer('raw_coordinates',(U.T@transform).to(dtype))
            self.register_buffer('score_edges',torch.stack([R[h].T@E[h] for h in range(8)]).to(dtype))
            # Proof/verification data deliberately not part of module state.
            self.audit_U=U
            self.audit_R=R
            self.basis_audit={'input_width':513,'reduced_input_width':83,'head_width':512,'reduced_head_width':183,
                'orthonormal_U_max_abs':float((U.T@U-torch.eye(83,dtype=torch.float64)).abs().max()),
                'orthonormal_R_max_abs':float((R.transpose(1,2)@R-torch.eye(183,dtype=torch.float64)).abs().max()),
                'input_reconstruction':error(transform,U@(U.T@transform)),
                'reachable_Q_reconstruction':error(Q@U,R@self.query.double()),
                'reachable_K_reconstruction':error(K@U,R@self.key.double()),
                'edge_reconstruction':error(E,R@self.score_edges.double()),
                'decomposition_dtype':'float64','retained_parameter_dtype':str(dtype),'no_rank_threshold':True}

    def forward(self,raw,edge_index,edge_attr):
        x=self.input_proj(raw)
        aug=torch.cat((raw,torch.ones_like(raw[:,:1])),dim=-1)
        z=F.linear(aug,self.raw_coordinates)
        q=F.linear(z,self.query.flatten(0,1)).reshape(-1,8,183)
        k=F.linear(z,self.key.flatten(0,1)).reshape(-1,8,183)
        er=F.linear(edge_attr,self.score_edges.flatten(0,1)).reshape(-1,8,183)
        ev=self.lin_edge(edge_attr).reshape(-1,8,512)
        value=self.lin_value(x).reshape(-1,8,512)
        src,dst=edge_index
        logits=(q[dst]*(k[src]+er)).sum(-1)/math.sqrt(512)
        alpha=softmax(logits,dst,num_nodes=raw.shape[0])
        messages=(value[src]+ev)*alpha.unsqueeze(-1)
        aggregate=torch.zeros(raw.shape[0],8,512,dtype=raw.dtype,device=raw.device).index_add(0,dst,messages)
        return aggregate.mean(1)+self.lin_skip(x),alpha,logits


def original_featurizer(source_file):
    # Execute only the pinned upstream featurizer class, not remote model code.
    tree=ast.parse(source_file.read_text())
    selected=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='GraphFeaturizer')
    namespace={'torch':torch,'dense_to_sparse':dense_to_sparse}
    exec(compile(ast.Module(body=[selected],type_ignores=[]),str(source_file),'exec'),namespace)
    return namespace['GraphFeaturizer']()


def loss_fn(out,alpha,target):
    return (torch.tanh(out)-target).square().mean()+0.01*alpha.square().mean()


def train_probe(checkpoint,graphs,dtype,steps):
    source=Source(checkpoint,dtype)
    reduced=Reduced(source)
    first_reduced=copy.deepcopy(reduced)
    first_source=copy.deepcopy(source)
    optimizer=torch.optim.SGD([p for p in source.parameters() if p.requires_grad],lr=0.05,foreach=False)
    small_optimizer=torch.optim.SGD([reduced.query,reduced.key],lr=0.05,foreach=False)
    tol=1e-10 if dtype==torch.float64 else 1e-5
    records=[]
    for step in range(steps):
        graph=graphs[step%len(graphs)]
        x=graph['x'].to(dtype).clone().requires_grad_()
        xr=x.detach().clone().requires_grad_()
        e=graph['edge_attr'].to(dtype).clone().requires_grad_()
        er=e.detach().clone().requires_grad_()
        idx=graph['edge_index'].long()
        target=torch.sin(torch.arange(x.shape[0]*512,dtype=dtype).reshape(x.shape[0],512)*0.07+step*0.11)
        optimizer.zero_grad(set_to_none=True);small_optimizer.zero_grad(set_to_none=True)
        out,alpha,logits=source(x,idx,e)
        out_r,alpha_r,logits_r=reduced(xr,idx,er)
        loss=loss_fn(out,alpha,target);loss_r=loss_fn(out_r,alpha_r,target)
        loss.backward();loss_r.backward()
        pairs={'conv_output':(out,out_r),'attention':(alpha,alpha_r),'raw_logits':(logits,logits_r),
               'raw_node_gradient':(x.grad,xr.grad),'edge_feature_gradient':(e.grad,er.grad)}
        record={'step':step+1,'nodes':x.shape[0],'edges':e.shape[0],'loss_difference':float((loss-loss_r).detach().abs())}
        for name,pair in pairs.items():
            record[name]=error(*pair)
            torch.testing.assert_close(*pair,atol=tol,rtol=tol)
        optimizer.step();small_optimizer.step()
        if step in (0,steps-1):
            with torch.no_grad():
                for name in ('query','key'):
                    original=qk_aug(source.conv,name).double()
                    projected=reduced.audit_R.transpose(1,2)@original@reduced.audit_U
                    candidate=getattr(reduced,name)
                    record[name+'_projected_parameters']=error(projected,candidate)
                    torch.testing.assert_close(projected,candidate.double(),atol=tol,rtol=tol)
        records.append(record)
    # Explicit unusual COO graphs retain complete edge semantics: duplicate
    # directed edges, self-loop, an isolated node, and an entirely empty edge set.
    unusual=[]
    generator=torch.Generator().manual_seed(884)
    for label,idx in [('mixed_coo',torch.tensor([[0,0,1,2,2],[1,1,0,2,1]])),
                      ('no_edges',torch.empty((2,0),dtype=torch.long))]:
        x=torch.randn(4,82,generator=generator,dtype=dtype)
        e=torch.randn(idx.shape[1],17,generator=generator,dtype=dtype)
        a=first_source(x,idx,e);b=first_reduced(x,idx,e)
        entry={'case':label}
        for name,v,w in zip(('conv_output','attention','raw_logits'),a,b):
            entry[name]=error(v,w)
            entry[name]['mixed_tolerance_pass']=bool(torch.allclose(v,w,atol=tol,rtol=tol))
        unusual.append(entry)
    values_source=sum(t.numel() for t in source.state_dict().values())
    values_reduced=sum(t.numel() for t in reduced.state_dict().values())
    return {'dtype':str(dtype),'steps':steps,'all_training_gates_pass':True,
            'all_stress_gates_pass':all(v['mixed_tolerance_pass'] for case in unusual for v in case.values() if isinstance(v,dict)),
            'absolute_tolerance':tol,'relative_tolerance':tol,
            'basis_audit':reduced.basis_audit,'original_attention_scale':1/math.sqrt(512),
            'trainable_source_values':sum(p.numel() for p in source.parameters() if p.requires_grad),
            'trainable_reduced_values':sum(p.numel() for p in reduced.parameters() if p.requires_grad),
            'stored_source_component_values':values_source,'stored_reduced_component_values':values_reduced,
            'storage_includes':'input,query,key,value,sharededge,skip weights/biases and compiled constant coordinates/score-edge tables',
            'storage_excludes':'comparison-only U/R, activations, gradients, temporary products and all remaining Mol-JEPA layers',
            'maximum_over_trajectory':{k:max(r[k]['max_abs'] for r in records) for k in pairs},
            'records':records,'unusual_coo_checks':unusual}


def consumer_obstructions(checkpoint,graph):
    source=Source(checkpoint,torch.float64)
    candidate=Reduced(source)
    source.requires_grad_(True)
    x=graph['x'].double();e=graph['edge_attr'].double();idx=graph['edge_index'].long()
    out,alpha,_=source(x,idx,e)
    target=torch.sin(torch.arange(out.numel(),dtype=torch.float64).reshape_as(out)*0.07)
    loss_fn(out,alpha,target).backward()
    R=candidate.audit_R
    Egrad=source.conv.lin_edge.weight.grad.reshape(8,512,17)
    escaped=Egrad-R@(R.transpose(1,2)@Egrad)
    augmented_grad=torch.zeros(513,83,dtype=torch.float64)
    augmented_grad[:512,:82]=source.input_proj.weight.grad
    augmented_grad[:512,82]=source.input_proj.bias.grad
    Q=qk_aug(source.conv,'query').detach()
    delta=Q@augmented_grad
    escaped_input=delta-R@(R.transpose(1,2)@delta)
    return {'shared_edge_gradient_outside_retained_score_span':float(escaped.norm()),
            'shared_edge_gradient_outside_fraction':float(escaped.norm()/Egrad.norm()),
            'query_response_to_input_projection_gradient_outside_span':float(escaped_input.norm()),
            'query_response_outside_fraction':float(escaped_input.norm()/delta.norm()),
            'interpretation':'Nonzero omitted directions show why unrestricted shared-edge/input-projection SGD is outside this fixed-subspace contract.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--source-file',type=Path,required=True)
    p.add_argument('--expected-checkpoint-sha256',default=EXPECTED_CHECKPOINT_SHA256)
    p.add_argument('--expected-source-sha256',default=EXPECTED_SOURCE_SHA256)
    p.add_argument('--output',type=Path,default=Path(__file__).with_name('replay-results.json'))
    p.add_argument('--overwrite',action='store_true',help='Explicitly allow replacing the requested output file')
    p.add_argument('--check-inputs-only',action='store_true',help='Verify hashes and exit before executing source code or model probes')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--steps',type=int,default=20)
    args=p.parse_args()
    if args.steps<1:p.error('--steps must be positive')
    if args.threads<1:p.error('--threads must be positive')
    for name in ('checkpoint','source_file'):
        if not getattr(args,name).is_file():p.error(f'{name} must point to an existing local file')
    checkpoint_hash=sha256(args.checkpoint)
    source_hash=sha256(args.source_file)
    if checkpoint_hash!=args.expected_checkpoint_sha256:
        p.error(f'Checkpoint SHA256 mismatch: {checkpoint_hash}')
    if source_hash!=args.expected_source_sha256:
        p.error(f'Source SHA256 mismatch: {source_hash}; source code was not executed')
    if args.check_inputs_only:
        print(json.dumps({'verified':True,'checkpoint_sha256':checkpoint_hash,'source_sha256':source_hash}))
        return
    if args.output.exists() and not args.overwrite:
        p.error('Output already exists; choose another path or explicitly pass --overwrite')
    torch.set_num_threads(args.threads)
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    smiles=['CCO','c1ccccc1','C[C@H](N)C(=O)O','CC(=O)Oc1ccccc1C(=O)O']
    feats=original_featurizer(args.source_file)
    graphs=[feats.encode(s) for s in smiles]
    result={'status':'isolated_component_SGD_validation','whole_model_training_validated':False,
            'novelty_claim':False,'distillation':False,'quantization':False,'torch':torch.__version__,
            'torch_geometric':torch_geometric.__version__,
            'checkpoint':str(args.checkpoint),'checkpoint_bytes':args.checkpoint.stat().st_size,
            'checkpoint_sha256':checkpoint_hash,'source_sha256':source_hash,
            'smiles':smiles,'trajectories':[train_probe(args.checkpoint,graphs,dtype,args.steps) for dtype in (torch.float64,torch.float32)],
            'consumer_obstructions':consumer_obstructions(args.checkpoint,graphs[-1])}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w' if args.overwrite else 'x') as stream:
        stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({**{k:v for k,v in result.items() if k!='trajectories'},'trajectories':[
        {k:v for k,v in r.items() if k!='records'} for r in result['trajectories']]},indent=2))


if __name__=='__main__':main()
