#!/usr/bin/env python3
"""Isolated numerical check of a fixed hidden-subspace change of coordinates.

No singular-value truncation: reduced QR retains min(h,m+n) columns. These
contain the column span of [A.T, B] in real arithmetic. The resulting reduced
factors use ordinary torch.optim.SGD. This script does not modify compressme.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from verify_gram_sgd import difference, objective


def trajectory(dtype, momentum, weight_decay, steps=20):
    m,h,n,batch=7,64,5,13
    eta=0.03
    generator=torch.Generator(device='cpu').manual_seed(731)
    A=nn.Parameter((torch.randn(m,h,generator=generator,dtype=torch.float64)/h**0.5).to(dtype))
    B=nn.Parameter((torch.randn(h,n,generator=generator,dtype=torch.float64)/n**0.5).to(dtype))
    with torch.no_grad():
        # Keep every reduced-QR column, even for a rank-deficient input. There
        # is no numerical rank threshold or approximate singular-value drop.
        R=torch.linalg.qr(torch.cat((A.T,B),dim=1),mode='reduced').Q
        Ar=nn.Parameter(A@R)
        Br=nn.Parameter(R.T@B)
        basis_audit={'basis_shape':list(R.shape),
                     'no_singular_value_truncation':True,
                     'orthonormality_max_abs':float((R.T@R-torch.eye(R.shape[1],dtype=dtype)).abs().max()),
                     'A_reconstruction':difference(A,Ar@R.T),
                     'B_reconstruction':difference(B,R@Br)}
    source_optimizer=torch.optim.SGD([A,B],lr=eta,momentum=momentum,weight_decay=weight_decay,foreach=False)
    reduced_optimizer=torch.optim.SGD([Ar,Br],lr=eta,momentum=momentum,weight_decay=weight_decay,foreach=False)
    truth=torch.randn(m,n,generator=generator,dtype=torch.float64).to(dtype)
    tolerance=1e-10 if dtype==torch.float64 else 1e-5
    records=[]
    for step in range(steps):
        data=torch.randn(batch,n,generator=generator,dtype=torch.float64).to(dtype)
        targets=0.4*torch.sin(data@truth.T)+0.2*torch.cos(0.7*data@truth.T)
        x=data.detach().clone().requires_grad_()
        xr=data.detach().clone().requires_grad_()
        source_optimizer.zero_grad(set_to_none=True)
        reduced_optimizer.zero_grad(set_to_none=True)
        y=F.linear(F.linear(x,B),A)
        yr=F.linear(F.linear(xr,Br),Ar)
        loss=objective(y,targets);reduced_loss=objective(yr,targets)
        loss.backward();reduced_loss.backward()
        pairs={'output':(y,yr),'input_gradient':(x.grad,xr.grad)}
        source_optimizer.step();reduced_optimizer.step()
        with torch.no_grad():
            pairs.update({'C':(A@B,Ar@Br),'P':(A@A.T,Ar@Ar.T),'Q':(B.T@B,Br.T@Br),
                          'A_fixed_basis':(A@R,Ar),'B_fixed_basis':(R.T@B,Br)})
            if momentum:
                pairs['A_projected_momentum']=(source_optimizer.state[A]['momentum_buffer']@R,
                                                reduced_optimizer.state[Ar]['momentum_buffer'])
                pairs['B_projected_momentum']=(R.T@source_optimizer.state[B]['momentum_buffer'],
                                                reduced_optimizer.state[Br]['momentum_buffer'])
            record={'step':step+1, 'absolute_loss_difference':float((loss-reduced_loss).abs().detach())}
            for name,(source,candidate) in pairs.items():
                record[name]=difference(source,candidate)
                torch.testing.assert_close(source,candidate,atol=tolerance,rtol=tolerance)
            records.append(record)
    labels=list(pairs)
    states=2 if momentum else 1
    return {'dtype':str(dtype),'steps':steps,'dimensions':{'m':m,'h':h,'n':n,'batch':batch,'reduced_h':R.shape[1]},
            'basis_audit':basis_audit,'learning_rate':eta,'momentum':momentum,'coupled_weight_decay':weight_decay,
            'dampening':0.0,'nesterov':False,'all_recorded_gates_pass':True,
            'absolute_tolerance':tolerance,'relative_tolerance':tolerance,
            'maximum_over_trajectory':{name:max(record[name]['max_abs'] for record in records) for name in labels},
            'state_values_before':states*(A.numel()+B.numel()),
            'state_values_after':states*(Ar.numel()+Br.numel()),
            'basis_retained_for_comparison_only':True,
            'deployment_basis_required':False,
            'storage_excludes':'Comparison-only R, gradients, activations and optimizer metadata',
            'records':records}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).with_name('hidden-subspace-sgd-results.json'))
    parser.add_argument('--steps',type=int,default=20)
    args=parser.parse_args()
    if args.steps<1:parser.error('--steps must be positive')
    torch.set_num_threads(1)
    result={'status':'isolated_research_validation_not_production','torch_version':torch.__version__,
            'device':'cpu','quantization':False,'distillation':False,'novelty_claim':False,
            'trajectories':[trajectory(dtype,momentum,decay,args.steps)
                            for dtype in (torch.float64,torch.float32)
                            for momentum,decay in ((0.0,0.0),(0.9,0.02))]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'output':str(args.output),'trajectories':[
        {k:v for k,v in item.items() if k not in ('records','storage_excludes')}
        for item in result['trajectories']]},indent=2))


if __name__=='__main__':main()
