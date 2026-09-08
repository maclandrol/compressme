#!/usr/bin/env python3
"""Research probe: compressed SGD state for two consecutive bias-free maps.

No training package integration, quantization, fitting of a replacement model,
or Adam equivalence is implemented. Run with Python and torch:
  python verify_gram_sgd.py --output gram-sgd-results.json
"""
from __future__ import annotations
import argparse
from fractions import Fraction
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


def gram_step(C,P,Q,G,eta):
    """Simultaneous plain-SGD update; every RHS uses the OLD state."""
    GQ=G@Q
    PG=P@G
    GCt=G@C.T
    CtG=C.T@G
    correction=GCt@G if C.shape[0]<=C.shape[1] else G@CtG
    return (C-eta*(GQ+PG)+(eta*eta)*correction,
            P-eta*(GCt+GCt.T)+(eta*eta)*(GQ@G.T),
            Q-eta*(CtG+CtG.T)+(eta*eta)*(G.T@PG))


def frozen_step(C,Q,G,eta,velocity=None,momentum=0.0,weight_decay=0.0):
    """Frozen B, PyTorch-style SGD, zero dampening and no Nesterov.

    velocity is the projected factor momentum V_A B. Weight decay is coupled
    into the factor gradient before momentum. The default has no momentum.
    """
    effective=G@Q+weight_decay*C
    if velocity is None:
        return C-eta*effective,None
    next_velocity=momentum*velocity+effective
    return C-eta*next_velocity,next_velocity


def _transpose(x):return list(map(list,zip(*x)))
def _matmul(x,y):
    return [[sum(a*b for a,b in zip(row,col)) for col in zip(*y)] for row in x]
def _add(x,y,scale=1):
    return [[a+scale*b for a,b in zip(ar,br)] for ar,br in zip(x,y)]
def _scale(x,s):return [[s*a for a in row] for row in x]


def exact_rational_check():
    f=Fraction
    A=[[f(1,2),f(-2,3),f(3,4)],[f(4,5),f(5,6),f(-6,7)]]
    B=[[f(7,8),f(8,9)],[f(-9,10),f(10,11)],[f(11,12),f(-12,13)]]
    G=[[f(2,3),f(-4,5)],[f(6,7),f(8,9)]]
    eta=f(1,17)
    C=_matmul(A,B);P=_matmul(A,_transpose(A));Q=_matmul(_transpose(B),B)
    next_A=_add(A,_matmul(G,_transpose(B)),-eta)
    next_B=_add(B,_matmul(_transpose(A),G),-eta)
    GCt=_matmul(G,_transpose(C));CtG=_matmul(_transpose(C),G)
    next_C=_add(_add(C,_add(_matmul(G,Q),_matmul(P,G)),-eta),_matmul(GCt,G),eta**2)
    next_P=_add(_add(P,_add(GCt,_transpose(GCt)),-eta),_matmul(_matmul(G,Q),_transpose(G)),eta**2)
    next_Q=_add(_add(Q,_add(CtG,_transpose(CtG)),-eta),_matmul(_matmul(_transpose(G),P),G),eta**2)
    assert next_C==_matmul(next_A,next_B)
    assert next_P==_matmul(next_A,_transpose(next_A))
    assert next_Q==_matmul(_transpose(next_B),next_B)
    frozen_C=_add(C,_matmul(G,Q),-eta)
    assert frozen_C==_matmul(next_A,B)
    return {'arithmetic':'fractions.Fraction','simultaneous_C_P_Q_identity':True,
            'frozen_B_C_Q_identity':True,'rounding':False}


def difference(a,b):
    delta=a.detach().double()-b.detach().double()
    return {'max_abs':float(delta.abs().max()),
            'relative_l2':float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(a.detach().double()).clamp_min(1e-30)),
            'bitwise':bool(torch.equal(a,b))}


def objective(predicted,target):
    # Arbitrary smooth nonlinearity AFTER the affine pair is allowed.
    return (torch.tanh(predicted)-target).square().mean()+0.03*predicted.square().mean()


def trajectory(dtype,forward_mode,update_mode,steps=20):
    m,h,n,batch=7,64,5,13
    eta=0.03
    generator=torch.Generator(device='cpu').manual_seed(731)
    A=nn.Parameter((torch.randn(m,h,generator=generator,dtype=torch.float64)/h**0.5).to(dtype))
    B=nn.Parameter((torch.randn(h,n,generator=generator,dtype=torch.float64)/n**0.5).to(dtype),requires_grad=update_mode=='both')
    C=(A@B).detach();P=(A@A.T).detach();Q=(B.T@B).detach()
    momentum=0.9 if update_mode=='frozen_momentum_decay' else 0.0
    decay=0.02 if update_mode=='frozen_momentum_decay' else 0.0
    optimizer=torch.optim.SGD([A,B] if update_mode=='both' else [A],lr=eta,
                              momentum=momentum,weight_decay=decay,foreach=False)
    projected_velocity=torch.zeros_like(C) if momentum else None
    truth=torch.randn(m,n,generator=generator,dtype=torch.float64).to(dtype)
    tolerance=1e-10 if dtype==torch.float64 else 1e-5
    records=[]
    for step in range(steps):
        data=torch.randn(batch,n,generator=generator,dtype=torch.float64).to(dtype)
        targets=0.4*torch.sin(data@truth.T)+0.2*torch.cos(0.7*data@truth.T)
        source_input=data.detach().clone().requires_grad_()
        compact_input=data.detach().clone().requires_grad_()
        optimizer.zero_grad(set_to_none=True)
        if forward_mode=='factorized_forward':
            source_output=F.linear(F.linear(source_input,B),A)
        else:
            source_output=F.linear(source_input,A@B)
        source_loss=objective(source_output,targets)
        source_loss.backward()
        compact_C=C.detach().requires_grad_()
        compact_output=F.linear(compact_input,compact_C)
        compact_loss=objective(compact_output,targets)
        G,compact_input_gradient=torch.autograd.grad(compact_loss,(compact_C,compact_input))
        record={'step':step+1,'output':difference(source_output,compact_output),
                'input_gradient':difference(source_input.grad,compact_input_gradient),
                'absolute_loss_difference':float((source_loss-compact_loss).abs().detach())}
        with torch.no_grad():
            if update_mode=='both':
                C,P,Q=gram_step(C,P,Q,G,eta)
            else:
                C,projected_velocity=frozen_step(C,Q,G,eta,projected_velocity,momentum,decay)
            optimizer.step()
            record['C']=difference(A@B,C)
            if update_mode=='both':
                record['P']=difference(A@A.T,P)
                record['Q']=difference(B.T@B,Q)
                block=torch.cat((torch.cat((P,C),dim=1),torch.cat((C.T,Q),dim=1)),dim=0)
                symmetric=(block+block.T)/2
                record['gram_symmetry_max_abs']=float((block-block.T).abs().max())
                record['gram_minimum_eigenvalue']=float(torch.linalg.eigvalsh(symmetric.double()).min())
            elif momentum:
                buffer=optimizer.state[A]['momentum_buffer']
                record['projected_momentum']=difference(buffer@B,projected_velocity)
        # These are numerical verification gates, not all-trajectory bounds.
        for label,pair in [('output',(source_output,compact_output)),('C',(A@B,C))]:
            torch.testing.assert_close(*pair,atol=tolerance,rtol=tolerance)
        torch.testing.assert_close(source_input.grad,compact_input_gradient,atol=tolerance,rtol=tolerance)
        if update_mode=='both':
            torch.testing.assert_close(A@A.T,P,atol=tolerance,rtol=tolerance)
            torch.testing.assert_close(B.T@B,Q,atol=tolerance,rtol=tolerance)
        elif momentum:
            torch.testing.assert_close(buffer@B,projected_velocity,atol=tolerance,rtol=tolerance)
        records.append(record)
    labels=['output','input_gradient','C']+(['P','Q'] if update_mode=='both' else [])+(['projected_momentum'] if momentum else [])
    stored=(C.numel()+P.numel()+Q.numel()) if update_mode=='both' else C.numel()+Q.numel()+(projected_velocity.numel() if momentum else 0)
    reference_stored=A.numel()+B.numel()+(A.numel() if momentum else 0)
    return {'dtype':str(dtype),'forward_mode':forward_mode,'update_mode':update_mode,'steps':steps,
            'dimensions':{'m':m,'h':h,'n':n,'batch':batch},'learning_rate':eta,
            'momentum':momentum,'coupled_weight_decay':decay,'dampening':0.0,'nesterov':False,
            'all_recorded_gates_pass':True,'absolute_tolerance':tolerance,'relative_tolerance':tolerance,
            'maximum_over_trajectory':{name:max(record[name]['max_abs'] for record in records) for name in labels},
            'state_values_before':reference_stored,'state_values_after':stored,
            'state_bytes_before':reference_stored*torch.empty((),dtype=dtype).element_size(),
            'state_bytes_after':stored*torch.empty((),dtype=dtype).element_size(),
            'storage_excludes':'Gradients, temporary products, activations, optimizer metadata and surrounding model',
            'records':records}


def storage_cost(m,h,n):
    factors=h*(m+n)
    full= m*n+m*m+n*n
    packed=m*n+m*(m+1)//2+n*(n+1)//2
    return {'m':m,'h':h,'n':n,'factor_values':factors,'gram_full_values':full,
            'gram_symmetric_packed_values':packed,'full_state_storage_ratio':factors/full,
            'packed_state_storage_ratio':factors/packed,
            'frozen_B_C_Q_values':m*n+n*n,
            'frozen_B_C_Q_momentum_values':2*m*n+n*n,
            'factor_gradient_products_macs_given_G':2*m*n*h,
            'gram_update_macs_shared_products':3*m*n*(m+n)+m*n*min(m,n),
            'frozen_B_projected_update_macs_given_G':m*n*n,
            'caveat':'MAC counts exclude gradient-G evaluation, activations, scalar updates and memory traffic; no wall-clock speed claim'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).with_name('gram-sgd-results.json'))
    parser.add_argument('--steps',type=int,default=20)
    args=parser.parse_args()
    if args.steps<1:parser.error('--steps must be positive')
    torch.set_num_threads(1)
    result={'status':'isolated_research_validation_not_production','torch_version':torch.__version__,
            'device':'cpu','quantization':False,'distillation':False,'novelty_claim':False,
            'exact_rational':exact_rational_check(),
            'trajectories':[trajectory(dtype,forward,mode,args.steps)
                for dtype in (torch.float64,torch.float32)
                for forward in ('factorized_forward','product_forward')
                for mode in ('both','frozen','frozen_momentum_decay')],
            'storage_examples':[storage_cost(7,64,5),storage_cost(512,2048,512),storage_cost(4096,512,82),storage_cost(512,8,512)]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'exact_rational':result['exact_rational'],'trajectories':[
        {k:v for k,v in item.items() if k not in ('records','storage_excludes')} for item in result['trajectories']],
        'output':str(args.output)},indent=2))


if __name__=='__main__':main()
