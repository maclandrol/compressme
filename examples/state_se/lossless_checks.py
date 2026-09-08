"""Complete original State SE output probes shared by portable backend validation."""
import torch
from torch.nn import functional as F

def cpu_tree(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, tuple): return tuple(cpu_tree(v) for v in value)
    if isinstance(value, list): return [cpu_tree(v) for v in value]
    if isinstance(value, dict): return {k:cpu_tree(v) for k,v in value.items()}
    return value

def outputs(model, batch, raw):
    computed = model._compute_embedding_for_batch(batch)
    B,T = batch[0].shape
    src = F.normalize(model.pe_embedding(batch[0]), dim=-1)
    src[:,0,:] = model.cls_token.expand(B,-1)
    mask = batch[5]
    if model.dataset_token is not None:
        src = torch.cat((src, model.dataset_token.expand(B,-1).unsqueeze(1)),dim=1)
        mask = torch.cat((mask, torch.zeros(B,1,device=mask.device,dtype=torch.bool)),dim=1)
    full = model(src, mask, counts=batch[7], dataset_nums=batch[8])
    dataset_latents = model.dataset_embedder(full[2])
    merged = model.resize_batch(full[1], computed[0][0], task_counts=batch[6], ds_emb=dataset_latents)
    return cpu_tree({'computed':computed,'full':full,
        'dataset_classifier':model.dataset_encoder(full[2]), 'dataset_latents':dataset_latents,
        'gene_decoder':model.binary_decoder(merged),
        'raw_forward':model(raw, torch.zeros(raw.shape[:2],dtype=torch.bool,device=raw.device),counts=None),
        'raw_encoder':model.gene_embedding_layer(raw)})

def cases(device):
    gen=torch.Generator().manual_seed(19083)
    shapes=[(1,1,1),(1,8,7),(4,32,13),(2,128,31),(2,512,31),(1,2048,31)]
    shapes += [(1,n,n) for n in (9,10,11,12,13,14,15,16,23,24,25,48,49,50)]
    shapes += [(2,6,6),(3,4,4),(2,7,7),(2,8,8),(4,4,4)]
    for i,(B,T,Q) in enumerate(shapes):
        ids=torch.randint(19790,(B,T*2),generator=gen)[:,::2]; ids[:,0]=3
        tasks=torch.randint(19790,(B,Q*2),generator=gen)[:,::2]
        mask=torch.rand(B,T,generator=gen)>.6
        counts=torch.rand(B,T,generator=gen)*12 if i%2 else None
        batch=(ids,tasks,torch.rand(B,Q,generator=gen),torch.arange(B),torch.rand(B,Q,generator=gen),mask,
               torch.full((B,),4.),counts,torch.arange(B,dtype=torch.int32))
        batch=tuple(v.to(device) if isinstance(v,torch.Tensor) else v for v in batch)
        # Preserve noncontiguous index cases after the host-to-device copy.
        if i%2:
            vals=[]
            for k,v in enumerate(batch):
                if k in (0,1):
                    expanded=torch.empty((*v.shape[:-1],v.shape[-1]*2),dtype=v.dtype,device=device)
                    expanded[...,::2]=v; v=expanded[...,::2]
                vals.append(v)
            batch=tuple(vals)
        raw=torch.randn(B,min(T+1,33),5120,generator=gen).to(device)
        yield {'B':B,'T':T,'Q':Q,'counts':counts is not None,'ids_stride':list(batch[0].stride())},batch,raw
