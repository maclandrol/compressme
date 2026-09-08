"""Exact value transport for independent, non-gradient CPU tensors.

Coalesce aligned raw bytes into one host-to-device copy, then return contiguous
typed views with original shapes/dtypes. Values are never interpreted or cast.
The returned tensors share one allocation and have disjoint byte ranges.
This utility preserves values, shapes and dtypes, not original strides/aliases.
"""
import torch

def transfer_by_dtype(tensors,device,*,alignment=64):
    """Coalesce each dtype separately; avoids reinterpretation of device views."""
    tensors=list(tensors)
    if not isinstance(alignment,int) or isinstance(alignment,bool) or alignment<1 or alignment&(alignment-1):
        raise ValueError('alignment must be a positive power of two')
    groups={};result=[None]*len(tensors)
    for i,tensor in enumerate(tensors):
        if (not isinstance(tensor,torch.Tensor) or tensor.layout!=torch.strided or tensor.device.type!='cpu'
                or tensor.is_quantized or tensor.is_nested):
            raise ValueError('Expected ordinary dense CPU tensors')
        if tensor.requires_grad or tensor.is_conj() or tensor.is_neg():
            raise ValueError('Gradient and unresolved conjugate/negative views are unsupported')
        if tensor.element_size()>alignment:
            raise ValueError('Alignment is smaller than the tensor element size')
        groups.setdefault(tensor.dtype,[]).append((i,tensor))
    for dtype,group in groups.items():
        parts=[];metadata=[];offset=0;boundary=alignment//group[0][1].element_size()
        for i,tensor in group:
            padding=(-offset)%boundary
            if padding:parts.append(torch.zeros(padding,dtype=dtype));offset+=padding
            flat=tensor.contiguous().reshape(-1);parts.append(flat)
            metadata.append((i,offset,flat.numel(),tensor.shape));offset+=flat.numel()
        combined=torch.cat(parts).to(device)
        for i,start,size,shape in metadata:result[i]=combined.narrow(0,start,size).reshape(shape)
    return result

def transfer_tensors(tensors,device,*,alignment=64):
    tensors=list(tensors)
    if not isinstance(alignment,int) or isinstance(alignment,bool) or alignment<1 or alignment&(alignment-1):
        raise ValueError('alignment must be a positive power of two')
    if not tensors:return []
    metadata=[];chunks=[];offset=0
    for tensor in tensors:
        if (not isinstance(tensor,torch.Tensor) or tensor.layout!=torch.strided or tensor.device.type!='cpu'
                or tensor.is_quantized or tensor.is_nested):
            raise ValueError('Expected ordinary dense CPU tensors')
        if tensor.requires_grad or tensor.is_conj() or tensor.is_neg():
            raise ValueError('Gradient and unresolved conjugate/negative views are unsupported')
        if tensor.element_size()>alignment:
            raise ValueError('Alignment is smaller than the tensor element size')
        padding=(-offset)%alignment
        if padding:chunks.append(torch.zeros(padding,dtype=torch.uint8));offset+=padding
        raw=tensor.contiguous().reshape(-1).view(torch.uint8)
        metadata.append((offset,raw.numel(),tensor.shape,tensor.dtype))
        chunks.append(raw);offset+=raw.numel()
    combined=torch.cat(chunks).to(device)
    return [combined.narrow(0,start,size).view(dtype).reshape(shape)
            for start,size,shape,dtype in metadata]
