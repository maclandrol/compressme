"""Interpret a bounded pickle opcode stream as inert metadata only.

Never import a pickle global or call a reducer/constructor. GLOBAL, REDUCE,
NEWOBJ, BUILD and persistent storage references become local data records.
This reader covers only the explicitly enumerated opcodes in these checkpoints.
"""
from dataclasses import dataclass
from pathlib import Path
import collections
import hashlib
import json
import math
import pickletools

@dataclass(frozen=True)
class Global:
    name:str

@dataclass
class Box:
    cls:object
    args:object
    state:object=None

@dataclass
class Storage:
    descriptor:tuple

@dataclass
class Tensor:
    storage:Storage
    offset:int
    shape:tuple
    stride:tuple
    requires_grad:bool


def parse(data):
    stack=[];memo={};mark=object()
    def popmark():
        i=len(stack)-1
        while stack[i] is not mark:i-=1
        values=stack[i+1:];del stack[i:]
        return values
    for index,(op,arg,pos) in enumerate(pickletools.genops(data)):
        if index>500000:raise ValueError('Opcode budget exceeded')
        name=op.name
        if name=='PROTO':
            if arg!=2:raise ValueError('Only the observed protocol2 is supported')
        elif name=='MARK':stack.append(mark)
        elif name in ('BINUNICODE','BININT1','BININT2','BININT','BINFLOAT'):stack.append(arg)
        elif name=='NONE':stack.append(None)
        elif name in ('NEWFALSE','NEWTRUE'):stack.append(name=='NEWTRUE')
        elif name=='EMPTY_DICT':stack.append({})
        elif name=='EMPTY_LIST':stack.append([])
        elif name=='EMPTY_TUPLE':stack.append(())
        elif name=='GLOBAL':stack.append(Global(arg))
        elif name in ('BINPUT','LONG_BINPUT'):memo[arg]=stack[-1]
        elif name in ('BINGET','LONG_BINGET'):stack.append(memo[arg])
        elif name=='TUPLE':stack.append(tuple(popmark()))
        elif name in ('TUPLE1','TUPLE2','TUPLE3'):
            n=int(name[-1]);values=stack[-n:];del stack[-n:];stack.append(tuple(values))
        elif name=='BINPERSID':stack.append(Storage(stack.pop()))
        elif name=='REDUCE':
            args=stack.pop();fn=stack.pop()
            if not isinstance(fn,Global):raise ValueError('Unexpected symbolic reducer')
            if fn.name in ('collections OrderedDict','collections defaultdict'):
                if fn.name=='collections OrderedDict' and args:
                    raise ValueError('Unexpected OrderedDict constructor arguments')
                stack.append({})
            elif fn.name=='torch._utils _rebuild_tensor_v2':
                storage,offset,shape,stride,grad,*unused=args
                if not isinstance(storage,Storage):raise ValueError('Expected inert storage record')
                stack.append(Tensor(storage,offset,shape,stride,grad))
            else:stack.append(Box(fn,args))
        elif name=='NEWOBJ':
            args=stack.pop();cls=stack.pop();stack.append(Box(cls,args))
        elif name=='BUILD':
            state=stack.pop();obj=stack[-1]
            if not isinstance(obj,Box):raise ValueError('Expected inert object state')
            obj.state=state
        elif name=='SETITEM':
            value=stack.pop();key=stack.pop();stack[-1][key]=value
        elif name=='SETITEMS':
            items=popmark()
            if len(items)%2:raise ValueError('Odd dictionary items')
            stack[-1].update(zip(items[::2],items[1::2]))
        elif name=='APPEND':
            value=stack.pop();stack[-1].append(value)
        elif name=='APPENDS':
            values=popmark();stack[-1].extend(values)
        elif name=='STOP':
            if len(stack)!=1:raise ValueError('Unexpected final stack')
            return stack[0]
        else:raise ValueError(f'Unsupported opcode: {name}')
    raise ValueError('Missing STOP')


def plain(value,seen=None):
    if seen is None:seen=set()
    if value is None or isinstance(value,(str,int,float,bool)):return value
    if id(value) in seen:return '<cycle>'
    seen=seen|{id(value)}
    if isinstance(value,Global):return {'symbolic_global':value.name}
    if isinstance(value,Box):
        state=value.state
        if isinstance(state,dict):
            if '_content' in state:return plain(state['_content'],seen)
            if '_val' in state:return plain(state['_val'],seen)
        return {'symbolic_object':plain(value.cls,seen),'state':plain(state,seen)}
    if isinstance(value,dict):return {str(k):plain(v,seen) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [plain(v,seen) for v in value]
    return str(type(value))


def audit(root,name):
    payload=(root/(name+'.data.pkl')).read_bytes()
    result=parse(payload)
    state=result['state_dict']
    records=[];groups=collections.defaultdict(list);breakdown=collections.Counter()
    for key,value in state.items():
        if not isinstance(value,Tensor):raise ValueError(f'Non-tensor state entry: {key}')
        _,kind,storage_key,location,storage_numel=value.storage.descriptor
        if not isinstance(kind,Global) or kind.name!='torch FloatStorage':raise ValueError('Unexpected storage dtype')
        record={'name':key,'dtype':'float32','shape':value.shape,'stride':value.stride,'offset':value.offset,
                'numel':math.prod(value.shape),'storage_key':storage_key,'storage_numel':storage_numel,
                'storage_device_label':location,'requires_grad_flag':value.requires_grad}
        records.append(record);groups[storage_key].append(record)
        breakdown[key.split('.')[0]]+=record['numel']
    aliases=[{'storage_key':k,'names':[r['name'] for r in v],'views':[{'offset':r['offset'],'shape':r['shape'],'stride':r['stride']} for r in v]} for k,v in groups.items() if len(v)>1]
    summary={'status':'static_opcode_metadata_only_no_unpickling_or_model_execution',
        'pickle_bytes':len(payload),'pickle_sha256':hashlib.sha256(payload).hexdigest(),
        'top_level_keys':list(result),'tensor_entries':len(records),'named_tensor_values':sum(v['numel'] for v in records),
        'unique_state_storage_values':sum(v[0]['storage_numel'] for v in groups.values()),
        'named_values_by_top_module':dict(breakdown),'storage_alias_groups':aliases,
        'hyper_parameters':plain(result.get('hyper_parameters'))}
    (root/(name+'.tensor-metadata.json')).write_text(json.dumps(records,indent=2)+'\n')
    (root/(name+'.summary.json')).write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('storage_alias_groups','hyper_parameters')},indent=2))


if __name__=='__main__':
    root=Path(__file__).parent
    for name in ('boltz2_conf.ckpt','boltz2_aff.ckpt'):audit(root,name)
