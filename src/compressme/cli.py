"""Conservative command-line entry points with explicit local-code boundaries."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile


def _json_write(path, report):
    if path is None:
        return
    path=Path(path).expanduser()
    if os.path.lexists(path):
        raise FileExistsError(f'Report already exists: {path}')
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,temporary=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as stream:
            json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
        os.link(temporary,path)
    finally:
        if os.path.lexists(temporary):os.unlink(temporary)


def _source_kind(source):
    if source.startswith('hf://'):
        return 'hf',source[5:]
    path=Path(source).expanduser()
    if path.exists() or path.is_absolute() or source.startswith(('.', '~')) or source.endswith(('.safetensors','.ckpt','.pt','.pth','.bin','.json')):
        return 'local',str(path)
    if re.fullmatch(r'[\w.-]+/[\w.-]+',source):
        return 'hf',source
    raise ValueError('Choose a local checkpoint file or a Hugging Face OWNER/MODEL (optionally hf://)')


def _repository(args):
    if args.repo is None:return None
    from .checkpoint import inspect_repository
    return inspect_repository(args.repo)


def _inspection(args):
    from .checkpoint import inspect_checkpoint
    kind,source=_source_kind(args.source)
    if kind=='local':
        if args.filename is not None:
            raise ValueError('--filename selects a Hugging Face checkpoint, not a local path')
        report=inspect_checkpoint(source,include_tensors=args.details,include_hash=args.sha256)
    else:
        from .hub import inspect_huggingface
        report=inspect_huggingface(source,revision=args.revision,filename=args.filename)
    repository=_repository(args)
    if repository is not None:report['repository']=repository
    return report


def _torch():
    try:
        import torch
        import safetensors.torch
    except ImportError as error:
        raise ImportError("Model rewriting requires pip install 'compressme[torch]'.") from error
    return torch


def _device(torch, name):
    device=torch.device(name)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA was requested but is unavailable in this Python environment')
    if device.type=='mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS was requested but is unavailable in this Python environment')
    return device


@contextmanager
def _local_factories(model_spec,validation_spec):
    """Explicit trusted local code; --repo is deliberately not an import root."""
    from .checkpoint import _file
    parsed=[]
    for spec in (model_spec,validation_spec):
        if ':' not in spec:
            raise ValueError('Factories use /path/to/local.py:callable syntax')
        filename,name=spec.rsplit(':',1)
        path=_file(filename).resolve()
        if path.suffix!='.py' or not name.isidentifier():
            raise ValueError('Factories require a local .py file and one callable identifier')
        parsed.append((path,name))
    previous=sys.path[:]
    modules={};hashes={};provenance=[];previous_modules={}
    try:
        for path,name in parsed:
            if str(path.parent) not in sys.path:sys.path.insert(0,str(path.parent))
            if path not in modules:
                with path.open('rb') as handle:code=handle.read((16<<20)+1)
                if len(code)>16<<20:raise ValueError('Local factory source exceeds 16 MiB')
                hashes[path]=hashlib.sha256(code).hexdigest()
                module_name='_compressme_local_'+hashlib.sha256(str(path).encode()).hexdigest()[:16]
                specification=importlib.util.spec_from_file_location(module_name,path)
                module=importlib.util.module_from_spec(specification)
                previous_modules[module_name]=sys.modules.get(module_name)
                sys.modules[module_name]=module
                exec(compile(code,str(path),'exec'),module.__dict__)
                modules[path]=module
            function=getattr(modules[path],name,None)
            if not callable(function):raise ValueError(f'Factory is not callable: {path}:{name}')
            provenance.append({'file':str(path),'sha256':hashes[path],'callable':name})
        yield (getattr(modules[parsed[0][0]],parsed[0][1]),
               getattr(modules[parsed[1][0]],parsed[1][1]),provenance)
    finally:
        sys.path[:]=previous
        for name,prior in previous_modules.items():
            if prior is None:sys.modules.pop(name,None)
            else:sys.modules[name]=prior


def _move(value,device,torch):
    if isinstance(value,torch.Tensor):return value.to(device=device)
    if type(value) is tuple:return tuple(_move(v,device,torch) for v in value)
    if type(value) is list:return [_move(v,device,torch) for v in value]
    if type(value) is dict:return {k:_move(v,device,torch) for k,v in value.items()}
    if value is None or type(value) in (str,bool,int,float):return value
    raise TypeError('Validation inputs must contain tensors, plain containers and scalar values')


def _weights(args):
    from .checkpoint import inspect_safetensors,_hash
    from .hub import _load_tensors
    kind,source=_source_kind(args.source)
    if kind=='local':
        if args.filename is not None:raise ValueError('--filename is for Hugging Face inputs')
        path=Path(source)
        if path.suffix!='.safetensors':
            raise ValueError('Local semantic rewriting accepts a single safetensors file; pickle checkpoints are metadata-only')
        inventory=inspect_safetensors(path)
        from safetensors.torch import load_file
        return load_file(str(path),device='cpu'),{'kind':'local','file':str(path.resolve()),'sha256':_hash(path),'file_bytes':inventory['file_bytes']}
    from .hub import inspect_huggingface
    inspection=inspect_huggingface(source,revision=args.revision,filename=args.filename)
    if not inspection['selected']:
        raise ValueError('Choose --filename explicitly from checkpoint candidates: '+str(inspection['candidates']))
    if any(not name.endswith('.safetensors') for name in inspection['weight_files']):
        raise ValueError('Semantic CLI rewriting accepts safetensors only; inspect .ckpt metadata or use trusted conversion code')
    state,records=_load_tensors(inspection,None,None)
    return state,{'kind':'huggingface','repo_id':source,'revision':inspection['revision'],
                  'selected':inspection['selected'],'files':records,'remote_code_executed':False}


def _save_result(result,destination):
    """Build privately; reserve a new directory and atomically publish on POSIX."""
    destination=Path(destination).expanduser()
    if os.path.lexists(destination):raise FileExistsError(f'Output already exists: {destination}')
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=Path(tempfile.mkdtemp(prefix='.'+destination.name+'.',dir=destination.parent))
    reserved=False
    try:
        result.save(temporary)
        destination.mkdir()  # Exclusive reservation, including a racing writer.
        reserved=True
        os.replace(temporary,destination)
        reserved=False
    finally:
        if temporary.exists():shutil.rmtree(temporary)
        if reserved:
            try:destination.rmdir()  # Never remove a nonempty directory.
            except OSError:pass


def _semantic(args):
    torch=_torch()
    from .compiler import CompressionResult,parameter_count,state_bytes
    from .hub import _strict_load
    from .validation import Example,seeded,validate
    output=Path(args.output).expanduser()
    if os.path.lexists(output):raise FileExistsError(f'Output already exists: {output}')
    if args.report and (Path(args.report).expanduser().resolve()==output.resolve() or output.resolve() in Path(args.report).expanduser().resolve().parents):
        raise ValueError('--report must be outside the exported model directory')
    device=_device(torch,args.device)
    repository=_repository(args)
    state,weights=_weights(args)
    with _local_factories(args.factory,args.validation) as (make_model,make_examples,factories):
        with seeded(args.seed):model=make_model()
        loading=_strict_load(model,state)
        del state
        model.eval().requires_grad_(False)
        model=model.to(device=device)  # Device transfer only; never change precision.
        with seeded(args.seed):supplied=list(make_examples())
        if not supplied or any(type(e) is not Example for e in supplied):
            raise ValueError('Validation factory must return a nonempty iterable of plain compressme.Example objects')
        examples=[Example(_move(e.args,device,torch),_move(e.kwargs,device,torch)) for e in supplied]
        kwargs={'relative_tolerance':args.relative_tolerance,'absolute_tolerance':args.absolute_tolerance,'seed':args.seed}
        self_repeat=validate(model,model,examples,**kwargs)
        if args.method=='share' and not self_repeat['bitwise_identical']:
            self_repeat['accepted']=False
            self_repeat['required_bitwise']=True
        if not self_repeat['accepted']:
            return {'status':'rejected','reason':'Original self-repeat fails the requested output tolerances',
                    'validation':self_repeat,'artifact_written':False},2
        if args.method=='affine':
            from .affine import compile_affine
            result=compile_affine(model)
        elif args.method=='share':
            from .sharing import share_frozen_parameters
            result=share_frozen_parameters(model)
        else:
            from .constant_embeddings import deduplicate_embeddings
            candidate,details=deduplicate_embeddings(model)
            candidate._compressme_source_tensor_dtypes={name:str(t.dtype).removeprefix('torch.') for name,t in model.state_dict().items()}
            details.update(parameters_before=parameter_count(model),parameters_after=parameter_count(candidate),
                           tensor_bytes_before=state_bytes(model),tensor_bytes_after=state_bytes(candidate))
            result=CompressionResult(candidate,details)
        try:
            gate=validate(model,result.model,examples,**kwargs)
        except (ValueError,TypeError,RuntimeError) as error:
            gate={'accepted':False,'error':str(error),'scope':'All-output validation failed'}
        if args.method=='share':
            gate['required_bitwise']=True
            gate['accepted']=gate['accepted'] and gate.get('bitwise_identical',False)
        execution={'float32_matmul_precision':torch.get_float32_matmul_precision(),
                   'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
                   'deterministic_warn_only':torch.is_deterministic_algorithms_warn_only_enabled(),
                   'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
                   'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,
                   'cudnn_deterministic':torch.backends.cudnn.deterministic,
                   'cudnn_benchmark':torch.backends.cudnn.benchmark,
                   'autocast_cpu':torch.is_autocast_enabled('cpu'),
                   'autocast_selected_device':torch.is_autocast_enabled(device.type)}
        provenance={'weights':weights,'factories':factories,'repository':repository,'loading':loading,
                    'local_python_executed':True,'repository_python_automatically_executed':False,
                    'device':str(device),'torch_version':torch.__version__,
                    'parameter_dtypes':sorted({str(p.dtype) for p in model.parameters()}),
                    'precision_changed':False,'training':False,'seed':args.seed,'execution_settings':execution}
        result.report.update(validation=gate,self_repeat=self_repeat,cli_provenance=provenance)
        if not gate['accepted']:
            result.report.update(status='rejected_no_artifact',artifact_written=False)
            return result.report,2
        before=result.report.get('resident_storage_bytes_before',result.report.get('tensor_bytes_before'))
        after=result.report.get('resident_storage_bytes_after',result.report.get('tensor_bytes_after'))
        reduced=before is not None and after is not None and after<before
        result.report.update(status='accepted_on_validation_examples' if reduced else 'accepted_no_reduction',
                             reduction_observed=reduced,artifact_written=True)
        _save_result(result,output)
        result.report['artifact_directory']=str(output.resolve())
        return result.report,0


def _small(report):
    if not isinstance(report,dict):return report
    if 'cli_provenance' not in report:
        result=dict(report)
        if isinstance(result.get('repository'),dict):
            result['repository']={k:v for k,v in result['repository'].items() if k not in ('files','issues')}
        return result
    result={key:report[key] for key in ('status','method','artifact_written','artifact_directory',
            'reduction_observed','parameters_before','parameters_after','tensor_bytes_before','tensor_bytes_after',
            'resident_storage_bytes_before','resident_storage_bytes_after','saved_resident_storage_bytes') if key in report}
    result['device']=report['cli_provenance']['device']
    result['precision_changed']=False
    for key in ('self_repeat','validation'):
        gate=report[key]
        summary={k:v for k,v in gate.items() if k!='metrics'}
        metrics=[leaf for example in gate.get('metrics',[]) for leaf in example.values()]
        if metrics:
            summary['max_abs']=max(x['max_abs'] for x in metrics)
            summary['max_relative_l2']=max(x['relative_l2'] for x in metrics)
            summary['tensor_outputs_checked']=len(metrics)
        result[key]=summary
    return result


def _finite(value):
    parsed=float(value)
    if not math.isfinite(parsed) or parsed<0:raise argparse.ArgumentTypeError('Use a finite nonnegative tolerance')
    return parsed


def parser():
    result=argparse.ArgumentParser(prog='compressme',description='Inspect safely, pack losslessly, or validate explicit model rewrites.')
    commands=result.add_subparsers(dest='command',required=True)
    inspect=commands.add_parser('inspect',help='Inspect a local checkpoint or Hugging Face repo without executing model code')
    inspect.add_argument('source');inspect.add_argument('--revision',default='main');inspect.add_argument('--filename')
    inspect.add_argument('--repo',help='Source provenance: URL reference or bounded local AST inventory; never an import root')
    inspect.add_argument('--details',action='store_true',help='Include every local safetensors tensor record')
    inspect.add_argument('--sha256',action='store_true',help='Also stream-hash a local file (reads all bytes)')
    inspect.add_argument('--report',help='Write a new full JSON report; existing reports are never overwritten')
    targets=commands.add_parser('targets',help='List surveyed/validated target statuses')
    targets.add_argument('name',nargs='?');targets.add_argument('--include-inactive',action='store_true')
    for command in ('pack','unpack'):
        p=commands.add_parser(command,help='Lossless file '+('compression' if command=='pack' else 'reconstruction'))
        p.add_argument('source');p.add_argument('output');p.add_argument('--overwrite',action='store_true')
        p.add_argument('--report',help='Write a new JSON report')
        if command=='pack':
            p.add_argument('--level',type=int,default=9);p.add_argument('--group-size',type=int,default=4)
            p.add_argument('--block-size',type=int,default=1<<20)
        else:
            p.add_argument('--max-output-bytes',type=int,required=True,help='Finite maximum reconstructed file size')
            p.add_argument('--max-window-bytes',type=int,default=64<<20)
    p=commands.add_parser('compress',help='Execute trusted local factories, validate a model rewrite, then export')
    p.add_argument('source');p.add_argument('output')
    p.add_argument('--factory',required=True,help='Trusted local.py:make_model; executes Python')
    p.add_argument('--validation',required=True,help='Trusted local.py:make_examples; returns Example objects')
    p.add_argument('--method',choices=('affine','constant_embeddings','share'),default='affine')
    p.add_argument('--device',choices=('cpu','mps','cuda'),default='cpu')
    p.add_argument('--revision',default='main');p.add_argument('--filename');p.add_argument('--repo')
    p.add_argument('--relative-tolerance',type=_finite,default=1e-5);p.add_argument('--absolute-tolerance',type=_finite,default=1e-5)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--report')
    return result


def main(argv=None):
    args=parser().parse_args(argv)
    try:
        if getattr(args,'report',None):
            report_path=Path(args.report).expanduser()
            if os.path.lexists(report_path):raise FileExistsError(f'Report already exists: {report_path}')
            for name in ('source','output'):
                value=getattr(args,name,None)
                if value and report_path.resolve()==Path(value).expanduser().resolve():
                    raise ValueError('Report must differ from source/output paths')
        code=0
        if args.command=='inspect':report=_inspection(args)
        elif args.command=='targets':
            from .targets import get_target,list_targets
            report=get_target(args.name) if args.name else list_targets(include_inactive=args.include_inactive)
        elif args.command in ('pack','unpack'):
            from .packing_files import pack_file,unpack_file
            if args.command=='pack':
                report=pack_file(args.source,args.output,level=args.level,group_size=args.group_size,
                                 block_size=args.block_size,overwrite=args.overwrite)
            else:
                report=unpack_file(args.source,args.output,max_output_bytes=args.max_output_bytes,
                                   max_window_bytes=args.max_window_bytes,overwrite=args.overwrite)
            report.update(status='verified_lossless_file',output=str(Path(args.output).resolve()),
                          scope='Disk/transport only; no resident-model memory or inference-speed reduction')
        else:report,code=_semantic(args)
        _json_write(getattr(args,'report',None),report)
        print(json.dumps(_small(report),indent=2,allow_nan=False))
        return code
    except (ValueError,TypeError,OSError,ImportError,RuntimeError,KeyError) as error:
        print(f'compressme: {type(error).__name__}: {error}',file=sys.stderr)
        return 2
