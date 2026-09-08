"""Bounded checkpoint/source inventories; never import or deserialize model code."""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import pickletools
import stat
import struct
from urllib.parse import urlsplit
import zipfile

_HEADER_LIMIT = 16 << 20
_METADATA_LIMIT = 8 << 20
_DTYPE_BITS = {'BOOL':8,'U8':8,'I8':8,'I16':16,'U16':16,'I32':32,'U32':32,
               'I64':64,'U64':64,'F16':16,'BF16':16,'F32':32,'F64':64,
               'F8_E4M3':8,'F8_E5M2':8,'F8_E8M0':8}


def _unique_json(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key!r}')
        result[key] = value
    return result


def _file(path):
    path = Path(path).expanduser()
    if not path.is_file() or not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f'Choose a regular local file: {path}')
    return path


def _hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def inspect_safetensors(path, *, include_tensors=False, include_hash=False,
                        max_header_bytes=_HEADER_LIMIT):
    """Read only a bounded safetensors JSON header; do not allocate tensor data.

    Known dtype sizes, all shapes/offsets, contiguous complete payload coverage,
    duplicate names and metadata types are checked. Unknown future dtypes remain
    visible but explicitly unverified; their value semantics are not guessed.
    """
    if type(max_header_bytes) is not int or not 2 <= max_header_bytes <= _HEADER_LIMIT:
        raise ValueError('max_header_bytes must lie between 2 and 16 MiB')
    path = _file(path)
    size = path.stat().st_size
    with path.open('rb') as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError('Truncated safetensors length header')
        length = int.from_bytes(prefix, 'little')
        if not 2 <= length <= max_header_bytes or length > size - 8:
            raise ValueError('Safetensors JSON header is truncated or exceeds its cap')
        raw = handle.read(length)
    if not raw.startswith(b'{'):
        raise ValueError('Safetensors JSON header must start with an object')
    try:
        header = json.loads(raw, object_pairs_hook=_unique_json)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError('Invalid safetensors JSON header') from error
    if not isinstance(header, dict):
        raise ValueError('Safetensors header must be an object')
    metadata = header.pop('__metadata__', {})
    if not isinstance(metadata, dict) or any(not isinstance(k,str) or not isinstance(v,str) for k,v in metadata.items()):
        raise ValueError('Safetensors metadata must map strings to strings')
    payload = size - 8 - length
    tensors, intervals, counts, byte_counts, unknown = [], [], Counter(), Counter(), set()
    for name, record in header.items():
        if not isinstance(name, str) or not isinstance(record, dict) or set(record) != {'dtype','shape','data_offsets'}:
            raise ValueError('Invalid safetensors tensor record')
        dtype, shape, offsets = record['dtype'], record['shape'], record['data_offsets']
        if not isinstance(dtype,str) or not isinstance(shape,list) or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError(f'Invalid dtype/shape for {name!r}')
        if not isinstance(offsets,list) or len(offsets)!=2 or any(type(n) is not int for n in offsets):
            raise ValueError(f'Invalid data offsets for {name!r}')
        start, end = offsets
        if not 0 <= start <= end <= payload:
            raise ValueError(f'Out-of-range data offsets for {name!r}')
        bits = _DTYPE_BITS.get(dtype)
        elements = 0 if 0 in shape else 1
        if elements:
            limit = max(payload * 8, 1)
            for dimension in shape:
                if dimension > limit // elements:
                    raise ValueError(f'Shape exceeds possible payload size for {name!r}')
                elements *= dimension
        if bits is None:
            unknown.add(dtype)
        elif elements * bits != (end-start) * 8:
            raise ValueError(f'Shape/dtype and stored byte count disagree for {name!r}')
        if end > start:
            intervals.append((start,end))
        tensors.append({'name':name,'dtype':dtype,'shape':shape,'elements':elements,
                        'bytes':end-start,'data_offsets':offsets})
        counts[dtype] += 1
        byte_counts[dtype] += end-start
    cursor = 0
    for start,end in sorted(intervals):
        if start != cursor:
            raise ValueError('Safetensors data overlap or uncovered payload bytes')
        cursor = end
    if cursor != payload:
        raise ValueError('Safetensors contains unreferenced trailing payload bytes')
    result = {'source':str(path.resolve()),'kind':'safetensors','file_bytes':size,
              'header_bytes':length+8,'tensor_bytes':payload,'tensor_count':len(tensors),
              'stored_elements':sum(t['elements'] for t in tensors),'dtype_counts':dict(counts),
              'dtype_bytes':dict(byte_counts),'unverified_dtypes':sorted(unknown),
              'metadata_keys':sorted(metadata),'largest_tensors':sorted(tensors,key=lambda t:t['bytes'],reverse=True)[:10],
              'tensor_values_read':False,'code_executed':False,
              'scope':'Storage metadata only; architecture, biological quality and rewrite eligibility are not inferred'}
    if include_tensors:
        result['tensors'] = tensors
    if include_hash:
        result['sha256'] = _hash(path)
    return result


def _zip_bounds(path):
    """Bound central-directory allocation before calling Python's ZIP reader."""
    size = path.stat().st_size
    with path.open('rb') as handle:
        handle.seek(max(0,size-65557))
        tail = handle.read(65557)
        offset = tail.rfind(b'PK\x05\x06')
        if offset < 0 or len(tail)-offset < 22:
            raise ValueError('Missing ZIP end record')
        record = struct.unpack_from('<4s4H2LH',tail,offset)
        _,disk,start_disk,on_disk,total,cd_bytes,cd_offset,comment = record
        eocd = max(0,size-65557)+offset
        if eocd+22+comment != size or disk or start_disk or on_disk != total:
            raise ValueError('Unsupported multipart ZIP or trailing archive data')
        if total==65535 or cd_bytes==0xffffffff or cd_offset==0xffffffff:
            if eocd < 20:
                raise ValueError('Missing ZIP64 locator')
            handle.seek(eocd-20)
            signature,disk,location,disks=struct.unpack('<4sLQL',handle.read(20))
            if signature!=b'PK\x06\x07' or disk or disks!=1 or not 0<=location<=eocd-76:
                raise ValueError('Invalid ZIP64 locator')
            handle.seek(location)
            values=struct.unpack('<4sQ2H2L4Q',handle.read(56))
            signature,length,_,_,disk,start_disk,on_disk,total,cd_bytes,cd_offset=values
            if signature!=b'PK\x06\x06' or disk or start_disk or on_disk!=total or length<44 or location+12+length!=eocd-20:
                raise ValueError('Invalid ZIP64 directory metadata')
        if total>100000 or cd_bytes>_HEADER_LIMIT or cd_offset+cd_bytes>eocd:
            raise ValueError('ZIP central directory exceeds inspection limits')
    return total


def inspect_checkpoint(path, *, include_tensors=False, include_hash=False):
    """Inspect safetensors or bounded PyTorch ZIP metadata without unpickling.

    Legacy/non-ZIP .pt/.ckpt files remain opaque. ZIP data.pkl is disassembled,
    never executed; its globals are untrusted names, not import instructions.
    Storage/optimizer contents are not relabelled as inference parameters.
    """
    path = _file(path)
    if path.name.endswith('.safetensors'):
        return inspect_safetensors(path,include_tensors=include_tensors,include_hash=include_hash)
    result={'source':str(path.resolve()),'kind':'opaque_checkpoint','file_bytes':path.stat().st_size,
            'code_executed':False,'pickle_deserialized':False,'tensor_values_read':False,
            'scope':'Container metadata only; no inferred tensor shapes, inference size or model graph'}
    with path.open('rb') as handle:
        magic=handle.read(4)
    if magic==b'PK\x03\x04':
        _zip_bounds(path)
        try:
            with zipfile.ZipFile(path) as archive:
                entries=archive.infolist()
                if len({entry.filename for entry in entries})!=len(entries):
                    raise ValueError('Duplicate ZIP member names')
                result.update(kind='pytorch_or_generic_zip',archive_entries=len(entries),
                    archive_uncompressed_bytes=sum(e.file_size for e in entries),
                    largest_members=[{'name':e.filename,'bytes':e.file_size,'compressed_bytes':e.compress_size}
                                     for e in sorted(entries,key=lambda e:e.file_size,reverse=True)[:10]])
                pickles=[e for e in entries if e.filename=='data.pkl' or e.filename.endswith('/data.pkl')]
                if len(pickles)==1:
                    entry=pickles[0]
                    if entry.file_size>_METADATA_LIMIT or entry.compress_size>_METADATA_LIMIT or entry.flag_bits&1:
                        result['pickle_metadata']={'status':'skipped_size_limit_or_encryption'}
                    else:
                        with archive.open(entry) as stream:
                            data=stream.read(_METADATA_LIMIT+1)
                        if len(data)>_METADATA_LIMIT:
                            raise ValueError('Checkpoint pickle metadata exceeds cap')
                        globals_,opcodes,complete=set(),Counter(),False
                        try:
                            for i,(opcode,arg,position) in enumerate(pickletools.genops(data)):
                                if i>=200000:
                                    break
                                opcodes[opcode.name]+=1
                                if opcode.name=='GLOBAL' and len(globals_)<100:
                                    globals_.add(str(arg)[:512])
                                if opcode.name=='STOP':
                                    complete=position+1==len(data)
                            result['pickle_metadata']={'status':'disassembled_only','complete':complete,
                                'opcode_count':sum(opcodes.values()),'global_names':sorted(globals_),
                                'stack_global_opcodes':opcodes['STACK_GLOBAL']}
                        except (ValueError,UnicodeError,RecursionError) as error:
                            result['pickle_metadata']={'status':'unparsed','reason':type(error).__name__}
                else:
                    result['pickle_metadata']={'status':'absent_or_ambiguous'}
        except (zipfile.BadZipFile,RuntimeError,NotImplementedError) as error:
            raise ValueError('Invalid or unsupported checkpoint ZIP') from error
    else:
        result['reason']='Non-ZIP checkpoint left opaque; use safetensors or trusted local conversion code'
    if include_hash:
        result['sha256']=_hash(path)
    return result


def inspect_repository(repository, *, max_files=1000, max_file_bytes=1<<20,
                       max_total_bytes=32<<20):
    """Record a remote URL, or statically inventory a bounded local Python tree.

    URLs are neither fetched nor cloned. Local Python is parsed as AST only;
    no imports, installation, repository scripts or git hooks are run.
    """
    value=str(repository)
    parsed=urlsplit(value)
    if parsed.scheme in ('http','https'):
        if not parsed.netloc or parsed.username or parsed.password or parsed.query:
            raise ValueError('Repository URL must have a host, no embedded credentials and no query string')
        return {'kind':'remote_url','url':value,'fetched':False,'code_executed':False,
                'scope':'Provenance reference only; no source or model API was verified'}
    if parsed.scheme:
        raise ValueError('Use an HTTPS repository URL or a local source directory')
    for number,name in ((max_files,'max_files'),(max_file_bytes,'max_file_bytes'),(max_total_bytes,'max_total_bytes')):
        if type(number) is not int or number<1:
            raise ValueError(f'{name} must be positive')
    root=Path(value).expanduser().resolve()
    if not root.is_dir():
        raise ValueError('Local --repo must be a source directory')
    skip={'.git','.venv','venv','__pycache__','node_modules','.tox','.mypy_cache','.pytest_cache','.ruff_cache'}
    counts,modules,records,issues=Counter(),Counter(),[],[]
    total=0;limited=False
    for base,dirs,files in os.walk(root,followlinks=False):
        dirs[:]=sorted(d for d in dirs if d not in skip and not d.startswith('.venv') and not Path(base,d).is_symlink())
        for name in sorted(files):
            path=Path(base,name)
            if path.suffix!='.py' or path.is_symlink():
                continue
            relative=str(path.relative_to(root))
            if len(records)>=max_files:
                limited=True;break
            size=path.stat().st_size
            if size>max_file_bytes or total+size>max_total_bytes:
                issues.append({'file':relative,'status':'skipped_size_limit'});limited=True;continue
            with path.open('rb') as handle:raw=handle.read(max_file_bytes+1)
            if len(raw)>max_file_bytes or total+len(raw)>max_total_bytes:
                limited=True;continue
            total+=len(raw)
            records.append({'file':relative,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
            try:tree=ast.parse(raw,filename=relative)
            except (SyntaxError,ValueError,RecursionError) as error:
                issues.append({'file':relative,'status':'parse_failed','error':type(error).__name__});continue
            for node in ast.walk(tree):
                if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):counts['functions']+=1
                elif isinstance(node,ast.ClassDef):counts['classes']+=1
                elif isinstance(node,ast.Attribute) and node.attr=='weight':counts['weight_attribute_accesses']+=1
                elif isinstance(node,ast.Call):
                    leaf=node.func.attr if isinstance(node.func,ast.Attribute) else node.func.id if isinstance(node.func,ast.Name) else None
                    if leaf in {'Linear','LayerNorm','Embedding','MultiheadAttention','TransformerEncoderLayer','Conv1d','Conv2d'}:modules[leaf]+=1
                    if leaf in {'eval','exec'}:counts['dynamic_execution_call_sites']+=1
                elif isinstance(node,ast.Constant) and isinstance(node.value,str) and node.value in {'cuda','mps'}:
                    counts[node.value+'_string_literals']+=1
        if len(records)>=max_files:
            break
    return {'kind':'local_directory','path':str(root),'python_files_inspected':len(records),'source_bytes_read':total,
            'counts':dict(counts),'recognised_constructor_call_sites':dict(modules),'files':records,'issues':issues[:50],
            'truncated':limited,'code_executed':False,'symlinks_followed':False,
            'scope':'Bounded AST inventory, not a security audit or proof of rewrite eligibility'}
