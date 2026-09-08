"""Safe Boltz ZIP tensor export and exact cross-checkpoint byte deduplication.

The checkpoint pickle is interpreted only by the adjacent inert opcode reader.
No torch.load, pickle.Unpickler, checkpoint globals, reducers or constructors run.
All state tensors in the pinned checkpoints must be full contiguous float32
storages. This exporter refuses other layouts rather than guessing.
"""
import argparse,contextlib,hashlib,json,math,mmap,os,struct,sys,zipfile
from pathlib import Path
from static_metadata import parse,plain,Tensor,Global
from download_checkpoints import FILES,REVISION

def chunks(view,n=4*1024*1024):
 for i in range(0,len(view),n):yield view[i:i+n]
def sha_file(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  while b:=f.read(4*1024*1024):h.update(b)
 return h.hexdigest()
def payload_offset(mm,info):
 if info.compress_type!=0:raise ValueError('Compressed ZIP member unsupported')
 off=info.header_offset
 if mm[off:off+4]!=b'PK\x03\x04':raise ValueError('Invalid ZIP local header')
 nlen,xlen=struct.unpack_from('<HH',mm,off+26)
 start=off+30+nlen+xlen
 if start+info.file_size>len(mm):raise ValueError('ZIP payload outside file')
 return start

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--checkpoints',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
 target=args.output/'shared.safetensors'
 if target.exists():raise ValueError('Refusing to overwrite existing shared artifact')
 catalog={};models={};upstream={};maps={};report={}
 with contextlib.ExitStack() as stack:
  for name,(size,expected) in FILES.items():
   path=args.checkpoints/name
   if path.stat().st_size!=size or sha_file(path)!=expected:raise ValueError(f'Publisher checkpoint hash failed: {name}')
   print(f'Verified complete publisher SHA256: {name}',flush=True)
   handle=stack.enter_context(path.open('rb'));mm=stack.enter_context(mmap.mmap(handle.fileno(),0,access=mmap.ACCESS_READ));maps[name]=mm
   with zipfile.ZipFile(handle) as archive:
    pkl_info=next(i for i in archive.infolist() if i.filename.endswith('/data.pkl'))
    if pkl_info.file_size>2*1024*1024:raise ValueError('Metadata size exceeds audit budget')
    parsed=parse(archive.read(pkl_info));prefix=pkl_info.filename.removesuffix('data.pkl')
    if archive.read(prefix+'byteorder')!=b'little':raise ValueError('Only little-endian checkpoint storage supported')
    state=parsed['state_dict'];bindings={};seen_storages={};original_named=0;storage_records={}
    for key,value in state.items():
     if not isinstance(value,Tensor):raise ValueError('Non-tensor state')
     _,kind,storage_key,location,storage_numel=value.storage.descriptor
     stride=[];n=1
     for dim in reversed(value.shape):stride.append(n);n*=dim
     if not isinstance(kind,Global) or kind.name!='torch FloatStorage' or value.offset!=0 or n!=storage_numel or list(value.stride)!=stride[::-1]:raise ValueError('Unsupported tensor storage/layout')
     if storage_key in seen_storages:blob=seen_storages[storage_key]
     else:
      info=archive.getinfo(prefix+'data/'+storage_key)
      if info.file_size!=n*4:raise ValueError('Tensor storage byte size mismatch')
      start=payload_offset(mm,info);h=hashlib.sha256()
      for data in chunks(memoryview(mm)[start:start+n*4]):h.update(data)
      del data
      blob=h.hexdigest();seen_storages[storage_key]=blob;storage_records[storage_key]={'start':start,'bytes':n*4,'blob':blob}
      if blob not in catalog:catalog[blob]={'checkpoint':name,'offset':start,'bytes':n*4,'values':n}
      else:
       previous=catalog[blob]
       if previous['bytes']!=n*4:raise ValueError('Hash collision with differing byte lengths')
       other=maps[previous['checkpoint']]
       for block_start in range(0,n*4,4*1024*1024):
        count=min(4*1024*1024,n*4-block_start)
        if mm[start+block_start:start+block_start+count]!=other[previous['offset']+block_start:previous['offset']+block_start+count]:raise ValueError('SHA match failed exact byte comparison')
     bindings[key]={'tensor':blob,'shape':list(value.shape)};original_named+=n
    config=plain(parsed['hyper_parameters']);(args.output/(name+'.hparams.json')).write_text(json.dumps(config,indent=2)+'\n')
    unique=set(v['tensor'] for v in bindings.values());models[name]={'bindings':bindings,'hyper_parameters':name+'.hparams.json'}
    upstream[name]={'file_bytes':size,'sha256':expected,'publisher_hash_verified':True,'state_dict_named_values':original_named,'state_dict_storage_values':sum(r['bytes']//4 for r in storage_records.values()),'independently_byte_deduplicated_inference_values':sum(catalog[h]['values'] for h in unique)}
    print(f'Indexed {name}: {len(bindings)} named tensors, {len(unique)} unique byte blobs',flush=True)
  conf=models['boltz2_conf.ckpt']['bindings'];aff=models['boltz2_aff.ckpt']['bindings'];common=sorted(set(conf)&set(aff));equal=[key for key in common if conf[key]==aff[key]];different=[key for key in common if conf[key]!=aff[key]]
  report={'repository':'boltz-community/boltz-2','revision':REVISION,'checkpoint_metadata_reader':'inert symbolic opcode interpreter, no unpickling','upstream':upstream,'equality_check':'SHA256 buckets followed by exact byte comparison of every reused blob','common_tensor_names':len(common),'bitwise_equal_common_tensor_names':len(equal),'different_common_tensor_names':different,'common_named_equal_values':sum(math.prod(conf[k]['shape']) for k in equal),'independent_deduplicated_inference_values':sum(v['independently_byte_deduplicated_inference_values'] for v in upstream.values()),'joint_unique_values':sum(v['values'] for v in catalog.values()),'joint_unique_blob_count':len(catalog),'storage_scope':'Joint inference tensor distribution; no claim of less arithmetic or shared resident model parameters.'}
  report['joint_saved_values_over_independent_deduplicated_inference']=report['independent_deduplicated_inference_values']-report['joint_unique_values'];report['joint_fraction_saved_over_independent_deduplicated_inference']=report['joint_saved_values_over_independent_deduplicated_inference']/report['independent_deduplicated_inference_values']
  header={};offset=0
  for key,record in catalog.items():header[key]={'dtype':'F32','shape':[record['values']],'data_offsets':[offset,offset+record['bytes']]};offset+=record['bytes']
  header['__metadata__']={'format':'compressme-boltz2-shared-state-audit-v1','revision':REVISION};encoded=json.dumps(header,separators=(',',':')).encode();encoded+=b' '*((-len(encoded))%8)
  part=target.with_suffix('.safetensors.part');digest=hashlib.sha256()
  with part.open('wb') as out:
   prefix=struct.pack('<Q',len(encoded))+encoded;out.write(prefix);digest.update(prefix)
   for record in catalog.values():
    mm=maps[record['checkpoint']]
    for data in chunks(memoryview(mm)[record['offset']:record['offset']+record['bytes']]):out.write(data);digest.update(data)
    del data
  os.replace(part,target);report['artifact_tensor_file_bytes']=target.stat().st_size;report['artifact_tensor_file_sha256']=digest.hexdigest()
  manifest={'format':'compressme-boltz2-shared-state-audit-v1','tensor_file':'shared.safetensors','tensor_file_sha256':digest.hexdigest(),'repository':'boltz-community/boltz-2','revision':REVISION,'models':models,'source_hparams_sha256':{name:sha_file(args.output/(name+'.hparams.json')) for name in models}}
  (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(args.output/'export-report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='different_common_tensor_names'},indent=2))
if __name__=='__main__':main()
