"""Download pinned official checkpoints, streaming SHA256; never deserialize them."""
import argparse,concurrent.futures,hashlib,json,os,urllib.request
from pathlib import Path
REVISION='6fdef46d763fee7fbb83ca5501ccceff43b85607'
FILES={'boltz2_conf.ckpt':(2286561469,'090e82ac8c92f5e943fa1b39e7410a44027bea7243c0bbb3caa67a77fc1428e1'),'boltz2_aff.ckpt':(2062139170,'dcc5cd3722b1c9eaa34267e4ae32f55cbbf1963f4c19319381ccfa30fdd2ca9e')}
def fetch(directory,name):
 size,expected=FILES[name];target=directory/name;partial=directory/(name+'.part');digest=hashlib.sha256();written=0
 if target.exists():
  with target.open('rb') as handle:
   while block:=handle.read(4*1024*1024):digest.update(block);written+=len(block)
  if written!=size or digest.hexdigest()!=expected:raise ValueError(f'Existing file hash mismatch: {name}')
  return {'name':name,'bytes':written,'sha256':digest.hexdigest(),'verified':True,'downloaded':False}
 url=f'https://huggingface.co/boltz-community/boltz-2/resolve/{REVISION}/{name}?download=true'
 with urllib.request.urlopen(url,timeout=120) as response,partial.open('wb') as handle:
  if response.status!=200:raise ValueError('Expected full-file200 response')
  while block:=response.read(4*1024*1024):
   handle.write(block);digest.update(block);written+=len(block)
   if written>size:raise ValueError('Publisher size exceeded')
   if written%(256*1024*1024)<len(block):print(f'{name}: {written}/{size} bytes',flush=True)
 if written!=size or digest.hexdigest()!=expected:raise ValueError(f'Download size/hash mismatch: {name}')
 os.replace(partial,target)
 return {'name':name,'bytes':written,'sha256':digest.hexdigest(),'verified':True,'downloaded':True}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--directory',type=Path,required=True);args=ap.parse_args();args.directory.mkdir(parents=True,exist_ok=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(lambda name:fetch(args.directory,name),FILES))
 result={'repository':'boltz-community/boltz-2','revision':REVISION,'files':results,'deserialization':'none'}
 (args.directory/'download-verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
