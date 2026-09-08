"""Independent complete Boltz native gate and interleaved timing of request constants."""
import argparse
import contextlib
import hashlib
import json
import pathlib
import platform
import random
import statistics
import sys
import time
import traceback

import numpy as np
import torch

HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from sharing_probe import batch_from_existing,comparison
from request_constants import constant_conditioning_sampling_scope

def seed(number):
    random.seed(number);np.random.seed(number);torch.manual_seed(number)

def module_snapshot(model):
    return {name:(id(mod),mod.training,id(vars(mod).get('forward')) if 'forward' in vars(mod) else None,
                       id(vars(mod).get('sample')) if 'sample' in vars(mod) else None)
            for name,mod in model.named_modules()}

def parameter_snapshot(model):
    return {name:(id(value),value.data_ptr(),value._version,tuple(value.shape),value.dtype,value.requires_grad)
            for name,value in model.named_parameters(remove_duplicate=False)}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--device',required=True,choices=['cpu','mps'])
    p.add_argument('--mode',choices=['gate','timing'],default='gate')
    p.add_argument('--artifact',default='/private/tmp/compressme-models/boltz2/compiled-bundle')
    p.add_argument('--native-dir',default=str(HERE.parent))
    p.add_argument('--output',required=True)
    p.add_argument('--accepted-gate')
    p.add_argument('--rounds',type=int,default=5)
    p.add_argument('--warmups',type=int,default=2)
    a=p.parse_args()
    torch.set_num_threads(4);torch.set_grad_enabled(False);torch.set_float32_matmul_precision('highest')
    source_hash=hashlib.sha256((HERE/'request_constants.py').read_bytes()).hexdigest()
    report={'args':vars(a),'torch':torch.__version__,'python':sys.version,'platform':platform.platform(),
            'cpu_threads':torch.get_num_threads(),'precision':'original float32, highest matmul precision; no autocast',
            'prototype_sha256':source_hash,'harness_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
            'input_scope':'Same 20-aa protein+ethanol fixture, exact native feature tensors reused within each comparison; fresh diffusion seed each timing round',
            'timing_scope':'Native predict_step, including request preparation and cache cleanup; excludes unchanged input featurization and model load',
            'schedules':{'confidence':{'sampling_steps':200,'recycling_steps':3,'diffusion_samples':1},
                         'affinity':{'sampling_steps':200,'recycling_steps':5,'diffusion_samples':3}},'models':{}}
    stage='load'
    try:
        from compressme import load_boltz2
        if a.mode=='timing':
            if not a.accepted_gate:raise ValueError('Timing requires accepted full native gate')
            gate=json.loads(pathlib.Path(a.accepted_gate).read_text())
            if not gate['accepted_bitwise'] or gate['prototype_sha256']!=source_hash or gate['args']['device']!=a.device:
                raise ValueError('Full-output gate does not match prototype/backend')
        bundle=load_boltz2(a.artifact,device=a.device)
        if any(v.dtype!=torch.float32 for v in bundle.parameters()):raise ValueError('This experiment uses original FP32 weights')
        root=pathlib.Path(a.native_dir)
        batches={}
        for kind,member in [('conf','confidence'),('aff','affinity')]:
            conf_dir=root/f'{a.device}-fp32-default-sampling'
            base=conf_dir if kind=='conf' else root/f'{a.device}-aff-fp32-default-sampling'
            seed(1729)
            dm,batch=batch_from_existing(base,root/'mols',conf_dir/'predictions' if kind=='aff' else None)
            batches[member]=dm.transfer_batch_to_device(batch,torch.device(a.device),0)
        def sync():
            if a.device=='mps':torch.mps.synchronize()
        def run(member,candidate,number):
            seed(number);sync();stats={}
            start=time.perf_counter()
            scope=constant_conditioning_sampling_scope(bundle[member]) if candidate else contextlib.nullcontext(stats)
            with scope as stats:
                output=bundle[member].predict_step(batches[member],0)
            sync();elapsed=time.perf_counter()-start
            if output.get('exception'):raise ValueError('Native model returned exception flag')
            return output,elapsed,stats
        print('MODELS_AND_INPUTS_READY',a.device,a.mode,flush=True)
        for member in ['confidence','affinity']:
            stage=member
            model=bundle[member]
            before_modules=module_snapshot(model);before_params=parameter_snapshot(model)
            if a.mode=='gate':
                original,_,_=run(member,False,1729)
                repeated,_,_=run(member,False,1729)
                control=comparison(original,repeated);del repeated
                candidate,_,stats=run(member,True,1729)
                check=comparison(original,candidate);del candidate
                restored,_,_=run(member,False,1729)
                restore_check=comparison(original,restored)
                del original,restored
                clean=module_snapshot(model)==before_modules and parameter_snapshot(model)==before_params
                requests=stats.get('requests',[])
                cache_clean=bool(requests) and all(x.get('cache_cleared') for x in requests)
                optimized=bool(requests) and all(x.get('optimized_score_calls')==x.get('score_calls') and x.get('score_calls',0)>0 for x in requests)
                data={'self_repeat':control,'candidate':check,'after_restoration':restore_check,
                      'module_and_parameter_identities_restored':clean,'cache_cleared':cache_clean,
                      'all_score_calls_optimized':optimized,'request_statistics':stats,
                      'accepted_bitwise':all(x['all_bitwise_equal'] and x['all_finite'] for x in [control,check,restore_check]) and clean and cache_clean and optimized}
                print('FULL_GATE',member,data['accepted_bitwise'],'MAX_ABS',check['max_abs'],'STATS',stats,flush=True)
            else:
                for index in range(a.warmups):
                    for candidate in [False,True]:run(member,candidate,2000+index)
                rows=[]
                for index in range(a.rounds):
                    order=[False,True] if index%2==0 else [True,False]
                    outputs={};elapsed={};stats={}
                    for candidate in order:
                        output,seconds,details=run(member,candidate,3000+index)
                        key='candidate' if candidate else 'original'
                        outputs[key]=output;elapsed[key]=seconds
                        if candidate:stats=details
                    check=comparison(outputs['original'],outputs['candidate'])
                    if not check['all_bitwise_equal'] or not check['all_finite']:
                        raise ValueError(f'Timing-round output mismatch: {member}/{index}')
                    row={'round':index,'order':['candidate' if v else 'original' for v in order],
                         'seconds':elapsed,'speedup':elapsed['original']/elapsed['candidate'],
                         'bitwise_equal':True,'request_statistics':stats}
                    rows.append(row);del outputs
                    report['models'][member]={'rounds':rows,'completed':False,'accepted_bitwise':False}
                    pathlib.Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
                    print('ROUND',member,index,elapsed,row['speedup'],flush=True)
                med={key:statistics.median(row['seconds'][key] for row in rows) for key in ['original','candidate']}
                data={'rounds':rows,'median_seconds':med,'ratio_of_medians':med['original']/med['candidate'],
                      'module_and_parameter_identities_restored':module_snapshot(model)==before_modules and parameter_snapshot(model)==before_params,
                      'accepted_bitwise':True}
            report['models'][member]=data
            pathlib.Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
        report['accepted_bitwise']=all(v['accepted_bitwise'] for v in report['models'].values())
    except Exception as error:
        report.update(accepted_bitwise=False,failure_stage=stage,error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
        traceback.print_exc()
    pathlib.Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
    print('SAVED',a.output,'ACCEPTED',report['accepted_bitwise'],flush=True)

if __name__=='__main__':main()
