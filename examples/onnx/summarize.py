"""Summarize an existing ONNX comparison without executing a model or timing it."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1 << 20),b''):digest.update(block)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--models',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Use a new summary path')
    import onnx
    data=json.loads(args.report.read_text())
    if data.get('status')!='accepted':raise ValueError('Expected an accepted comparison report')
    metrics=[]
    def collect(value):
        if isinstance(value,dict):
            if {'max_abs','relative_l2','bitwise','accepted'}.issubset(value):metrics.append(value)
            else:
                for child in value.values():collect(child)
        elif isinstance(value,list):
            for child in value:collect(child)
    collect(data)
    result={'report_file':args.report.name,'report_sha256':sha256(args.report),
            'benchmark_script_sha256':data['script_sha256'],'summary_script_sha256':sha256(__file__),
            'tensor_comparisons':len(metrics),'all_accepted':all(v['accepted'] for v in metrics),
            'byte_equal_comparisons':sum(v['bitwise'] for v in metrics),
            'max_absolute_error':max(v['max_abs'] for v in metrics),
            'max_relative_l2_error':max(v['relative_l2'] for v in metrics),
            'exports':{},'timing':{}}
    for name,item in data['exports'].items():
        files=item['onnx']['files'];folder=args.models/(name+'_'+item['accepted_exporter'])
        for filename,record in files.items():
            path=folder/filename
            if path.stat().st_size!=record['bytes'] or sha256(path)!=record['sha256']:
                raise ValueError('Export bytes differ from measured report: '+str(path))
        graph=onnx.load(str(folder/'model.onnx'),load_external_data=False)
        values={}
        for tensor in graph.graph.initializer:
            dtype=onnx.TensorProto.DataType.Name(tensor.data_type)
            values[dtype]=values.get(dtype,0)+math.prod(tensor.dims)
        result['exports'][name]={'initializer_values_by_dtype':values,
                                  'total_graph_and_external_data_bytes':sum(f['bytes'] for f in files.values()),
                                  'files':files,'graph_nodes':len(graph.graph.node)}
    for scope,item in data['timing'].items():
        ratios=[row['arms']['original_onnxruntime']['milliseconds']/row['arms']['compressed_onnxruntime']['milliseconds']
                for row in item['rounds']]
        result['timing'][scope]={'arms':item['summary'],
                                 'original_ort_over_compressed_ort':{'median_paired_ratio':statistics.median(ratios),
                                                                   'min_paired_ratio':min(ratios),'max_paired_ratio':max(ratios),
                                                                   'paired_ratios':ratios}}
    a,b=result['exports']['original'],result['exports']['compressed']
    result['export_comparison']={
        'float_initializer_values_removed':a['initializer_values_by_dtype']['FLOAT']-b['initializer_values_by_dtype']['FLOAT'],
        'float_initializer_reduction_percent':100*(1-b['initializer_values_by_dtype']['FLOAT']/a['initializer_values_by_dtype']['FLOAT']),
        'onnx_file_reduction_percent':100*(1-b['total_graph_and_external_data_bytes']/a['total_graph_and_external_data_bytes'])}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['exports','timing']},indent=2))


if __name__=='__main__':main()
