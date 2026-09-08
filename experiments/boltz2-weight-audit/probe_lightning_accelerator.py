"""Resolve Lightning's generic GPU accelerator without running a model."""
import argparse,hashlib,inspect,json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    if args.output.exists():raise ValueError('Refusing to overwrite existing output')
    import pytorch_lightning
    import torch
    from pytorch_lightning.trainer.connectors.accelerator_connector import _AcceleratorConnector
    if pytorch_lightning.__version__!='2.5.0':raise ValueError('This source probe is pinned to Lightning2.5.0')
    fn=_AcceleratorConnector._choose_gpu_accelerator_backend
    source=Path(inspect.getsourcefile(fn))
    result={'lightning':pytorch_lightning.__version__,'torch':torch.__version__,'mps_available':torch.backends.mps.is_available(),'generic_gpu_resolves_to':fn(),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'source_url':'https://github.com/Lightning-AI/pytorch-lightning/blob/2.5.0/src/pytorch_lightning/trainer/connectors/accelerator_connector.py','scope':'Accelerator resolution only, not a complete Boltz CLI invocation.'}
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
