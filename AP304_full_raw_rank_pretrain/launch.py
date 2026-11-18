"""Start a fresh AP304 ranking pretraining run using the archived specification."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

import yaml

package = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--index', type=Path, required=True)
parser.add_argument('--cache', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--prepare-workers', type=int, default=8)
args = parser.parse_args()

config = yaml.safe_load((package / 'resolved_spec.yaml').read_text())
output = args.output.resolve()
config['workspace'] = str(output)
config['dataset']['index'] = str(args.index.resolve(strict=True))
config['dataset']['cache_root'] = str(args.cache.resolve(strict=True))
config['architecture']['source'] = str(package / 'source_snapshot/model_source.py')
config['artifacts']['root'] = 'run'
output.mkdir(parents=True, exist_ok=False)
settings = output / 'config.yaml'
settings.write_text(yaml.safe_dump(config, sort_keys=False))
env = dict(os.environ, PYTHONPATH=str(package / 'source_snapshot'))
module = 'emg2force.ap304_rank_train'
subprocess.run([sys.executable, '-m', module, '--config', str(settings),
                '--mode', 'prepare', '--prepare-workers', str(args.prepare_workers)], env=env, check=True)
subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone',
                '--nproc-per-node=8', '--module', module, '--config', str(settings),
                '--mode', 'train'], env=env, check=True)
