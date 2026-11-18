"""Launch the selected historical trainer into a fresh output directory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import yaml

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('experiment', choices=sorted(p.name for p in (root/'experiments').iterdir()))
parser.add_argument('--output', type=Path)
parser.add_argument('--check', action='store_true')
for key in ('index', 'cache-root', 'source-checkpoint', 'val-filtered', 'test-filtered'):
    parser.add_argument('--'+key, type=Path)
args = parser.parse_args()
folder = root/'experiments'/args.experiment
config = yaml.safe_load((folder/'resolved_config.yaml').read_text())
snapshot = folder/'source_snapshot'
module = 'emg2force.' + next((snapshot/'emg2force').glob('ap304_stage2*train.py')).stem
config.update(model_source=str(snapshot/'model_source.py'), emg_norm=str(folder/'emg_norm.npz'),
              calibration_package=str(folder/'calibration'), ranking_prepared=str(folder/'prepared'))
for key in ('index', 'cache_root', 'source_checkpoint'):
    path = getattr(args, key)
    if path is not None:
        if config[key] is None:
            parser.error('This experiment uses random initialization.')
        config[key] = str(path.resolve())
for split, spec in config.get('filtered_evaluation', {}).items():
    override = getattr(args, split+'_filtered')
    if override:
        spec['path'] = str(override.resolve())

def verify(path, expected):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError('SHA-256 mismatch: '+str(path))

for key in ('index', 'model_source', 'emg_norm', 'source_checkpoint'):
    if config[key]:
        verify(config[key], config[key+'_sha256'])
for spec in config.get('filtered_evaluation', {}).values():
    verify(spec['path'], spec['sha256'])
if not Path(config['cache_root']).is_dir():
    parser.error('Raw cache directory unavailable. Supply --cache-root.')
env = dict(os.environ, PYTHONPATH=str(snapshot), PYTHONDONTWRITEBYTECODE='1')
subprocess.run([sys.executable, '-m', module, '--help'], env=env, cwd=snapshot, check=True,
               stdout=subprocess.DEVNULL)
if args.check:
    print(args.experiment, 'inputs and trainer import OK')
    sys.exit()
if args.output is None:
    parser.error('Supply a fresh --output directory or use --check.')
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=False)
config['run_root'] = str(out)
for name in ('PREPARED.json', 'train_norm.npz'):
    shutil.copy2(folder/name, out/name)
shutil.copytree(folder/'prepared', out/'prepared')
config_file = out/'resolved_config.yaml'
config_file.write_text(yaml.safe_dump(config, sort_keys=False))
subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone',
                '--nproc-per-node='+str(config['devices']), '--module', module,
                '--config', str(config_file), '--mode', 'train'], env=env, cwd=snapshot, check=True)
