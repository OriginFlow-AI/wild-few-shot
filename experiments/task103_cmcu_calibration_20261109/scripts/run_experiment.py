"""Portable entry point for the frozen Task103 two-layer calibration comparison."""
import argparse
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True, help='Writable output/cache directory, excluded from Git')
    parser.add_argument('--stage', choices=['all', 'extract', 'prepare', 'infer', 'score', 'report'], default='all')
    parser.add_argument('--endpoint-url', default=os.environ.get('TASK103_S3_ENDPOINT'))
    parser.add_argument('--bucket', default=os.environ.get('TASK103_S3_BUCKET', 'tes-bucket'))
    parser.add_argument('--profile', default=None)
    parser.add_argument('--mcap-dir', default=None)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--score-workers', type=int, default=4)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--precision', choices=['bf16', 'fp32'], default='bf16')
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    if args.workers < 1 or args.score_workers < 1: parser.error('Worker counts must be positive')
    if args.device == 'cpu' and args.precision != 'fp32': parser.error('Use --precision fp32 with CPU')
    root = args.run_root.expanduser().resolve(); root.mkdir(parents=True, exist_ok=True)
    scripts = Path(__file__).resolve().parent
    source = scripts.parent/'manifests/sources.json'
    if not (root/'sources.json').exists(): shutil.copyfile(source, root/'sources.json')
    env = dict(os.environ, TASK103_RANK_ROOT=str(root))
    def command(name, *parameters): return [sys.executable, str(scripts/name), *map(str, parameters)]
    def run(name, *parameters): subprocess.run(command(name, *parameters), env=env, check=True)
    def selected(stage): return args.stage in ['all', stage]
    if selected('extract'):
        parameters = ['--workers', args.workers, '--bucket', args.bucket]
        for flag, value in [('--endpoint-url', args.endpoint_url), ('--profile', args.profile), ('--mcap-dir', args.mcap_dir)]:
            if value: parameters.extend([flag, value])
        run('extract.py', *parameters)
    if selected('prepare'): run('prepare.py')
    if selected('infer'): run('infer.py', '--device', args.device, '--precision', args.precision, '--batch-size', args.batch_size)
    if selected('score'):
        import pandas as pd
        jobs = [subprocess.Popen(command('score.py', '--shard', i, '--shards', args.score_workers), env=env)
                for i in range(args.score_workers)]
        failures = [job.wait() for job in jobs]
        if any(failures): raise SystemExit('A scoring shard failed; inspect its output')
        for name in ['metrics.csv', 'sensor_metrics.csv', 'score_exclusions.csv', 'train_audit.csv']:
            frames = []
            for i in range(args.score_workers):
                try: frames.append(pd.read_csv(root/f'score_shard_{i}'/name))
                except pd.errors.EmptyDataError: pass
            pd.concat(frames, ignore_index=True).to_csv(root/name, index=False)
        with gzip.open(root/'calibration_maps.json.gz', 'wt') as output:
            output.write('['); first = True
            for i in range(args.score_workers):
                with gzip.open(root/f'score_shard_{i}'/'calibration_maps.json.gz', 'rt') as input_file:
                    for mapping in json.load(input_file):
                        if not first: output.write(',')
                        json.dump(mapping, output); first = False
            output.write(']')
    if args.stage in ['all', 'score', 'report']:
        run('summarize.py'); run('report.py')
        # Rebuild the unbounded-force supplement from saved primary scoring points.
        # This step also regenerates all summaries and the final HTML.
        run('extend_range.py')
        run('standalone.py', root/'index.html', root/'report.html')
        print('Detailed report:', root/'report.html', flush=True)


if __name__ == '__main__':
    main()
