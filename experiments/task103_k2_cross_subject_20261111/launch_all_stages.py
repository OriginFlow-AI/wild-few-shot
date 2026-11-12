"""Launch and verify the native-force Task103 person-only baseline."""
import argparse
import fcntl
from pathlib import Path
import sys

from launch_parallel import atomic, invoke, train_all


def main(root, source, prior):
    (root / 'logs').mkdir(parents=True, exist_ok=True)
    with (root / 'launcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic(root / 'checks/launcher_process.json', dict(pid=os.getpid(), argv=sys.argv))
        try:
            if not (root / 'prepared/dataset.json').is_file():
                atomic(root / 'STATUS.json', dict(phase='preparing_all_stage_person_split'))
                invoke('scripts/prepare_all_stages.py', '--source', source, '--prior-root', prior, '--root', root)
            train_all(root, adopt_running=True, branches=('raw_original',))
            invoke('analysis/report_all_stages.py', '--root', root)
        except Exception as error:
            atomic(root / 'STATUS.json', dict(phase='failed', error=str(error)))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--prior-root', type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.source, args.prior_root)
