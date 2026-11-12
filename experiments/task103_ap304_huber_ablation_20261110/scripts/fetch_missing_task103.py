"""Fill a separate cache for recordings excluded from the calibration experiment."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from common import atomic, load
from extract import one


def main(source, root, workers):
    plan = load(source / 'plan.json')
    rows = [row for row in plan['rows']
            if not (source / 'cache/native' / f"{row['id']}.npz").is_file()
            and not (root / 'cache/emg' / row['id'] / 'done.json').is_file()]
    rows.sort(key=lambda row: (row['stage'] not in ('mix-1', 'mix-11'), row['stage'], row['id']))
    errors = {}
    success = Counter()
    audit = root / 'checks/missing_native_extraction.json'
    print(f'Extracting {len(rows)} missing recordings', flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(one, root, row, False): row for row in rows}
        for i, future in enumerate(as_completed(pending), 1):
            row = pending[future]
            try:
                future.result()
                success[row['stage']] += 1
            except Exception as error:
                errors[row['id']] = dict(stage=row['stage'], reason=str(error))
            if i % 50 == 0:
                atomic(audit, dict(phase='extracting', attempted=i, total=len(rows),
                                   success_by_stage=success, errors=errors))
                print(i, len(rows), len(errors), flush=True)
    atomic(audit, dict(phase='complete', attempted=len(rows), total=len(rows),
                       success_by_stage=success, errors=errors))
    print(f'Extraction complete, {sum(success.values())} succeeded, {len(errors)} failed', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=24)
    args = parser.parse_args()
    main(args.source, args.root, args.workers)
