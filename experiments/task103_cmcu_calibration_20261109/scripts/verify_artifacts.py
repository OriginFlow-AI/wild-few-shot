"""Offline, standard-library checks for the published report/weights/statistics."""
import csv
import gzip
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path

EXPERIMENT = Path(__file__).resolve().parents[1]
REPO = EXPERIMENT.parents[1]
EXPECTED = {
    'ap304_rank_best_model_state_dict.pt': 'f6dfe88f7e25d5be934ee0b065ae76d94aa89edc88340164244ecd149fa89928',
    'train_norm.npz': '7ee5781e253ca45c2f31176e11ec5f065fd529df8afd9dc80bde5e3fec9d2de6',
}


class Links(HTMLParser):
    def __init__(self): super().__init__(); self.links = []; self.images = 0
    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ['src', 'href'] and value:
                if value.startswith('data:image/'): self.images += 1
                elif not value.startswith(('http:', 'https:', '#', 'mailto:')): self.links.append(value)


def main():
    for name, expected in EXPECTED.items():
        path = REPO/'models/ap304_rank'/name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected: raise SystemExit(f'Weight/normalization mismatch: {name}; run git lfs pull')
    parsed = Links(); parsed.feed((EXPERIMENT/'report.html').read_text())
    for name in parsed.links:
        if not (EXPERIMENT/name).is_file(): raise SystemExit('Missing report dependency: '+name)
    assert parsed.images == 8, parsed.images
    sources = json.loads((EXPERIMENT/'manifests/sources.json').read_text())
    assert len(sources) == 690 and len({r['id'] for r in sources}) == 690
    assert sum(r['stage'] == 'mix-test-5' for r in sources) == 465
    assert len(json.loads((EXPERIMENT/'manifests/side_rest_tare.json').read_text())) == 225
    effects = list(csv.DictReader(open(EXPERIMENT/'results/paired_effect.csv')))
    row = next(r for r in effects if r['tare'] == 'stable_side_rest_same_routing' and r['scope'] == 'both'
               and r['method'] == 'isotonic' and r['policy'] == 'domain' and r['date_scope'] == 'all' and r['band'] == '0-25')
    assert abs(float(row['raw_MAE_N']) - 6.5275094377696785) < 1e-9
    assert abs(float(row['calibrated_MAE_N']) - 3.26349894223891) < 1e-9
    for name in ['metrics.csv.gz', 'sensor_metrics.csv.gz', 'events.csv.gz', 'calibration_maps.json.gz']:
        with gzip.open(EXPERIMENT/'results'/name, 'rb') as stream:
            while stream.read(1 << 20): pass  # verify gzip CRC across the complete file
    assert len(list((EXPERIMENT/'examples').glob('*.npz'))) == 6
    print(f'PASS: model hashes, 690 sources, {parsed.images} embedded figures, report links, statistics and gzip CRCs')


if __name__ == '__main__': main()
