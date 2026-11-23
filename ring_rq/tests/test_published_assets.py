"""Published checkpoint loading and W100 wrapper checks without private signals."""
import contextlib
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ring_rq.infer import RingRQInference
from ring_rq.infer_w100 import RingRQW100Inference, ASSETS


class PublishedAssetsTests(unittest.TestCase):
    def test_phase1_endpoint_lineage_shared_assets_and_cpu_inference(self):
        torch.set_num_threads(2)
        root = ASSETS.parent/'phase1_ep6'
        receipt = json.loads((root/'MANIFEST.json').read_bytes())
        first = torch.load(root/'best.pt', map_location='cpu', weights_only=False)
        second = torch.load(ASSETS/'best.pt', map_location='cpu', weights_only=False)
        self.assertEqual((first['phase'], first['epoch'], first['global_step']), (1, 6, 19578))
        self.assertEqual(first['source_checkpoint_sha256'], second['binding']['parent_sha256'])
        self.assertEqual(hashlib.sha256((root/'best.pt').read_bytes()).hexdigest(), receipt['files']['best.pt']['sha256'])
        for name, binding in [('norm.npz', 'norm_sha256'), ('RQ_FROZEN.pt', 'RQ_sha256')]:
            actual = hashlib.sha256((ASSETS/name).read_bytes()).hexdigest()
            self.assertEqual(first['binding'][binding], actual)
            self.assertEqual(second['binding'][binding], actual)
            self.assertEqual(receipt['shared_assets'][name]['sha256'], actual)
        runtime = RingRQInference(root/'best.pt', ASSETS/'norm.npz', ASSETS/'RQ_FROZEN.pt', 'cpu')
        result = runtime.predict(np.zeros((1, 4000, 16), np.float32))
        self.assertEqual(result['dense_N'].shape, (1, 5, 3572))
        self.assertTrue(np.isfinite(result['dense_N']).all())

    def test_published_asset_hashes_and_real_checkpoint_cpu_inference(self):
        torch.set_num_threads(2)
        receipt = json.loads((ASSETS/'MANIFEST.json').read_bytes())
        for n, expected in receipt['files'].items():
            p = ASSETS/n
            self.assertEqual(p.stat().st_size, expected['bytes'])
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), expected['sha256'])
        runtime = RingRQInference(ASSETS/'best.pt', ASSETS/'norm.npz', ASSETS/'RQ_FROZEN.pt', 'cpu')
        emg = np.zeros((1, 4000, 16), np.float32)
        result = runtime.predict(emg)
        self.assertEqual(result['dense_N'].shape, (1, 5, 3572))
        self.assertTrue(np.isfinite(result['dense_N']).all())
        normalized = torch.from_numpy(((emg-runtime.norm['emg_mean'])/runtime.norm['emg_std']).transpose(0, 2, 1).copy())
        with torch.inference_mode():
            expected = runtime.model(normalized).numpy()
        self.assertTrue(np.array_equal(result['native_normalized'], expected))

    def test_w100_wrapper_full256_real_tail_and_per_call_reset(self):
        class FakeStream:
            def __init__(self):
                self.calls = []
                self.resets = 0
            def reset(self):
                self.i = 0
                self.resets += 1
            def push(self, value):
                self.calls.append(value.shape[-1])
                size = (value.shape[-1]+10)//40 if self.i == 0 else value.shape[-1]//40
                trim = min(size, max(0, 11-self.i))
                self.i += size
                return torch.full((size-trim, 5), -1., dtype=torch.float32)
        runtime = RingRQW100Inference.__new__(RingRQW100Inference)
        runtime.device = torch.device('cpu')
        runtime.norm = dict(emg_mean=np.zeros(16, np.float32), emg_std=np.ones(16, np.float32),
                            force_mean=np.zeros(5, np.float32), force_std=np.ones(5, np.float32))
        runtime.stream = FakeStream()
        with patch('torch.cuda.device', lambda _: contextlib.nullcontext()):
            a = runtime.predict(np.zeros((12000, 16), np.float32))
            b = runtime.predict(np.zeros((470, 16), np.float32))
        self.assertEqual(runtime.stream.calls, [10230, 1760, 470])
        self.assertEqual(runtime.stream.resets, 2)
        self.assertEqual(a['force_N'].shape, (289, 5))
        self.assertEqual(b['force_N'].shape, (1, 5))
        self.assertTrue(np.all(a['force_N'] == -1))  # No hidden terminal clipping.
        self.assertEqual(a['input_sample_indices'][-1], 11990)


if __name__ == '__main__':
    unittest.main()
