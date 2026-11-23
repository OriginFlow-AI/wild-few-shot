"""Synthetic CPU checks; no private data, trained checkpoints or Test access."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from infer import RingRQInference
from model import RingRQ
from source.rq_model import ForceTokenizer, assert_frozen
from source.stable_interpolation import cpu_oracle
from source.ordinal import selftest
from io_local import Index, LocalReader, zip_arrays
from label_policy import transform
from losses import total
from init_run import initialize, REQUIRED


class RingRQTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        torch.manual_seed(42)
        tok = ForceTokenizer()
        tok.initialize(torch.linspace(-3, 3, 256).view(-1, 1).repeat(1, 5))
        cls.rq = cls.root / "rq.pt"
        torch.save({"model": tok.state_dict()}, cls.rq)
        cls.norm = cls.root / "norm.npz"
        np.savez(cls.norm, emg_mean=np.zeros(16, np.float32), emg_std=np.ones(16, np.float32),
                 force_mean=np.zeros(5, np.float32), force_std=np.ones(5, np.float32))
        cls.model = RingRQ(cls.rq, activation_checkpoint=False).eval()
        cls.checkpoint = cls.root / "model.pt"
        torch.save({"model": cls.model.state_dict()}, cls.checkpoint)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_native_dense_and_inference_roundtrip(self):
        emg = np.random.default_rng(42).normal(size=(1, 4000, 16)).astype(np.float32)
        runtime = RingRQInference(self.checkpoint, self.norm, self.rq)
        result = runtime.predict(emg)
        with torch.inference_mode():
            expected = self.model(torch.from_numpy(emg.transpose(0, 2, 1).copy()))
        self.assertEqual(result["native_N"].shape, (1, 5, 89))
        self.assertEqual(result["dense_N"].shape, (1, 5, 3572))
        self.assertTrue(np.array_equal(result["native_normalized"], expected.numpy()))
        self.assertTrue(np.isfinite(result["dense_N"]).all())
        self.assertEqual(sum(p.numel() for p in runtime.model.parameters() if p.requires_grad), 18743770)
        assert_frozen(runtime.model.rq_readout, self.rq)
        with self.assertRaises(ValueError):
            runtime.predict(emg.astype(np.float64))

    def test_loss_and_interpolation_gradients(self):
        self.assertTrue(cpu_oracle()["backward_native_CPU_exact"])
        self.assertEqual(selftest()["status"], "PASS_FINITE_ORDER_TIES_SHIFT_SCALE_AND_DETERMINISTIC_PAIRS")
        norm = {"force_mean": np.zeros(5, np.float32), "force_std": np.ones(5, np.float32)}
        pred = torch.linspace(-1, 1, 100).reshape(2, 5, 10).requires_grad_()
        target = pred.detach() + .25
        for phase in (1, 2):
            loss, _, _, _, _ = total(pred, target, norm, [1]*5, 42, phase, .1)
            self.assertTrue(torch.isfinite(torch.autograd.grad(loss, pred)[0]).all())

    def test_gap_contained_sampling_and_test_closure(self):
        rows = [dict(stage="legacy/8", subject="synthetic", hands=[{"side": "left"}])]
        index = Index(rows, None, prepared=[[(0, 4100), (6000, 10500)]])
        for k in range(100):
            i, start = index.sample(1, k)
            self.assertEqual(i, 0)
            self.assertTrue(0 <= start <= 100 or 6000 <= start <= 6500)
            self.assertEqual(index.sample(1, k), (i, start))
        self.assertEqual(list(index.fixed_windows()), [(0, 0), (0, 6000)])
        with self.assertRaises(PermissionError):
            LocalReader().load({"split": "test"})

    def test_gen3_identity_and_unknown_domain_rejection(self):
        raw = np.arange(-10, 10, dtype=np.float32).reshape(4, 5)
        out = transform(raw, layout="1x5x3", domain="raw_extracted_N")
        self.assertTrue(np.array_equal(raw, out))
        self.assertFalse(np.shares_memory(raw, out))
        with self.assertRaises(AssertionError):
            transform(raw, layout="unknown", domain="raw_extracted_N")
        with self.assertRaises(AssertionError):
            transform(raw, layout="1x5x1", domain="unknown")

    def test_task83_stored_npz_mapping_and_filter_marker(self):
        import zipfile
        p = self.root / "synthetic.npz"
        emg, force = np.ones((4100, 16), np.float32), np.ones((110, 5), np.float32)
        np.savez(p, emg=emg, force=force)
        with self.assertRaises(AssertionError):
            zip_arrays(p)
        with zipfile.ZipFile(p, "a") as z:
            z.comment = b"old_sos_2khz_v1"
        e, f = zip_arrays(p)
        self.assertTrue(np.array_equal(e, emg) and np.array_equal(f, force))
        self.assertIsInstance(e, np.memmap)

    def test_independent_run_creation_and_asset_allowlist(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            assets, run = base / "private", base / "run"
            for n in REQUIRED:
                p = assets / n
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text('{}')
            (assets / 'data/CLOCK_BINDINGS.json').write_text('{"rows":{}}')
            (assets / '.env').write_text('not-an-asset')
            result = initialize(assets, run)
            self.assertEqual(result['status'], 'CREATED_NOT_TRAINING')
            self.assertTrue((run / 'train.py').is_file())
            self.assertTrue((run / 'source/ring_model.py').is_file())
            self.assertTrue((run / 'DATA_SCHEMA.md').is_file())
            self.assertFalse((run / '.env').exists())
            with self.assertRaises(FileExistsError):
                initialize(assets, run)


if __name__ == "__main__":
    unittest.main()
