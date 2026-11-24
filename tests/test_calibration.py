import json
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from cmcu_calibration.calibration import fit_lookup, predict_lookup, slope_check
from cmcu_calibration.pipeline import run, select_source
from cmcu_calibration.io import load_native
from cmcu_calibration.quality import basic_quality, sliding_audit


def native_fixture():
    hand_t = np.arange(1400, dtype=np.int64) * 10_000_000 + 1_000_000_000
    bsq_t = hand_t[::2]
    seconds = np.arange(1400) / 100
    reference = np.interp(
        seconds,
        [0, 2, 2.5, 4.5, 5, 8, 8.5, 10.5, 11, 14],
        [0, 0, 12, 12, 0, 0, 12, 12, 0, 0],
    )
    force = np.zeros((len(hand_t), 5))
    force[:, 0] = reference
    return dict(
        bsq_t=bsq_t,
        bsq_force=reference[::2],
        left_t=hand_t,
        right_t=hand_t,
        left_ad=force * 5000,
        right_ad=np.zeros_like(force),
        left_force=force,
        right_force=np.zeros_like(force),
        clock_ok=True,
    )


class LookupTests(unittest.TestCase):
    def test_duplicate_ad_medians_and_weighted_pava(self):
        model = fit_lookup([0, 0, 1, 1, 1, 2, 2, 2], [0, 0, 4, 4, 4, 2, 2, 2])
        self.assertEqual(model, {"ad": [0.0, 1.0, 2.0], "force_N": [0.0, 3.0, 3.0]})
        np.testing.assert_allclose(
            predict_lookup([0, 0.5, 1.5, 2], model), [0, 1.5, 3, 3]
        )
        self.assertTrue(np.isnan(predict_lookup([-1, 3, np.nan], model)).all())

    def test_insufficient_and_invalid_models(self):
        self.assertIsNone(fit_lookup([0] * 8, np.arange(8)))
        self.assertIsNone(fit_lookup([100] * 8, np.arange(8)))
        self.assertIsNone(fit_lookup(np.arange(7), np.arange(7)))
        with self.assertRaises(ValueError):
            fit_lookup([1, 2], [1])
        with self.assertRaises(ValueError):
            predict_lookup([1], {"ad": [0, 1], "force_N": [1, 0]})

    def test_zero_is_fixed_despite_conflicting_reference(self):
        ad = np.repeat([0, 100, 200], [4, 2, 2])
        reference = [1, 2, 2, 3, 3, 3, 5, 5]
        model = fit_lookup(ad, reference)
        self.assertEqual(model["force_N"][0], 0.0)
        np.testing.assert_allclose(predict_lookup([-0.0, 50, 200], model), [0, 1.5, 5])
        legacy = fit_lookup(ad, reference, zero_anchor=False)
        self.assertEqual(predict_lookup(0, legacy), 2.0)

    def test_added_zero_knot_preserves_observed_domain_and_quality_span(self):
        ad = np.linspace(100, 300, 20)
        reference = np.linspace(2.0, 2.5, len(ad))
        model = json.loads(json.dumps(fit_lookup(ad, reference)))
        np.testing.assert_allclose(
            predict_lookup([0, 1, 99, 100, 200, 300, 301], model),
            [0, np.nan, np.nan, 2, 2.25, 2.5, np.nan],
        )
        self.assertEqual(basic_quality(ad, reference, model), "collapsed_calibration_force_range")

    def test_monotone_negative_inputs_do_not_extend_the_domain(self):
        ad = [-200, -200, -100, -100, 100, 100, 200, 200]
        model = fit_lookup(ad, [1, 1, 2, 2, 3, 3, 5, 5])
        np.testing.assert_allclose(
            predict_lookup([-201, -200, -50, 0, 50, 200], model),
            [np.nan, 0, 0, 0, 1.5, 5],
        )

    def test_slope_includes_intercept_and_force_band(self):
        x = np.array([0, 1, 2, 8, 15, 25, 30], dtype=float)
        y = 2 * x + 7
        y[[0, 1, 6]] = 9999
        result = slope_check(x, y)
        self.assertEqual(result["points"], 4)
        self.assertAlmostEqual(result["slope"], 2)
        self.assertAlmostEqual(result["intercept_N"], 7)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(slope_check(x, x * 0.1)["status"], "low")
        self.assertEqual(slope_check(x, x * 4)["status"], "high")
        self.assertEqual(slope_check([3, 3], [1, 2])["status"], "insufficient")


class SlidingTests(unittest.TestCase):
    def test_native_frame_threshold_and_any_channel(self):
        data = native_fixture()
        data["left_ad"][:] = 0
        data["right_ad"][:] = 0
        data["right_ad"][250:384, 4] = 50000  # 134 native rows
        first = sliding_audit(data, 0)[1]
        self.assertTrue(first["eligible"])
        self.assertEqual(first["frames"], 134)
        self.assertFalse(first["passed"])
        data["right_ad"][384, 4] = 50000
        first = sliding_audit(data, 0)[1]
        self.assertEqual(first["frames"], 135)
        self.assertEqual(first["channel"], "R4")
        self.assertTrue(first["passed"])

    def test_short_bsq_is_not_a_bad_ad_diagnosis(self):
        data = native_fixture()
        data["bsq_force"][:] = 0
        data["bsq_force"][100:120] = 8
        data["left_ad"][:] = 50000
        row = sliding_audit(data, 0)[1]
        self.assertFalse(row["eligible"])
        self.assertEqual(row["reason"], "BSQ_duration_too_short")


class SourceTests(unittest.TestCase):
    def test_time_day_device_completion_and_no_self(self):
        start = datetime.fromisoformat("2026-11-18T09:00:00+08:00")
        row = dict(
            id="a",
            subject="one",
            date="2026-11-18",
            pc="pc",
            boot="boot",
            start=start.isoformat(),
        )
        source = dict(
            record=row, start=start, end=start + timedelta(minutes=2), models={}
        )
        target = dict(row, id="b", start=(start + timedelta(hours=1)).isoformat())
        self.assertIs(select_source(target, [source]), source)
        self.assertIsNone(
            select_source(
                dict(
                    target,
                    start=(start + timedelta(hours=1, microseconds=1)).isoformat(),
                ),
                [source],
            )
        )
        self.assertIsNone(select_source(row, [source]))
        self.assertIsNone(select_source(dict(target, pc="different"), [source]))
        self.assertIsNone(select_source(dict(target, date="2026-11-19"), [source]))
        self.assertIsNone(
            select_source(
                dict(target, start=(start + timedelta(seconds=60)).isoformat()),
                [source],
            )
        )


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        data = native_fixture()
        np.savez(self.root / "source.npz", **data)
        target = {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in data.items()
            if not key.startswith("bsq_")
        }
        target["left_ad"][300, 0] = 70000
        np.savez(self.root / "target.npz", **target)
        self.rows = [
            dict(
                id="source",
                subject="one",
                stage="mix-test-5",
                start="2026-11-18T09:00:00",
                end="2026-11-18T09:00:14",
                path="source.npz",
            ),
            dict(
                id="target",
                subject="one",
                stage="mix-17",
                start="2026-11-18T09:05:00",
                path="target.npz",
            ),
        ]
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(json.dumps(self.rows))

    def tearDown(self):
        self.temp.cleanup()

    def test_batch_previous_source_and_invalid_domain(self):
        output = self.root / "out"
        summary = run(self.manifest, output)
        self.assertEqual(summary["statuses"], {"no_source": 1, "calibrated": 1})
        with np.load(output / "force/target.npz") as result:
            self.assertEqual(result["source_id"].item(), "source")
            self.assertFalse(result["left_valid"][300, 0])
            self.assertTrue(np.isnan(result["left_force_N"][300, 0]))
            self.assertFalse(result["left_valid"][:, 1:].any())
            data = native_fixture()
            mask = result["left_valid"][:, 0]
            np.testing.assert_allclose(
                result["left_force_N"][mask, 0], data["left_force"][mask, 0], atol=1e-12
            )
        self.assertEqual(
            json.loads((output / "run.json").read_text())["status"], "complete"
        )

    def test_batch_zero_anchor_with_nonzero_source_baseline(self):
        source = native_fixture()
        source["left_ad"][:, 0] += 5000
        np.savez(self.root / "source.npz", **source)
        with np.load(self.root / "target.npz") as payload:
            target = dict(payload)
        target["left_ad"][:3, 0] = [0, 1, 5000]
        np.savez(self.root / "target.npz", **target)

        output = self.root / "out"
        run(self.manifest, output)
        with np.load(output / "force/source.npz") as result:
            self.assertFalse(result["left_valid"].any())
        with np.load(output / "force/target.npz") as result:
            np.testing.assert_array_equal(result["left_valid"][:3, 0], [True, False, True])
            np.testing.assert_allclose(result["left_force_N"][:3, 0], [0, np.nan, 0])
            self.assertFalse(result["left_valid"][:, 1:].any())

    def test_legacy_nine_channel_native_arrays(self):
        data = native_fixture()
        for hand in ("left", "right"):
            for field in ("ad", "force"):
                key = hand + "_" + field
                data[key] = np.column_stack([data[key], np.full((1400, 4), 99999)])
        path = self.root / "nine_channels.npz"
        np.savez(path, **data)
        loaded = load_native(path, needs_reference=True)
        self.assertEqual(loaded["left_ad"].shape, (1400, 5))
        np.testing.assert_equal(loaded["left_ad"], native_fixture()["left_ad"])

    def test_rejected_record_and_retry_remove_stale_success(self):
        output = self.root / "out"
        run(self.manifest, output)
        data = native_fixture()
        data["clock_ok"] = False
        np.savez(self.root / "source.npz", **data)
        summary = run(self.manifest, output)
        self.assertEqual(summary["statuses"], {"error": 1, "no_source": 1})
        self.assertFalse((output / "force/source.npz").exists())
        with np.load(output / "force/target.npz") as result:
            self.assertFalse(result["left_valid"].any())

    def test_slope_rejects_entire_record(self):
        data = native_fixture()
        data["left_force"] *= 0.1
        np.savez(self.root / "source.npz", **data)
        summary = run(self.manifest, self.root / "out")
        self.assertEqual(summary["statuses"], {"quality_rejected": 1, "no_source": 1})
        self.assertEqual(summary["fitted_sources"], 0)

    def test_unreadable_mcap_is_recorded_and_batch_continues(self):
        (self.root / "broken.mcap").write_bytes(b"broken MCAP")
        self.rows[0]["path"] = "broken.mcap"
        self.manifest.write_text(json.dumps(self.rows))
        output = self.root / "broken"
        summary = run(self.manifest, output)
        self.assertEqual(summary["statuses"], {"error": 1, "no_source": 1})
        receipt = json.loads((output / "quality/source.json").read_text())
        self.assertIn("reason", receipt)
        self.assertTrue((output / "force/target.npz").exists())

    def test_real_cli_and_filter_switches(self):
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "cmcu_calibration",
                "--manifest",
                str(self.manifest),
                "--out",
                str(self.root / "cli"),
                "--routing",
                "default_force_peak",
                "--slope-scope",
                "off",
                "--sliding-scope",
                "press",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        result = json.loads(process.stdout)
        self.assertEqual(result["statuses"], {"no_source": 1, "calibrated": 1})
        self.assertEqual(result["settings"]["sliding_scope"], "press")

    def test_press_filter_masks_failed_target_press(self):
        data = native_fixture()
        data["left_ad"][200:500] = 0
        np.savez(self.root / "target.npz", **data)
        self.rows[1]["stage"] = "mix-test-5"
        self.manifest.write_text(json.dumps(self.rows))
        output = self.root / "press_mask"
        summary = run(
            self.manifest,
            output,
            slope_scope="off",
            sliding_scope="press",
            routing="default_force_peak",
        )
        self.assertEqual(summary["statuses"], {"no_source": 1, "calibrated": 1})
        with np.load(output / "force/target.npz") as result:
            self.assertFalse(result["left_valid"][350, 0])
            self.assertTrue(result["left_valid"][950, 0])


if __name__ == "__main__":
    unittest.main()
