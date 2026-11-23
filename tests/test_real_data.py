"""Optional regression against the frozen 11/18 audit, supplied outside Git."""

import json
import os
from pathlib import Path
import unittest

import numpy as np

from cmcu_calibration.io import load_native
from cmcu_calibration.quality import prepare_source, sliding_audit
from cmcu_calibration.routing import fixed_bsq_zero

NATIVE = os.environ.get("CMCU_NATIVE_DIR")
SLOPES = os.environ.get("CMCU_SLOPE_AUDIT")
SLIDING = os.environ.get("CMCU_SLIDING_AUDIT")


@unittest.skipUnless(
    NATIVE and SLOPES and SLIDING, "Frozen external audit paths not provided"
)
class FrozenAuditTests(unittest.TestCase):
    def test_all_archived_channel_slopes(self):
        rows = json.loads(Path(SLOPES).read_text())["rows"]
        grouped = {}
        for row in rows:
            grouped.setdefault(row["id"], []).append(row)
        labels = [
            hand + name
            for hand in ("左", "右")
            for name in ("拇指", "食指", "中指", "无名指", "小指")
        ]
        compared = 0
        for recording, channels in grouped.items():
            native = load_native(
                Path(NATIVE) / (recording + ".npz"), needs_reference=True
            )
            _, quality = prepare_source(
                native, routing="default_force_peak", slope_scope="off"
            )
            self.assertEqual(len(quality["channels"]), len(channels), recording)
            for row in channels:
                actual = quality["channels"][str(labels.index(row["finger"]))]["slope"][
                    "slope"
                ]
                np.testing.assert_allclose(
                    actual,
                    row["slope"],
                    atol=1e-5,
                    rtol=1e-5,
                    err_msg=recording + " " + row["finger"],
                )
                compared += 1
        self.assertGreater(compared, 0)

    def test_archived_native_sliding_decisions(self):
        rows = json.loads(Path(SLIDING).read_text())["records"]
        compared = 0
        for row in rows:
            path = Path(NATIVE) / (row["id"] + ".npz")
            if not path.exists():
                continue
            native = load_native(path, needs_reference=True)
            actual = sliding_audit(native, fixed_bsq_zero(native["bsq_force"]))
            expected = row["events"]
            self.assertEqual(len(actual), len(expected))
            for old in expected:
                current = actual[old["p"]]
                self.assertEqual(current["eligible"], old["eligible"])
                self.assertEqual(current["passed"], old["passed"])
                self.assertEqual(current["strict_passed"], old["strict"])
                if old["eligible"]:
                    self.assertEqual(current["frames"], old["frames"])
                compared += 1
        self.assertGreater(compared, 0)
