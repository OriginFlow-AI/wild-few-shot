"""Offline legacy Ring-RQ inference: independent two-second reset windows."""
import argparse
from pathlib import Path
import numpy as np
import torch
from model import RingRQ
from source.rq_model import assert_frozen
from source.stable_interpolation import OrderedLinearBackward


class RingRQInference:
    def __init__(self, checkpoint, norm, rq, device="cpu"):
        self.device = torch.device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        with np.load(norm, allow_pickle=False) as z:
            self.norm = {k: z[k].copy() for k in ("emg_mean", "emg_std", "force_mean", "force_std")}
        for k, n in (("emg_mean", 16), ("emg_std", 16), ("force_mean", 5), ("force_std", 5)):
            a = self.norm[k]
            if a.shape != (n,) or a.dtype != np.float32 or not np.isfinite(a).all():
                raise ValueError(f"Invalid norm field: {k}")
        if (self.norm["emg_std"] <= 0).any() or (self.norm["force_std"] <= 0).any():
            raise ValueError("Normalization scales must be positive")
        # torch.load checkpoints must come from a trusted source.
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        from hashlib import sha256
        for key, path in (("norm_sha256", norm), ("RQ_sha256", rq)):
            expected = payload.get("binding", {}).get(key)
            if expected and sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Checkpoint asset mismatch: {key}")
        rq_payload = torch.load(rq, map_location="cpu", weights_only=False)
        expected_norm = rq_payload.get("normalization_sha256")
        if expected_norm and sha256(Path(norm).read_bytes()).hexdigest() != expected_norm:
            raise ValueError("RQ was fitted with a different norm")
        self.model = RingRQ(rq, activation_checkpoint=False).to(self.device).eval()
        self.model.load_state_dict(payload["model"], strict=True)
        assert_frozen(self.model.rq_readout, rq)

    @torch.inference_mode()
    def predict(self, emg):
        """Input float32 [B,4000,16] filtered, unnormalized EMG in uV.

        Returns normalized native89 and dense3572 physical N. No force-label
        recalibration, smoothing or nonnegative clamp is applied to predictions.
        Every input window resets the network, matching legacy R evaluation.
        """
        a = np.asarray(emg)
        if a.ndim != 3 or a.shape[1:] != (4000, 16) or a.dtype != np.float32 or not np.isfinite(a).all():
            raise ValueError("Expected finite float32 [B,4000,16] filtered raw-scale EMG")
        x = (a - self.norm["emg_mean"]) / self.norm["emg_std"]
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(0, 2, 1))).to(self.device)
        native = self.model(x)
        dense = OrderedLinearBackward.apply(native)
        scale = torch.as_tensor(self.norm["force_std"], device=self.device).view(1, 5, 1)
        mean = torch.as_tensor(self.norm["force_mean"], device=self.device).view(1, 5, 1)
        return {
            "native_normalized": native.cpu().numpy(),
            "native_N": (native * scale + mean).cpu().numpy(),
            "dense_N": (dense * scale + mean).cpu().numpy(),
            "dense_input_sample_indices": np.arange(428, 4000, dtype=np.int64),
        }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--norm", required=True)
    p.add_argument("--rq", required=True)
    p.add_argument("--emg", required=True, help="float32 .npy [B,4000,16], filtered raw-scale uV")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=1)
    a = p.parse_args()
    if a.batch_size < 1:
        p.error("--batch-size must be positive")
    target = Path(a.output)
    if target.exists():
        raise FileExistsError(target)
    emg = np.load(a.emg, mmap_mode="r", allow_pickle=False)
    if emg.ndim != 3 or emg.shape[1:] != (4000, 16) or len(emg) == 0:
        raise ValueError("Expected nonempty [B,4000,16]")
    runtime = RingRQInference(a.checkpoint, a.norm, a.rq, a.device)
    rows = [runtime.predict(emg[i:i+a.batch_size]) for i in range(0, len(emg), a.batch_size)]
    result = {k: np.concatenate([v[k] for v in rows], axis=0) for k in ("native_normalized", "native_N", "dense_N")}
    result["dense_input_sample_indices"] = rows[0]["dense_input_sample_indices"]
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as f:
        np.savez(f, **result)


if __name__ == "__main__":
    main()
