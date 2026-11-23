"""Native50Hz W100 replay matching the published fixed-checkpoint evaluation."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
from pathlib import Path
import numpy as np
import torch
if __package__:
    from .infer import RingRQInference
    from .streaming.stream import ChunkStream
    from .streaming.batch256 import make_batch_class
else:
    from infer import RingRQInference
    from streaming.stream import ChunkStream
    from streaming.batch256 import make_batch_class

ASSETS = Path(__file__).resolve().parent / 'assets/phase2_ep1'


class RingRQW100Inference:
    """One audited recording segment per predict call; reset only between calls.

    CUDA FP32 only. Full256 encoder graph, real eager tail, decoder KV100,
    continuous recurrent encoder, absolute RoPE and the original Ring-RQ head.
    """
    def __init__(self, checkpoint=ASSETS/'best.pt', norm=ASSETS/'norm.npz',
                 rq=ASSETS/'RQ_FROZEN.pt', device='cuda:0'):
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('Published W100 backend requires CUDA; use infer.py for reset2s CPU inference')
        torch.cuda.set_device(self.device)
        self._runtime = RingRQInference(checkpoint, norm, rq, str(self.device))
        self.model = self._runtime.model.requires_grad_(False).eval()
        self.norm = self._runtime.norm
        self.stream = None

    @torch.inference_mode()
    def predict(self, emg):
        """Finite float32[T,16] filtered unnormalized uV -> all valid native frames.

        Each call resets at one recording/technical-gap boundary. Do not split
        one continuous segment into two-second calls. Incomplete terminal input
        is handled exactly as in the fixed-checkpoint evaluator, without padding.
        """
        raw = np.asarray(emg)
        if raw.dtype != np.float32 or raw.ndim != 2 or raw.shape[1] != 16 or len(raw) < 470 or not np.isfinite(raw).all():
            raise ValueError('Expected finite float32[T,16], T>=470, filtered raw-scale uV')
        tokens = (len(raw) - 30)//40 + 1
        if tokens > 200000:
            raise ValueError('Segment exceeds original replay capacity of 200000 tokens')
        with torch.cuda.device(self.device):
            if self.stream is None:
                self.stream = make_batch_class(ChunkStream)(self.model, capacity=200000, window=100)
            self.stream.reset()
            parts = []
            for start in range(0, tokens, 256):
                stop = min(tokens, start + 256)
                left = 0 if start == 0 else 30 + (start - 1)*40
                block = raw[left:30 + (stop - 1)*40]
                x = np.ascontiguousarray(((block-self.norm['emg_mean'])/self.norm['emg_std']).T)
                parts.append(self.stream.push(torch.from_numpy(x[None]).to(self.device)).clone())
            normalized = torch.cat(parts)
            assert normalized.shape == (tokens-11, 5) and torch.isfinite(normalized).all()
            scale = torch.as_tensor(self.norm['force_std'], device=self.device)
            mean = torch.as_tensor(self.norm['force_mean'], device=self.device)
            force = normalized*scale + mean
            return dict(native_normalized=normalized.cpu().numpy(), force_N=force.cpu().numpy(),
                        input_sample_indices=470 + 40*np.arange(tokens-11, dtype=np.int64))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default=str(ASSETS/'best.pt'))
    p.add_argument('--norm', default=str(ASSETS/'norm.npz'))
    p.add_argument('--rq', default=str(ASSETS/'RQ_FROZEN.pt'))
    p.add_argument('--emg', required=True, help='float32.npy [T,16], one continuous filtered segment in uV')
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    target = Path(a.output)
    if target.exists():
        raise FileExistsError(target)
    raw = np.load(a.emg, mmap_mode='r', allow_pickle=False)
    result = RingRQW100Inference(a.checkpoint, a.norm, a.rq, a.device).predict(raw)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as f:
        np.savez(f, **result)


if __name__ == '__main__':
    main()
