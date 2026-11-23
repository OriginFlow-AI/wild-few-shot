"""Memory-bounded repeated gradient calibration, same exact model operations."""
import types
from torch.utils.checkpoint import checkpoint

def install(model):
    def encode(self,emg):
        embedded=checkpoint(self.patch_embedding,emg,use_reentrant=False,preserve_rng_state=True)
        tokens=embedded+self.electrode_fourier[None,None]
        for block in self.encoder_blocks:
            block.split_checkpoint_enabled=True
            tokens=checkpoint(block,tokens,use_reentrant=False,preserve_rng_state=True)
        return self.encoder_norm(tokens)
    model.encode=types.MethodType(encode,model)
    return model
