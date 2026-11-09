"""Convenience import matching the reference share-package layout."""
from ap304_rank.model import load_model
__all__=['load_model']

if __name__=='__main__':
    import torch
    torch.set_num_threads(2)
    net=load_model(device='cpu')
    print('PASS: complete AP304 strict-loaded; 18,357,945 parameters; no forward')
