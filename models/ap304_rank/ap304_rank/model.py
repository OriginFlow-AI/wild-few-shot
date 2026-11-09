from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch
from torch import nn

BUNDLE_ROOT=Path(__file__).resolve().parents[1]
WEIGHTS={'best':'ap304_rank_best_model_state_dict.pt','last':'ap304_rank_last_model_state_dict.pt'}


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for data in iter(lambda:handle.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


def verify_file(root,relative):
    manifest=json.loads((root/'MANIFEST.json').read_text(encoding='utf-8'))
    record=manifest['files'][relative]
    path=root/relative
    if path.stat().st_size!=record['bytes'] or sha256(path)!=record['sha256']:
        raise ValueError('Bundle integrity mismatch: '+relative)


class AP304RankingModel(nn.Module):
    """Normalized EMG [B,16,4000] -> dimensionless scores [B,5,89]."""
    def __init__(self,network):
        super().__init__();self.network=network
        self.output_units='dimensionless_ranking_score'

    def forward(self,normalized_emg):
        native=self.network(normalized_emg)
        return native.float()/float(self.network.config.output_scale)


def load_model(*,variant='best',device='cpu',root=None,verify=True):
    """Load complete encoder/decoder/affine head; parameters remain trainable.

    Default mode is eval. Use Normalization before this low-level forward, or
    RankPredictor for an API accepting raw time-major EMG. Exported state_dict
    keys belong to model.network; model.forward includes the /0.01 score scale.
    """
    root=Path(root).resolve() if root is not None else BUNDLE_ROOT
    if variant not in WEIGHTS:raise ValueError('variant must be best or last')
    source='model_reference/model_source.py'
    if verify:
        for relative in (WEIGHTS[variant],source,'MODEL_CONFIG.json','train_norm.npz'):
            verify_file(root,relative)
    config=json.loads((root/'MODEL_CONFIG.json').read_text(encoding='utf-8'))
    module_name='_portable_ap304_rank_'+hashlib.sha256(str(root).encode()).hexdigest()[:16]
    spec=importlib.util.spec_from_file_location(module_name,root/source)
    module=importlib.util.module_from_spec(spec);sys.modules[module_name]=module
    spec.loader.exec_module(module)
    network=module.K2RingNetwork(module.K2RingConfig())
    state=torch.load(root/WEIGHTS[variant],map_location='cpu',weights_only=True)
    if len(state)!=config['state_tensor_count'] or not all(isinstance(v,torch.Tensor) and torch.isfinite(v).all() for v in state.values()):
        raise ValueError('Invalid exported model state_dict')
    network.load_state_dict(state,strict=True)
    if sum(p.numel() for p in network.parameters())!=config['parameter_count']:
        raise ValueError('Model parameter count mismatch')
    return AP304RankingModel(network).to(device).eval()
