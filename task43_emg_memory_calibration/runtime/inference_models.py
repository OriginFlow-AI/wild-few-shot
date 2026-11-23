"""Load externally supplied, SHA-bound A / LoRA / F checkpoints."""
from pathlib import Path
import hashlib
import json
import torch
from arms import Arm

ROOT=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def checkpoint(arm,name):
    entry=json.loads((ROOT/'INFERENCE.json').read_bytes())['checkpoints'][name]
    path=Path(entry['path'])
    if not path.is_absolute():path=ROOT/path
    if sha(path)!=entry['sha256']:raise ValueError('Checkpoint SHA mismatch: '+name)
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if entry.get('epoch') is not None and (saved['epoch']!=entry['epoch'] or saved['step']!=entry['epoch']*752):raise ValueError('Selected checkpoint epoch/step mismatch; supply the actual best checkpoint')
    arm.load_state_dict(saved['model'],strict=True)
    return arm


def memory_model(device,trained=True):
    arm=Arm('F',device)
    if trained:checkpoint(arm,'F')
    original=torch.load(ROOT/'assets/A_ep1.pt',map_location='cpu',weights_only=True)
    if not all(torch.equal(v.cpu(),original[k]) for k,v in arm.base.state_dict().items()):
        raise ValueError('F must keep exactly the supplied A base frozen')
    return arm.eval().requires_grad_(False)


def lora_model(device):return checkpoint(Arm('B_L',device),'B_L').eval().requires_grad_(False)


def models(device):return memory_model(device),lora_model(device)
