"""Opt-in synthetic GPU integration check; requires a prepared private workdir."""
from pathlib import Path
import argparse
import json
import os
import sys
import numpy as np


def run(workdir):
    root=Path(workdir).resolve();prepared=json.loads((root/'PREPARED.json').read_bytes())
    if not prepared['synthetic_fixture']:raise ValueError('This integration check only accepts explicit synthetic fixtures')
    cfg=json.loads((root/'CONFIG_F.json').read_bytes());os.environ['CUDA_VISIBLE_DEVICES']=cfg['gpu_uuid'];os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    sys.path.insert(0,str(root));import torch
    import train_r6 as tr
    from infer import Predictor,graph_predictor
    from train_data import Dataset
    from support_cache_raw_r6 import unpack_raw,path_for
    tr.setup(cfg);data=Dataset(cfg['dev'],'dev',True)
    raw=np.stack([data.sample(i)[0] for i in range(3)])
    support_id=next(iter(data.supports));record=data.records[data.supports[support_id]['record']]
    native=Predictor('cuda:0',batch=3)
    cold=native.predict(raw);assert np.array_equal(cold['A'],cold['F'])
    native.bind(record);expected=native.predict(raw)
    cached=unpack_raw(path_for('dev',support_id,record['hand']).read_bytes())
    assert all(torch.equal(a[0].cpu(),b) for a,b in zip(native.memory,cached))
    graph=graph_predictor('cuda:0',64);graph.bind(record['key'],native.memory)
    actual=graph.predict(raw);errors={}
    for name in ['A','B_L','F']:
        np.testing.assert_allclose(actual[name],expected[name],atol=1e-5,rtol=1e-5)
        errors[name]=float(abs(actual[name]-expected[name]).max())
    graph.bind(None);cold=graph.predict(raw);assert np.array_equal(cold['A'],cold['F'])
    graph.bind(record['key']);again=graph.predict(raw)
    assert all(np.array_equal(actual[name],again[name]) for name in actual)
    output=root/'integration_inputs';output.mkdir(exist_ok=True)
    np.save(output/'raw.npy',raw)
    with (output/'memory.npz').open('wb') as f:np.savez(f,**{f'v{i}':v.cpu().numpy() for i,v in enumerate(native.memory)})
    with (output/'portable_native.npz').open('wb') as f:np.savez(f,**expected)
    result=dict(status='PASS',input='synthetic only',model_assets='externally supplied SHA-bound historical A/B_L/F',windows=3,native_frames=89,graph_batch=64,capture_count=graph.capture_count,torch_graph_vs_native_max_abs_N=errors,cold_F_equals_A_bitwise=True,own_cold_own_reset_bitwise=True,inference_memory_equals_training_static_bank_bitwise=True,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),torch_version=torch.__version__,numpy_version=np.__version__)
    (root/'MODEL_INTEGRATION_PASS.json').write_text(json.dumps(result,separators=(',',':')))
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workdir',required=True);run(p.parse_args().workdir)
