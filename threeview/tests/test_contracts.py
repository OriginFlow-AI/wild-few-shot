"""Synthetic, data-free checks of the real architecture, reader and inference."""
from pathlib import Path
import sys,tempfile,unittest,hashlib,zipfile
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from threeview.model import ThreeViewModel
from threeview.infer import ThreeViewInference
from threeview.io_local import Index,LocalReader,zip_arrays
from threeview.label_policy import transform
from threeview.init_run import initialize,REQUIRED
from threeview.losses import a_loss,total
class Contracts(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.set_num_threads(2);cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)
  cls.norm=cls.root/'norm.npz';np.savez(cls.norm,emg_mean=np.zeros(16,np.float32),emg_std=np.ones(16,np.float32),force_mean=np.zeros(5,np.float32),force_std=np.ones(5,np.float32))
  cls.model=ThreeViewModel(synthetic=True,backend='reference',streams_enabled=False).eval()
  cls.checkpoint=cls.root/'model.pt';torch.save(dict(model=cls.model.state_dict(),binding=dict(arm='candidate',norm_sha256=hashlib.sha256(cls.norm.read_bytes()).hexdigest())),cls.checkpoint)
 @classmethod
 def tearDownClass(cls):cls.temp.cleanup()
 def test_full_architecture_and_native_ensemble(self):
  x=np.random.default_rng(42).normal(size=(1,4000,16)).astype(np.float32)
  runtime=ThreeViewInference(self.checkpoint,self.norm,'cpu');out=runtime.predict(x)
  with torch.inference_mode():expected=self.model.members_prediction(torch.from_numpy(x.transpose(0,2,1).copy()))
  self.assertTrue(np.array_equal(out['members_normalized'],expected.permute(0,1,3,2).numpy()))
  self.assertTrue(np.array_equal(out['ensemble_normalized'],expected.mean(0).transpose(1,2).numpy()))
  self.assertEqual(out['members_N'].shape,(3,1,55,5));self.assertEqual(out['ensemble_N'].shape,(1,55,5))
  self.assertTrue(np.array_equal(out['input_sample_indices'],1829+40*np.arange(55)))
  self.assertTrue(np.array_equal(runtime.predict(x)['ensemble_N'],out['ensemble_N']))
  self.assertEqual(sum(p.numel() for p in runtime.model.parameters()),26106189)
  self.assertEqual(sum(p.numel() for p in runtime.model.parameters() if p.requires_grad),25986494)
  self.assertTrue(torch.equal((~runtime.model.core.view_masks).sum(1),torch.full((3,),24)))
  self.assertEqual(int((~runtime.model.core.view_masks).sum(0).max()),1)
  self.assertTrue(all(not p.requires_grad for p in runtime.model.core.tokenizer.parameters()))
  with self.assertRaises(ValueError):runtime.predict(x.astype(np.float64))
 def test_asset_binding(self):
  bad=self.root/'bad_norm.npz';np.savez(bad,emg_mean=np.zeros(16,np.float32),emg_std=np.ones(16,np.float32),force_mean=np.ones(5,np.float32),force_std=np.ones(5,np.float32))
  with self.assertRaisesRegex(ValueError,'mismatch'):ThreeViewInference(self.checkpoint,bad,'cpu')
 def test_gap_sampling_and_test_closure(self):
  rows=[dict(stage='legacy/8',subject='synthetic',hands=[dict(side='left')])];index=Index(rows,None,prepared=[[(0,4100),(6000,10500)]])
  for k in range(100):
   i,start=index.sample(1,k);self.assertEqual(i,0);self.assertTrue(0<=start<=100 or 6000<=start<=6500);self.assertEqual(index.sample(1,k),(i,start))
  self.assertEqual(list(index.fixed_windows()),[(0,0),(0,6000)])
  with self.assertRaises(PermissionError):LocalReader().load(dict(split='test'))
 def test_task83_readonly_npz(self):
  p=self.root/'wave.npz';e=np.ones((4100,16),np.float32);f=np.ones((110,5),np.float32);np.savez(p,emg=e,force=f)
  with self.assertRaises(AssertionError):zip_arrays(p)
  with zipfile.ZipFile(p,'a') as z:z.comment=b'old_sos_2khz_v1'
  a,b=zip_arrays(p);self.assertIsInstance(a,np.memmap);self.assertFalse(a.flags.writeable);self.assertTrue(np.array_equal(a,e));self.assertTrue(np.array_equal(b,f))
 def test_label_domains(self):
  x=np.arange(-10,10,dtype=np.float32).reshape(4,5);y=transform(x,layout='1x5x3',domain='raw_extracted_N')
  self.assertTrue(np.array_equal(x,y));self.assertFalse(np.shares_memory(x,y))
  with self.assertRaises(AssertionError):transform(x,layout='unknown',domain='raw_extracted_N')
  with self.assertRaises(AssertionError):transform(x,layout='1x5x1',domain='unknown')
 def test_private_asset_allowlist(self):
  with tempfile.TemporaryDirectory() as folder:
   base=Path(folder);assets=base/'private';run=base/'run'
   for n in REQUIRED:
    p=assets/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
   (assets/'data/CLOCK_BINDINGS.json').write_text('{"rows":{}}');(assets/'.env').write_text('synthetic-do-not-copy')
   result=initialize(assets,run);self.assertEqual(result['status'],'CREATED_NOT_TRAINING');self.assertTrue((run/'source_candidate/model.py').is_file());self.assertFalse((run/'.env').exists())
   with self.assertRaises(FileExistsError):initialize(assets,run)
 def test_full_batch_loss_and_two_phase_gradients(self):
  raw=torch.stack([torch.linspace(.1,1,275).reshape(5,55),torch.linspace(5,1,275).reshape(5,55)])
  pred=torch.cat([raw[:1]*2+.2,raw[1:]*.15+.7],0).requires_grad_()
  norm=dict(force_mean=np.zeros(5,np.float32),force_std=np.ones(5,np.float32))
  whole=a_loss(pred,raw,norm)[0];chunks=torch.stack([a_loss(p,y,norm)[0] for p,y in zip(pred.split(1),raw.split(1))]).mean()
  self.assertGreater(float(abs(whole-chunks).detach()),1e-3)
  for phase in (1,2):
   loss,a,rank,_,_=total(pred,raw,norm,[1]*5,42,phase,.1)
   self.assertTrue(torch.equal(loss,rank+.1*a if phase==1 else a+.1*rank))
   self.assertTrue(torch.isfinite(torch.autograd.grad(loss,pred)[0]).all())
if __name__=='__main__':unittest.main()
