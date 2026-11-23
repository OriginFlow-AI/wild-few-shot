"""Copy an explicit code/asset allowlist into a new independent run directory."""
from pathlib import Path
import argparse,hashlib,json,shutil
REQUIRED=['data/FULL_SPLIT.json.gz','data/TASK43_LOCAL_INVENTORY.json.gz','data/CLOCK_BINDINGS.json','data/LABEL_BINDINGS.json.gz','LABEL_POLICY.json','label_calibration/LABEL_POLICY.json','label_calibration/calibration_assets/coefficients.json','label_calibration/calibration_assets/tables.npz']
OPTIONAL=['data/INDEX.json.gz','data/norm.npz','MCAP_FAST_QUALIFIED.json','tokenizer/TOKENIZER_FROZEN.pt','tokenizer/TOKENIZER_FROZEN.json']
DOCUMENTS=['README.md','DATA_SCHEMA.md','PROTOCOL.json','SOURCE_PROVENANCE.json','requirements.txt','requirements-cuda.txt','VALIDATION.json']
def initialize(assets,destination):
 assets=Path(assets).resolve();destination=Path(destination).resolve();code=Path(__file__).resolve().parent
 if destination.exists():raise FileExistsError('Use a new run directory')
 if destination.is_relative_to(code) or assets.is_relative_to(code):raise ValueError('Private assets/runs must be outside the code package')
 if assets==destination or assets.is_relative_to(destination) or destination.is_relative_to(assets):raise ValueError('Separate source assets and destination')
 for n in REQUIRED:
  if not (assets/n).is_file():raise FileNotFoundError(assets/n)
 token_names=OPTIONAL[-2:]
 if (assets/token_names[0]).exists()!=(assets/token_names[1]).exists():raise ValueError('Tokenizer checkpoint and receipt must be supplied together')
 if (assets/token_names[0]).exists() and not (assets/'data/norm.npz').exists():raise ValueError('Fitted tokenizer requires its matching norm')
 clocks=json.loads((assets/'data/CLOCK_BINDINGS.json').read_bytes())['rows']
 for binding in clocks.values():
  relative=Path(binding['path'])
  if relative.is_absolute() or '..' in relative.parts or not relative.parts or relative.parts[0]!='data':raise ValueError('Clock artifacts must be relative paths under data/')
  p=assets/relative
  if hashlib.sha256(p.read_bytes()).hexdigest()!=binding['sha256']:raise ValueError('Clock digest mismatch')
 destination.mkdir(parents=True)
 selected=[p for p in code.rglob('*.py') if '__pycache__' not in p.parts and 'tests' not in p.relative_to(code).parts]
 selected += [code/n for n in DOCUMENTS]
 for p in selected:
  q=destination/p.relative_to(code);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
 names=set(REQUIRED+[n for n in OPTIONAL if (assets/n).exists()]+[v['path'] for v in clocks.values()]);receipts={}
 for n in sorted(names):
  q=destination/n;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(assets/n,q);receipts[n]=hashlib.sha256(q.read_bytes()).hexdigest()
 for n in ('calibration','qualification','runs','snapshots','tokenizer'):(destination/n).mkdir(exist_ok=True)
 receipt=dict(status='CREATED_NOT_TRAINING',private_assets=receipts,Test_read=False)
 (destination/'INITIALIZED.json').write_text(json.dumps(receipt,separators=(',',':')))
 return receipt
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--assets',required=True);p.add_argument('--run',required=True);a=p.parse_args();print(json.dumps(initialize(a.assets,a.run)))
