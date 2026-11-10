"""Start the authorized background pipeline with a short, private runtime TMPDIR."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from common import atomic

def main(root,workers,credential_file=None):
    root=Path(root).expanduser().resolve();root.mkdir(parents=True,exist_ok=True)
    for name in ['logs','checks']:(root/name).mkdir(exist_ok=True)
    record=root/'checks/background_launcher.json'
    if record.exists():
        old=json.loads(record.read_text());p=Path('/proc')/str(old['pid'])/'cmdline'
        if p.exists() and str(root).encode() in p.read_bytes() and b'pipeline.py' in p.read_bytes():
            print('Pipeline already running:',old['pid']);return
    temp=root.parent/('.t103h_'+hashlib.sha256(str(root).encode()).hexdigest()[:8]);temp.mkdir(exist_ok=True,mode=0o700)
    env=dict(os.environ,TMPDIR=str(temp),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    if credential_file:env['TASK103_CREDENTIAL_FILE']=str(Path(credential_file).resolve())
    command=[sys.executable,'-B',str(Path(__file__).with_name('pipeline.py')),'--root',str(root),'--workers',str(workers)]
    with (root/'logs/pipeline.log').open('a') as log:
        p=subprocess.Popen(command,env=env,cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    atomic(record,dict(pid=p.pid,command=command,tmp_directory=str(temp)))
    print(json.dumps(dict(pid=p.pid,root=str(root),log=str(root/'logs/pipeline.log'))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--credential-file');a=p.parse_args();main(a.root,a.workers,a.credential_file)
