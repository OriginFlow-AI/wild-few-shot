"""Evaluate all registered models with one worker per selected shared GPU."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
import time
from queue import Empty, Queue

from prepare_test import atomic, read, sha


def gpu_ids(count):
    readings={}
    for sample in range(3):
        cmd=['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits']
        for line in subprocess.check_output(cmd,text=True).splitlines():
            index,uuid,free,util=(item.strip() for item in line.split(','))
            readings.setdefault((int(index),uuid),[]).append((int(free),int(util)))
        if sample<2:
            time.sleep(2)
    scores=[]
    for (index,uuid),rows in readings.items():
        free=min(r[0] for r in rows)
        util=sum(r[1] for r in rows)/len(rows)
        if free>=15*1024:
            scores.append((util,-free,index,uuid))
    return [(index,uuid) for _,_,index,uuid in sorted(scores)[:count]]


def main(root,registry,contact,reference,max_workers):
    code=Path(__file__).resolve().parent
    command=[sys.executable,'-B']
    subprocess.run([*command,str(code/'verify_targets.py'),'--root',str(root),'--reference',str(reference)],check=True)
    jobs=Queue()
    signal_sha=sha(root/'signal_receipt.json')
    for model in read(registry):
        report=root/'results'/model['name']/'report.json'
        if not report.is_file():
            jobs.put(model)
        else:
            assert read(report)['signal_receipt_sha256']==signal_sha
    if not jobs.empty():
        selected=gpu_ids(max_workers)
        assert selected,'No GPU with enough currently available memory'
        (root/'logs').mkdir(exist_ok=True)
        atomic(root/'eval_gpus.json',dict(gpus=[dict(index=index,uuid=uuid) for index,uuid in selected],
                                        memory_ceiling_GiB=10))

        def work(card):
            index,uuid=card
            errors=[]
            while True:
                try:
                    model=jobs.get_nowait()
                except Empty:
                    break
                name=model['name']
                if 'branch' in model:
                    done=Path(model['root'])/'runs'/model['branch']/'DONE.json'
                    exported=Path(model['root'])/'exported_models'/model['branch']/'inference_parameters.json'
                    while not (done.is_file() and exported.is_file()):
                        status=Path(model['root'])/'STATUS.json'
                        if status.is_file() and read(status).get('phase')=='failed':
                            raise RuntimeError(f'Model training failed: {name}')
                        time.sleep(30)
                env={**os.environ,'CUDA_VISIBLE_DEVICES':uuid}
                with (root/'logs'/f'{name}.log').open('ab') as log:
                    for attempt in (1,2):
                        print(f'Start {name} on GPU {index}, attempt {attempt}',flush=True)
                        result=subprocess.run([*command,str(code/'evaluate.py'),'--root',str(root),
                           '--registry',str(registry),'--contact',str(contact),'--model',name],
                           env=env,stdout=log,stderr=subprocess.STDOUT)
                        if result.returncode==0:
                            break
                    if result.returncode:
                        errors.append(name)
            return errors

        with ThreadPoolExecutor(max_workers=len(selected)) as pool:
            failures=[name for errors in pool.map(work,selected) for name in errors]
        if failures:
            raise RuntimeError(f'Model evaluations failed: {failures}')
    subprocess.run([*command,str(code/'collect.py'),'--root',str(root),'--registry',str(registry),
                    '--output',str(root/'publication')],check=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','registry','contact','reference'):
        parser.add_argument(f'--{name}',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    main(args.root,args.registry,args.contact,args.reference,args.workers)
