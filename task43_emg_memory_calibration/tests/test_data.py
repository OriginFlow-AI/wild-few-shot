"""All fixtures are synthetic; no private assets, S3 credentials or dataset."""
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile
import numpy as np

PACKAGE=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(PACKAGE/'runtime'),str(PACKAGE/'tools')]
from source_reader import Source,relative_name
from s3_readonly import Client
from plan import validate
from admission import select_support


def example_plan():
    roles={};summary={}
    for role,subject in [('train','aa-1111'),('val','bb-2222')]:
        records={}
        for kind,stage,n,start in [('support',3,120000,1700000000),('query',7,8000,1700000120)]:
            key=subject+'__'+kind
            records[key]=dict(key=key,subject=subject,stage=stage,hand='left',layout='1x5x1',n=n,duration=n/2000,start=start,relative=subject+'/'+kind,content_hash='synthetic-'+key)
        support=subject+'__support';query=subject+'__query'
        roles[role]=dict(records=records,supports={support:dict(record=support,hand='left',stop=120000)},queries=[dict(query=query,support=support,start=0,windows=2,cross_hand=False)])
        summary[role]=dict(windows=2)
    return dict(schema='stage3-prefix60-crossstage-v1',roles=roles,summary=summary)


class FakeS3:
    def __init__(self,raw,bad_range=False):self.raw=raw;self.bad_range=bad_range;self.gets=0
    def request(self,method,bucket,key,**kwargs):
        headers={'content-length':str(len(self.raw)),'etag':'"synthetic"','last-modified':'Wed, 01 Jan 2020 00:00:00 GMT'}
        if method=='HEAD':return dict(status=200,headers=headers,body=b'')
        a,b=map(int,kwargs['headers']['Range'].split('=')[1].split('-'));self.gets+=1
        headers['content-range']=f'bytes {a}-{b}/{len(self.raw)}'
        return dict(status=200 if self.bad_range else 206,headers=headers,body=self.raw[a:b+1])


def payload(fortran=False,compressed=False):
    emg=np.arange(200*16,dtype=np.float32).reshape(200,16)/7
    force=np.arange(200*5,dtype=np.float32).reshape(200,5)
    if fortran:emg=np.asfortranarray(emg);force=np.asfortranarray(force)
    raw=io.BytesIO()
    with zipfile.ZipFile(raw,'w',compression=zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED) as z:
        for name,value in [('emg',emg),('cmcu05b_left_force_mn',force)]:
            b=io.BytesIO();np.save(b,value,allow_pickle=False);z.writestr(name+'.npy',b.getvalue())
    return raw.getvalue(),emg,force


class InputTests(unittest.TestCase):
    def test_c_and_fortran_ranges_match_numpy(self):
        for order in [False,True]:
            with self.subTest(fortran=order),tempfile.TemporaryDirectory() as temp:
                raw,emg,force=payload(order);client=FakeS3(raw)
                source=Source(dict(local_roots=[],metadata_cache=temp,s3_bucket='synthetic',s3_prefix='processed'),lambda:client)
                record=dict(relative='synthetic/record',n=200,hand='left',layout='1x5x1')
                for begin,end in [(0,3),(73,129),(198,200)]:
                    np.testing.assert_array_equal(source.rows(record,'emg',begin,end),emg[begin:end])
                    np.testing.assert_array_equal(source.rows(record,'fz',begin,end),force[begin:end]*np.float32(.001))
                self.assertGreater(client.gets,0)

    def test_local_force_is_not_converted_twice(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);emg=np.ones((200,16),np.float32);force=np.ones((200,5),np.float32)*2
            np.save(root/'sample.emg.npy',emg);np.save(root/'sample.fz.npy',force)
            source=Source(dict(local_roots=[temp],metadata_cache=str(root/'headers')))
            record=dict(relative='sample',n=200,hand='left',layout='1x5x1')
            np.testing.assert_array_equal(source.rows(record,'fz',2,4),force[2:4])
            record['local_force_unit']='mN'
            with self.assertRaises(ValueError):source.rows(record,'fz',2,4)

    def test_bad_range_and_compressed_npz_are_rejected(self):
        for bad,compressed in [(True,False),(False,True)]:
            with tempfile.TemporaryDirectory() as temp:
                raw,_,_=payload(compressed=compressed);source=Source(dict(metadata_cache=temp,s3_bucket='synthetic'),lambda:FakeS3(raw,bad))
                with self.assertRaises(ValueError):source.rows(dict(relative='sample',n=200,hand='left',layout='1x5x1'),'emg',0,3)

    def test_paths_and_readonly_api(self):
        for value in ['../outside','/absolute','C:/absolute','a/../b']:
            with self.assertRaises(ValueError):relative_name(value)
        with self.assertRaises(PermissionError):Client.__new__(Client).request('PUT','synthetic','x')

    def test_local_paths_cannot_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            source=Source(dict(local_roots=[temp],metadata_cache=temp))
            with self.assertRaises(PermissionError):source.rows(dict(relative='x',n=200,emg_path='/outside/data.npy'),'emg',0,2)


class PlanTests(unittest.TestCase):
    def test_valid_plan(self):self.assertEqual(validate(example_plan())['train']['windows'],2)
    def test_heldout_role_rejected(self):
        p=example_plan();p['roles']['test']=p['roles']['val']
        with self.assertRaises(ValueError):validate(p)
    def test_subject_alias_leakage_rejected(self):
        p=example_plan()
        for r in p['roles']['val']['records'].values():r['subject']='AA111100'
        with self.assertRaises(ValueError):validate(p)
    def test_future_support_and_forbidden_train_stage(self):
        for stage,future in [(4,False),(7,True)]:
            p=example_plan();c=p['roles']['train'];q=c['records'][c['queries'][0]['query']];q['stage']=stage
            if future:q['start']=1699999999
            with self.assertRaises(ValueError):validate(p)

    def test_nearest_completed_support_and_no_older_fallback(self):
        p=example_plan();records=p['roles']['train']['records'];s=next(r for r in records.values() if r['stage']==3);q=next(r for r in records.values() if r['stage']!=3)
        self.assertEqual(select_support(q,[s])[1],'supported')
        newer=dict(s,key='newer',start=s['start']+70,n=80000,duration=40)
        self.assertEqual(select_support(q,[s,newer])[1],'nearest_support_ineligible_no_fallback')
        other=dict(s,subject='other-3333')
        self.assertEqual(select_support(q,[other])[1],'no_prior_completed_same_hand_stage3')

    def test_cold_fallback_outside_time_and_gen3(self):
        p=example_plan();records=p['roles']['train']['records'];s=next(r for r in records.values() if r['stage']==3);q=next(r for r in records.values() if r['stage']!=3)
        self.assertEqual(select_support(dict(q,start=s['start']+7201),[s])[1],'outside_2h_or_date')
        self.assertEqual(select_support(dict(q,layout='1x5x3'),[s])[1],'query_not_CMCU')


if __name__=='__main__':unittest.main()
