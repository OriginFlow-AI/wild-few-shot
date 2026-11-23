"""Validate the supplied frozen cohort; no raw signal or held-out reads."""
import hashlib
import json
from pathlib import Path
import re
from datetime import datetime,timezone,timedelta


def identity(value):
    m=re.fullmatch(r'([a-z]+)-?(\d{4})(?:\d+)?',value.strip(),re.I|re.ASCII)
    if m is None:raise ValueError('Resolve subject identity before preparing a plan')
    return m[1].lower()+'-'+m[2]


def safe_key(key):
    if not isinstance(key,str) or not key or Path(key).name!=key or key in ('.','..') or '/' in key or '\\' in key or ':' in key:
        raise ValueError('Invalid flat recording key')


def validate(plan):
    if plan['schema']!='stage3-prefix60-crossstage-v1':raise ValueError('Unsupported plan schema')
    if set(plan['roles'])!={'train','val'}:raise ValueError('Training package accepts exactly Train and Val roles, no Test')
    people={};hashes={};counts={}
    for role,cohort in plan['roles'].items():
        records=cohort['records'];supports=cohort['supports'];queries=cohort['queries'];seen=set();seen_content=set();n=0
        people[role]={identity(r['subject']) for r in records.values()};hashes[role]=set()
        for key,r in records.items():
            safe_key(key)
            if r['key']!=key or r['hand'] not in ('left','right') or r['layout']!='1x5x1' or not isinstance(r['stage'],int) or r['stage'] not in range(1,16):raise ValueError('Record key/hand/layout/stage')
            if r['n']<4000 or abs(r['duration']-r['n']/2000)>1e-6:raise ValueError('Expected aligned2kHz record duration')
            if not r.get('content_hash'):raise ValueError('A frozen content identity is required')
            hashes[role].add(r['content_hash'])
        for rid,s in supports.items():
            safe_key(rid);r=records[s['record']]
            if rid!=s['record'] or s['stop']!=120000 or r['n']<120000 or r['stage']!=3 or s['hand']!=r['hand']:raise ValueError('Invalid Stage3 first60s support')
        for q in queries:
            r=records[q['query']];s=records[supports[q['support']]['record']]
            if q['query'] in seen:raise ValueError('Repeated query')
            seen.add(q['query'])
            if r['content_hash'] in seen_content:raise ValueError('Repeated query content')
            seen_content.add(r['content_hash'])
            if r['stage']==3 or (role=='train' and r['stage'] in (4,5)):raise ValueError('Stage3 is support-only and Train excludes4/5')
            gap=r['start']-s['start']
            if r['subject']!=s['subject'] or r['hand']!=s['hand'] or not s['duration']<=gap<=7200:raise ValueError('Support must be same person/hand and already completed within2h')
            zone=timezone(timedelta(hours=8))
            if datetime.fromtimestamp(r['start'],zone).date()!=datetime.fromtimestamp(s['start'],zone).date():raise ValueError('Support must be on the same calendar day (Asia/Shanghai)')
            if q['start']!=0 or q['windows']!=r['n']//4000 or q.get('cross_hand',False):raise ValueError('Window grid mismatch')
            if r['content_hash']==s['content_hash']:raise ValueError('Query duplicates its support')
            n+=q['windows']
        if not queries or n!=plan['summary'][role]['windows']:raise ValueError('Empty or incomplete window population')
        counts[role]=dict(people=len(people[role]),queries=len(queries),supports=len(supports),windows=n)
    if people['train']&people['val'] or hashes['train']&hashes['val']:raise ValueError('Train/Val subject or content leakage')
    return counts


def read(path):return json.loads(Path(path).read_bytes())
