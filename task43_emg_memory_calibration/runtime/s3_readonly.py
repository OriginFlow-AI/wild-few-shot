"""Configurable SigV4 read-only client, derived from the validated training client.

No SDK installation or credential copying. Credentials stay in remote memory.
Only HEAD and GET are supported. Exception text never includes signed headers.
"""
import hashlib
import hmac
import json
import os
from pathlib import Path
import urllib.parse
import urllib.request
import urllib.error
from datetime import datetime, timezone

REGION='us-east-1'


class ReadError(RuntimeError):
    def __init__(self,kind,status=None):
        super().__init__(kind)
        self.kind,self.status=kind,status


class Client:
    def __init__(self, endpoint=None, region=None, credentials_file=None):
        self.endpoint=(endpoint or os.environ.get('S3_ENDPOINT_URL','')).rstrip('/')
        parsed=urllib.parse.urlsplit(self.endpoint)
        if parsed.scheme not in ('http','https') or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError('Set a valid credential-free S3_ENDPOINT_URL')
        self.region=region or os.environ.get('AWS_DEFAULT_REGION',REGION)
        c={}
        credentials_file=credentials_file or os.environ.get('S3_CREDENTIALS_FILE')
        if credentials_file:
            p=Path(credentials_file).expanduser();stat=p.stat()
            if os.name=='posix' and (stat.st_uid!=os.getuid() or stat.st_mode&0o077):
                raise PermissionError('Credential file must be caller-owned and mode0600')
            c=json.loads(p.read_text())
        self.ak=os.environ.get('AWS_ACCESS_KEY_ID') or c.get('access_key_id')
        self.sk=os.environ.get('AWS_SECRET_ACCESS_KEY') or c.get('secret_access_key')
        self.token=os.environ.get('AWS_SESSION_TOKEN') or c.get('session_token')
        if not self.ak or not self.sk:raise ReadError('Credentials_missing')
        c.clear();self.host=parsed.netloc
        self.opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(self,method,bucket,key='',*,query=None,headers=None,limit=256<<20):
        if method not in ['HEAD','GET']:raise PermissionError('Read-only API')
        now=datetime.now(timezone.utc);date=now.strftime('%Y%m%d');stamp=now.strftime('%Y%m%dT%H%M%SZ')
        scope=date+'/'+self.region+'/s3/aws4_request';payload=hashlib.sha256(b'').hexdigest()
        uri=urllib.parse.quote('/'+bucket+('/'+key if key else ''),safe='/-_.~')
        quote=lambda s:urllib.parse.quote(str(s),safe='-_.~')
        qs='&'.join(quote(k)+'='+quote(v) for k,v in sorted((query or {}).items()))
        hs={'host':self.host,'x-amz-content-sha256':payload,'x-amz-date':stamp}
        if self.token:hs['x-amz-security-token']=self.token
        hs.update({k.lower():str(v).strip() for k,v in (headers or {}).items()})
        signed=';'.join(sorted(hs));canonical_headers=''.join(k+':'+hs[k]+'\n' for k in sorted(hs))
        canonical='\n'.join([method,uri,qs,canonical_headers,signed,payload])
        message='\n'.join(['AWS4-HMAC-SHA256',stamp,scope,hashlib.sha256(canonical.encode()).hexdigest()])
        sk=('AWS4'+self.sk).encode()
        for part in [date,self.region,'s3','aws4_request']:sk=hmac.new(sk,part.encode(),hashlib.sha256).digest()
        signature=hmac.new(sk,message.encode(),hashlib.sha256).hexdigest()
        hs['Authorization']='AWS4-HMAC-SHA256 Credential='+self.ak+'/'+scope+', SignedHeaders='+signed+', Signature='+signature
        req=urllib.request.Request(self.endpoint+uri+('?' +qs if qs else ''),method=method,headers=hs)
        try:
            with self.opener.open(req,timeout=15) as r:
                body=r.read(limit+1) if method=='GET' else b''
                if len(body)>limit:raise ReadError('Bounded_response_exceeded')
                return dict(status=r.status,headers=dict(r.headers),body=body)
        except urllib.error.HTTPError as e:raise ReadError('HTTP_'+str(e.code),e.code) from None
        except urllib.error.URLError as e:raise ReadError('URL_'+type(e.reason).__name__) from None
        except TimeoutError:raise ReadError('Timeout') from None
