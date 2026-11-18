"""Evidence-preserving, no-retry transport for the local SGLang OpenAI API."""
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from integration import append, write_json
from request_contract import logical_to_local_request, validate_local_request

MAX_RESPONSE_BYTES = 2_000_000
FORBIDDEN_PROMPT_MARKERS = ('evaluator_payload', '/work/suffix', '/work/verifier', 'credential')

class LocalHTTPTransport:
    def __init__(self, *, base_url, model, output, timeout_s):
        if not isinstance(timeout_s, (int,float)) or timeout_s <= 0: raise ValueError('timeout must be positive')
        self.endpoint=base_url.rstrip('/')+'/chat/completions';self.model=model;self.output=Path(output);self.timeout=timeout_s;self.index=0
    def __call__(self, logical_body):
        index=self.index;self.index+=1
        local=logical_to_local_request(logical_body,local_model=self.model);validate_local_request(local,expected_model=self.model)
        encoded=json.dumps(local,separators=(',',':'),ensure_ascii=False).encode()
        lowered=encoded.lower()
        found=[marker for marker in FORBIDDEN_PROMPT_MARKERS if marker.encode() in lowered]
        if found: raise ValueError('forbidden model prompt markers: '+repr(found))
        body_path=self.output/'local-wire'/f'{index:02d}.body'
        body_path.parent.mkdir(parents=True,exist_ok=True)
        fd=os.open(body_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o444)
        try: os.write(fd,encoded);os.fsync(fd)
        finally: os.close(fd)
        write_json(self.output/'local-wire'/f'{index:02d}.json',local)
        append(self.output/'lifecycle.jsonl',{'event':'local_wire_persisted','index':index,'bytes':len(encoded)})
        request=urllib.request.Request(self.endpoint,data=encoded,headers={'Content-Type':'application/json','Authorization':'Bearer local-model-no-key'},method='POST')
        started=time.monotonic()
        try:
            with urllib.request.urlopen(request,timeout=self.timeout) as response:
                status=response.status
                declared=response.headers.get('Content-Length')
                if declared is not None and int(declared)>MAX_RESPONSE_BYTES: raise ValueError('response exceeds byte cap')
                raw=response.read(MAX_RESPONSE_BYTES+1)
                if len(raw)>MAX_RESPONSE_BYTES: raise ValueError('response exceeds byte cap')
        except Exception as error:
            latency=round(time.monotonic()-started,6)
            raw=getattr(error,'read',lambda *_:b'')(MAX_RESPONSE_BYTES+1) if isinstance(error,urllib.error.HTTPError) else b''
            if isinstance(error,urllib.error.HTTPError): error.close()
            if raw:
                (self.output/'raw-http').mkdir(parents=True,exist_ok=True)
                (self.output/'raw-http'/f'{index:02d}.body').write_bytes(raw[:MAX_RESPONSE_BYTES])
            failure={'event':'local_http_failed','index':index,'failure_type':type(error).__name__,'message':str(error),'latency_seconds':latency,'status':getattr(error,'code',None)}
            write_json(self.output/'http-failures'/f'{index:02d}.json',failure);append(self.output/'lifecycle.jsonl',failure)
            raise
        latency=round(time.monotonic()-started,6)
        raw_path=self.output/'raw-http'/f'{index:02d}.body';raw_path.parent.mkdir(parents=True,exist_ok=True);raw_path.write_bytes(raw)
        write_json(self.output/'http-receipts'/f'{index:02d}.json',{'status':status,'latency_seconds':latency,'response_bytes':len(raw)})
        append(self.output/'lifecycle.jsonl',{'event':'local_http_response_persisted','index':index,'status':status,'latency_seconds':latency,'bytes':len(raw)})
        try: value=json.loads(raw)
        except Exception as error:
            failure={'event':'local_response_malformed','index':index,'failure_type':type(error).__name__,'message':str(error),'latency_seconds':latency}
            write_json(self.output/'http-failures'/f'{index:02d}.json',failure);append(self.output/'lifecycle.jsonl',failure);raise ValueError('malformed local response JSON') from error
        if not isinstance(value,dict) or not isinstance(value.get('choices'),list) or not isinstance(value.get('usage'),dict):
            failure={'event':'local_response_malformed','index':index,'failure_type':'ProtocolError','message':'response envelope','latency_seconds':latency}
            write_json(self.output/'http-failures'/f'{index:02d}.json',failure);append(self.output/'lifecycle.jsonl',failure);raise ValueError('malformed local response envelope')
        return value
