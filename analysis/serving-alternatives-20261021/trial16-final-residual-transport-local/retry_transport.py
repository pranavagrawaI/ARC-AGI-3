"""Only selected GET ConnectTimeout/no-response/zero-byte is retryable, once."""
import contextlib,json,os,time
import requests
from settings import require,safe_name

class TrackedResponse:
    def __init__(self,response):self.response=response;self.received=0
    @property
    def status_code(self):return self.response.status_code
    @property
    def headers(self):return self.response.headers
    def raise_for_status(self):return self.response.raise_for_status()
    def iter_content(self,chunk_size):
        for block in self.response.iter_content(chunk_size=chunk_size):
            self.received+=len(block);yield block

class RetryRemote:
    def __init__(self,base,names,destination,end,clock=time.monotonic,sleep=time.sleep):
        self.base=base;self.names=set(names);self.destination=destination;self.end=end;self.clock=clock;self.sleep=sleep
        self.urls={};self.opened=set();self.attempts=0
    def remaining(self,end=None):
        value=min(self.end,self.end if end is None else end)-self.clock()
        require(value>0,'final shared cooperative deadline');return value
    def readback(self,destination,remaining):return self.base.readback(destination,min(remaining,self.remaining()))
    def list_page(self,token,size):
        self.remaining();page,next_token=self.base.list_page(token,size);self.remaining()
        for name,url in page:
            if name not in self.names:continue
            safe_name(name);require(url not in self.urls,'duplicate selected signed URL');self.urls[url]=name
        return page,next_token
    def persisted(self,name):
        path=self.destination/'members'/name
        return path.stat().st_size if path.exists() else 0
    def record(self,name,index,event,**values):
        row=dict(collection_pass=3,name=name,attempt_in_pass=index,global_attempt=self.attempts,event=event,at=self.clock(),**values)
        with (self.destination/'transport-attempts.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n');stream.flush();os.fsync(stream.fileno())
    def error_record(self,error):return self.base.error_record(error)
    @contextlib.contextmanager
    def download(self,url,remaining):
        require(url in self.urls,'download outside selected final residual')
        name=self.urls[url];require(name not in self.opened,'no repeated file invocation/success redownload');self.opened.add(name)
        local_end=min(self.end,self.clock()+remaining)
        for index in (1,2):
            left=self.remaining(local_end);require(self.attempts<334,'final GET attempt cap');self.attempts+=1
            self.record(name,index,'START')
            try:response=self.base.download(url,self.remaining(local_end))
            except BaseException as error:
                received=getattr(error,'received_bytes',0);persisted=self.persisted(name)
                eligible=isinstance(error,requests.ConnectTimeout) and getattr(error,'response',None) is None and received==0 and persisted==0 and index==1
                self.record(name,index,'ERROR',phase='GET',type=type(error).__name__,response_absent=getattr(error,'response',None) is None,
                    received_bytes=received,persisted_bytes=persisted,retry_eligible=eligible,safe_error=self.error_record(error))
                if not eligible:raise
                require(self.remaining(local_end)>1,'no budget for one-second retry delay')
                self.sleep(1);self.remaining(local_end);continue
            tracked=TrackedResponse(response)
            try:
                with response as raw:
                    tracked.response=raw;yield tracked
            except BaseException as error:
                self.record(name,index,'ERROR',phase='RESPONSE_CONTEXT',type=type(error).__name__,response_absent=False,
                    received_bytes=tracked.received,persisted_bytes=self.persisted(name),retry_eligible=False,safe_error=self.error_record(error))
                raise
            else:
                self.record(name,index,'RESPONSE_CONTEXT_EXITED',received_bytes=tracked.received,persisted_bytes=self.persisted(name),retry_eligible=False)
            return
