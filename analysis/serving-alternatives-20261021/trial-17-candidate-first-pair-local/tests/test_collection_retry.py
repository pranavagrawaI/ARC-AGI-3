import json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'collection_tools'))
import requests
from retry_transport import RetryRemote

class Clock:
    def __init__(self):self.now=0
    def __call__(self):return self.now
    def sleep(self,seconds):self.now+=seconds
class Response:
    def __init__(self,status=200,stream_error=None):self.status_code=status;self.headers={'content-length':'1'};self.stream_error=stream_error
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def raise_for_status(self):
        if self.status_code!=200:raise requests.HTTPError(response=self)
    def iter_content(self,chunk_size):
        yield b'x'
        if self.stream_error:raise self.stream_error
class Base:
    def __init__(self,outcomes):self.outcomes=list(outcomes);self.calls=0
    def list_page(self,token,size):return [('a','https://fixture.invalid/a?secret=HIDDEN')],''
    def download(self,url,remaining):
        self.calls+=1;v=self.outcomes.pop(0)
        if isinstance(v,BaseException):raise v
        return v
    def readback(self,*args):raise requests.ConnectTimeout()
    @staticmethod
    def error_record(error):return {'type':type(error).__name__}

class RetryTests(unittest.TestCase):
    def adapter(self,p,outcomes,end=900):
        clock=Clock();base=Base(outcomes);api=RetryRemote(base,{'a'},p,end,clock=clock,sleep=clock.sleep)
        api.list_page('',100);return api,base,clock
    def fetch(self,api):
        with api.download('https://fixture.invalid/a?secret=HIDDEN',900) as response:
            response.raise_for_status();return b''.join(response.iter_content(chunk_size=65536))
    def records(self,p):return [json.loads(x) for x in (p/'transport-attempts.jsonl').read_bytes().splitlines()]
    def test_zero_response_timeout_then_success(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);api,base,clock=self.adapter(p,[requests.ConnectTimeout(),Response()])
            self.assertEqual(self.fetch(api),b'x');self.assertEqual(base.calls,2);self.assertEqual(clock.now,1)
            r=self.records(p);self.assertEqual([x['event'] for x in r],['START','ERROR','START','RESPONSE_CONTEXT_EXITED'])
            self.assertTrue(r[1]['retry_eligible']);self.assertEqual(r[2]['attempt_in_pass'],2);self.assertEqual(r[3]['received_bytes'],1)
            self.assertNotIn('HIDDEN',json.dumps(r));self.assertNotIn('https://',json.dumps(r))
    def test_second_timeout_stops(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);api,base,_=self.adapter(p,[requests.ConnectTimeout(),requests.ConnectTimeout(),Response()])
            with self.assertRaises(requests.ConnectTimeout):self.fetch(api)
            self.assertEqual(base.calls,2);self.assertFalse(self.records(p)[-1]['retry_eligible'])
    def test_response_present_timeout_not_retried(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);e=requests.ConnectTimeout(response=Response());api,base,_=self.adapter(p,[e,Response()])
            with self.assertRaises(requests.ConnectTimeout):self.fetch(api)
            self.assertEqual(base.calls,1)
    def test_read_timeout_not_retried(self):
        with tempfile.TemporaryDirectory() as t:
            api,base,_=self.adapter(Path(t),[requests.ReadTimeout(),Response()])
            with self.assertRaises(requests.ReadTimeout):self.fetch(api)
            self.assertEqual(base.calls,1)
    def test_generic_connection_error_not_retried(self):
        with tempfile.TemporaryDirectory() as t:
            api,base,_=self.adapter(Path(t),[requests.ConnectionError(),Response()])
            with self.assertRaises(requests.ConnectionError):self.fetch(api)
            self.assertEqual(base.calls,1)
    def test_http_error_not_retried(self):
        with tempfile.TemporaryDirectory() as t:
            api,base,_=self.adapter(Path(t),[Response(503),Response()])
            with self.assertRaises(requests.HTTPError):self.fetch(api)
            self.assertEqual(base.calls,1)
    def test_partial_body_timeout_not_retried(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);api,base,_=self.adapter(p,[Response(stream_error=requests.ConnectTimeout()),Response()])
            with self.assertRaises(requests.ConnectTimeout):self.fetch(api)
            self.assertEqual(base.calls,1);self.assertEqual(self.records(p)[-1]['received_bytes'],1);self.assertFalse(self.records(p)[-1]['retry_eligible'])
    def test_nonzero_persisted_timeout_not_retried(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'members').mkdir();(p/'members/a').write_bytes(b'x');api,base,_=self.adapter(p,[requests.ConnectTimeout(),Response()])
            with self.assertRaises(requests.ConnectTimeout):self.fetch(api)
            self.assertEqual(base.calls,1);self.assertEqual(self.records(p)[-1]['persisted_bytes'],1)
    def test_wait_cannot_exceed_shared_deadline(self):
        with tempfile.TemporaryDirectory() as t:
            api,base,_=self.adapter(Path(t),[requests.ConnectTimeout(),Response()],end=.5)
            with self.assertRaises(ValueError):self.fetch(api)
            self.assertEqual(base.calls,1)
    def test_success_cannot_be_redownloaded(self):
        with tempfile.TemporaryDirectory() as t:
            api,base,_=self.adapter(Path(t),[Response(),Response()]);self.fetch(api)
            with self.assertRaises(ValueError):self.fetch(api)
            self.assertEqual(base.calls,1)
    def test_nonzero_observed_bytes_not_retried(self):
        with tempfile.TemporaryDirectory() as t:
            e=requests.ConnectTimeout();e.received_bytes=1;api,base,_=self.adapter(Path(t),[e,Response()])
            with self.assertRaises(requests.ConnectTimeout):self.fetch(api)
            self.assertEqual(base.calls,1)
    def test_shared_cap_rechecked_after_journal_persistence(self):
        with tempfile.TemporaryDirectory() as t:
            api,base,clock=self.adapter(Path(t),[Response()]);original=api.record
            def delayed(*args,**kwargs):
                original(*args,**kwargs)
                if args[2]=='START':clock.now=901
            api.record=delayed
            with self.assertRaises(ValueError):self.fetch(api)
            self.assertEqual(base.calls,0)
    def test_global_attempt_cap_blocks_before_get(self):
        with tempfile.TemporaryDirectory() as t:
            api,base,_=self.adapter(Path(t),[Response()]);api.attempts=40000
            with self.assertRaises(ValueError):self.fetch(api)
            self.assertEqual(base.calls,0)
    def test_actual_download_timeouts_unchanged(self):
        from remote import Remote
        from unittest.mock import patch
        with patch('requests.get',return_value=Response()) as get:
            Remote().download('https://fixture.invalid/a',900)
            self.assertEqual(get.call_args.kwargs['timeout'],(10,30))
            self.assertFalse(get.call_args.kwargs['allow_redirects']);self.assertTrue(get.call_args.kwargs['stream'])
            self.assertEqual(get.call_args.kwargs['headers'],{'Accept-Encoding':'identity'})
    def test_readback_timeout_not_retried(self):
        with tempfile.TemporaryDirectory() as t:
            api,base,_=self.adapter(Path(t),[])
            with self.assertRaises(requests.ConnectTimeout):api.readback(Path(t)/'before',900)
            self.assertEqual(base.calls,0)
    def test_listing_timeout_not_retried(self):
        class BadList(Base):
            def list_page(self,*args):self.calls+=1;raise requests.ConnectTimeout()
        with tempfile.TemporaryDirectory() as t:
            b=BadList([]);c=Clock();a=RetryRemote(b,{'a'},Path(t),900,clock=c,sleep=c.sleep)
            with self.assertRaises(requests.ConnectTimeout):a.list_page('',100)
            self.assertEqual(b.calls,1)
if __name__=='__main__':unittest.main()
