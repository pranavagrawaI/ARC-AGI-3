"""HTTP fixture only: actual installed SDK call/_prepare_response, intercepted send."""
import contextlib,copy,json
from pathlib import Path
from unittest.mock import patch
try:
    import requests
    from kagglesdk.kaggle_client import KaggleClient
except ImportError:requests=None;KaggleClient=None
HERE=Path(__file__).resolve().parents[1]
BODY=(HERE/'operational-evidence/coordinator-wire-diagnosis/response-body.json').read_bytes()
RAW=json.loads(BODY)
class HttpApi:
    def __init__(self,bodies=None,*,endpoint=None,status=200,content_type='application/json',response_type=None,missing=False,prior_error=False,mutate_typed=None):
        self.bodies=copy.deepcopy(bodies or [BODY]);self.endpoint=endpoint;self.status=status;self.content_type=content_type;self.response_type=response_type;self.missing=missing;self.prior_error=prior_error;self.mutate_typed=mutate_typed
        self.sends=[];self.prior_calls=[];self.clients=[]
    @contextlib.contextmanager
    def build_kaggle_client(self):
        def prior(response):
            self.prior_calls.append(response)
            if self.prior_error:raise RuntimeError('CPU_CALLBACK_FAILURE')
        client=KaggleClient(response_processor=prior);http=client._http_client
        http._session=requests.Session();self.clients.append((http,prior));original=http._prepare_response
        def prepare(response_type,response):
            active=http._response_processor
            if self.missing:http._response_processor=None
            try:result=original(response_type,response)
            finally:http._response_processor=active
            if self.mutate_typed is not None:self.mutate_typed(result)
            return result if self.response_type is None else self.response_type
        def send(request,**kwargs):
            self.sends.append((request.method,request.url))
            body=self.bodies.pop(0);body=body if isinstance(body,bytes) else json.dumps(body,separators=(',',':')).encode()
            response=requests.Response();response.status_code=self.status;response._content=body;response.request=request
            if self.endpoint is not None:response.request=requests.Request('POST',self.endpoint).prepare()
            response.url=response.request.url
            response.headers.update({'Content-Type':self.content_type,'Date':'Sat, 24 Oct 2026 00:35:33 GMT','Set-Cookie':'CPU_SECRET_MUST_NOT_BE_RECORDED'})
            return response
        with patch.object(http._session,'send',side_effect=send),patch.object(http,'_prepare_response',side_effect=prepare):
            try:yield client
            finally:http._session.close()
