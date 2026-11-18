"""One SDK quota request; bounded actual HTTP evidence joined to typed durations."""
import base64,datetime as dt,hashlib,json,math,re
MAX_BODY=64*1024
ENDPOINT='https://api.kaggle.com/v1/kernels.KernelsApiService/GetAcceleratorQuotaStatistics'
SAFE_HEADERS=('date','content-type','content-length')
def require(ok,message):
    if not ok:raise ValueError('quota wire: '+message)
def quota_values(raw):
    gpu=raw['gpuQuota']
    def seconds(key):
        value=gpu[key]
        require(isinstance(value,str) and re.fullmatch(r'\d+(?:\.\d+)?s',value) is not None,'missing/malformed duration '+key)
        value=float(value[:-1]);require(math.isfinite(value),'nonfinite duration');return value
    refresh=dt.datetime.fromisoformat(raw['quotaRefreshTime'].replace('Z','+00:00'))
    require(refresh.tzinfo is not None and refresh.utcoffset()==dt.timedelta(0),'refresh must be explicit UTC')
    return dict(used_s=seconds('timeUsed'),reserved_s=seconds('timeReserved'),total_allowed_s=seconds('totalTimeAllowed'),minimum_allowed_s=seconds('minimumTimeAllowed'),refresh_time_utc=refresh.astimezone(dt.timezone.utc).isoformat())
def values_from_receipt(receipt):
    require(receipt['schema']=='QUOTA_HTTP_TYPED_V1' and type(receipt['capture_count']) is int and receipt['capture_count']==1,'receipt schema/capture count')
    require(receipt['sdk_response_type']=='ApiGetAcceleratorQuotaStatisticsResponse','SDK quota response type')
    http=receipt['http'];body=base64.b64decode(http['body_base64'],validate=True)
    require(http['complete'] is True and 0<len(body)<=MAX_BODY and len(body)==http['bytes'] and hashlib.sha256(body).hexdigest()==http['sha256'],'body bound/completeness/hash')
    require(http['status']==200 and http['method']=='POST' and http['endpoint']==ENDPOINT,'status/method/endpoint')
    require(set(http['headers'])<=set(SAFE_HEADERS) and all(isinstance(v,str) and len(v)<=2048 for v in http['headers'].values()),'unsafe headers')
    require(http['headers'].get('content-type','').split(';')[0].strip().lower()=='application/json','content type')
    return quota_values(json.loads(body))
def _capture(response):
    import requests
    require(isinstance(response,requests.Response),'HTTP response type')
    body=response.content;require(isinstance(body,bytes),'HTTP body type')
    request=response.request;method=getattr(request,'method',None);url=getattr(request,'url',None)
    # Never persist arbitrary URLs/query strings or credential-bearing headers.
    match=url==ENDPOINT and response.url==ENDPOINT
    headers={k:response.headers[k] for k in SAFE_HEADERS if k in response.headers}
    require(all(isinstance(v,str) and len(v)<=2048 for v in headers.values()),'safe header bound')
    return dict(status=response.status_code,method=method if method in ('POST','GET','PUT','DELETE') else None,endpoint=ENDPOINT if match else None,
                headers=headers,bytes=len(body),complete=len(body)<=MAX_BODY,sha256=hashlib.sha256(body).hexdigest() if len(body)<=MAX_BODY else None,
                body_base64=base64.b64encode(body[:MAX_BODY]).decode())
def read(api):
    """Existing per-client callback is chained/restored. No retry or extra request."""
    from kagglesdk.kernels.types.kernels_api_service import ApiGetAcceleratorQuotaStatisticsResponse
    receipt=dict(schema='QUOTA_HTTP_TYPED_V1',utc=dt.datetime.now(dt.timezone.utc).isoformat(),http=None);captures=[]
    try:
        with api.build_kaggle_client() as client:
            http=client._http_client
            require(client.kernels.kernels_api_client._client is http,'SDK transport identity')
            previous=http._response_processor
            def process(response):
                try:captures.append(_capture(response))
                finally:
                    if previous is not None:previous(response)
            http._response_processor=process
            try:typed=client.kernels.kernels_api_client.get_accelerator_quota_statistics()
            finally:http._response_processor=previous
            receipt['sdk_response_type']=type(typed).__name__
            require(type(typed) is ApiGetAcceleratorQuotaStatisticsResponse,'SDK quota response type')
            require(len(captures)==1,'missing/multiple HTTP capture')
            receipt['http']=captures[0];receipt['capture_count']=1;gpu=typed.gpu_quota;refresh=typed.quota_refresh_time
            # Installed SDK strips wire Z into naive UTC. Its source is pinned.
            if refresh.tzinfo is None:refresh=refresh.replace(tzinfo=dt.timezone.utc)
            values=dict(used_s=gpu.time_used.total_seconds(),reserved_s=gpu.time_reserved.total_seconds(),total_allowed_s=gpu.total_time_allowed.total_seconds(),minimum_allowed_s=gpu.minimum_time_allowed.total_seconds(),refresh_time_utc=refresh.astimezone(dt.timezone.utc).isoformat())
            receipt.update(values);require(values==values_from_receipt(receipt),'wire/typed disagreement')
    except Exception as error:
        # Exception text may contain request/auth data. Keep only its type.
        receipt['parse_error']=dict(type=type(error).__name__,reason=str(error) if str(error).startswith('quota wire:') else None)
    finally:
        if len(captures)==1:receipt['http']=captures[0]
        receipt['capture_count']=len(captures);receipt['utc']=dt.datetime.now(dt.timezone.utc).isoformat()
    return receipt
