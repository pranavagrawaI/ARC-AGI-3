"""Collection-only Decimal wire authority and exact pinned SDK microsecond projection."""
import base64,datetime as dt,hashlib,json,re,sys
from decimal import Decimal,localcontext
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import quota_wire as frozen
FIELDS={'used_s':'timeUsed','reserved_s':'timeReserved','total_allowed_s':'totalTimeAllowed','minimum_allowed_s':'minimumTimeAllowed'}
SCHEMA='COLLECTION_QUOTA_DECIMAL_PROJECTION_V1'
SERIALIZER_SHA256='17e37bfc533a0083fb91dd994d4ed35bc729e74c5a5e4a442f46eb45126032e5'
def check_parser():
    import inspect,kagglesdk.kaggle_object
    from pathlib import Path
    require(hashlib.sha256(Path(inspect.getfile(kagglesdk.kaggle_object)).read_bytes()).hexdigest()==SERIALIZER_SHA256,'installed parser source changed')
def require(ok,message):
    if not ok:raise ValueError('collection quota projection: '+message)
def wire_duration(text):
    require(isinstance(text,str) and re.fullmatch(r'\d+(?:\.\d{1,9})?s',text) is not None,'invalid installed-parser duration')
    number=text[:-1];whole,_,fraction=number.partition('.')
    nanos=int(whole)*1_000_000_000+int((fraction+'000000000')[:9])
    return Decimal(number),nanos//1000

def wire_values(receipt):
    require(receipt['schema']==SCHEMA and type(receipt['capture_count']) is int and receipt['capture_count']==1,'schema/capture count')
    require(receipt['sdk_response_type']=='ApiGetAcceleratorQuotaStatisticsResponse','SDK response type')
    http=receipt['http'];body=base64.b64decode(http['body_base64'],validate=True)
    require(http['complete'] is True and 0<len(body)<=frozen.MAX_BODY and len(body)==http['bytes'] and hashlib.sha256(body).hexdigest()==http['sha256'],'body bound/hash')
    require(http['status']==200 and http['method']=='POST' and http['endpoint']==frozen.ENDPOINT,'HTTP status/method/endpoint')
    require(set(http['headers'])<=set(frozen.SAFE_HEADERS) and all(isinstance(v,str) and len(v)<=2048 for v in http['headers'].values()),'safe headers')
    require(http['headers'].get('content-type','').split(';')[0].strip().lower()=='application/json','content type')
    raw=json.loads(body);refresh=dt.datetime.fromisoformat(raw['quotaRefreshTime'].replace('Z','+00:00'))
    require(refresh.tzinfo is not None and refresh.utcoffset()==dt.timedelta(0),'explicit UTC refresh')
    values={key:wire_duration(raw['gpuQuota'][wire])[0] for key,wire in FIELDS.items()}
    return values,refresh.astimezone(dt.timezone.utc).isoformat(),raw

def typed_fields(delta):
    return dict(days=delta.days,seconds=delta.seconds,microseconds=delta.microseconds,
                total_microseconds=(delta.days*86400+delta.seconds)*1_000_000+delta.microseconds,
                total_seconds_display_diagnostic=repr(delta.total_seconds()))

def verify_projection(receipt):
    values,refresh,raw=wire_values(receipt)
    require(receipt['refresh_time_utc']==refresh,'refresh projection')
    for key,wire in FIELDS.items():
        value,expected=wire_duration(raw['gpuQuota'][wire]);observed=receipt['typed_projection'][key]
        require(receipt['wire_seconds'][key]==str(value),'Decimal authority changed')
        require(all(type(observed[k]) is int for k in ('days','seconds','microseconds','total_microseconds')),'typed integer components')
        require(observed['days']>=0 and 0<=observed['seconds']<86400 and 0<=observed['microseconds']<1_000_000,'typed component range')
        actual=(observed['days']*86400+observed['seconds'])*1_000_000+observed['microseconds']
        require(actual==observed['total_microseconds']==expected,'exact SDK microsecond projection mismatch')
    return values

def read(api):
    """Verbatim frozen per-client callback behavior, one request; new typed receipt join."""
    from kagglesdk.kernels.types.kernels_api_service import ApiGetAcceleratorQuotaStatisticsResponse
    receipt=dict(schema=SCHEMA,utc=dt.datetime.now(dt.timezone.utc).isoformat(),http=None);captures=[]
    try:
        check_parser()
        with api.build_kaggle_client() as client:
            http=client._http_client
            require(client.kernels.kernels_api_client._client is http,'SDK transport identity')
            previous=http._response_processor
            def process(response):
                try:captures.append(frozen._capture(response))
                finally:
                    if previous is not None:previous(response)
            http._response_processor=process
            try:typed=client.kernels.kernels_api_client.get_accelerator_quota_statistics()
            finally:http._response_processor=previous
            receipt['sdk_response_type']=type(typed).__name__
            require(type(typed) is ApiGetAcceleratorQuotaStatisticsResponse,'SDK response type')
            require(len(captures)==1,'missing/multiple HTTP capture')
            receipt['http']=captures[0];receipt['capture_count']=1
            values,refresh,_=wire_values(receipt);sdk_refresh=typed.quota_refresh_time
            if sdk_refresh.tzinfo is None:sdk_refresh=sdk_refresh.replace(tzinfo=dt.timezone.utc)
            receipt['refresh_time_utc']=sdk_refresh.astimezone(dt.timezone.utc).isoformat()
            receipt['wire_seconds']={key:str(value) for key,value in values.items()}
            receipt['typed_projection']={key:typed_fields(getattr(typed.gpu_quota,field)) for key,field in [('used_s','time_used'),('reserved_s','time_reserved'),('total_allowed_s','total_time_allowed'),('minimum_allowed_s','minimum_time_allowed')]}
            verify_projection(receipt)
    except Exception as error:receipt['parse_error']=dict(type=type(error).__name__,reason=str(error) if str(error).startswith('collection quota projection:') else None)
    finally:
        if len(captures)==1:receipt['http']=captures[0]
        receipt['capture_count']=len(captures);receipt['utc']=dt.datetime.now(dt.timezone.utc).isoformat()
    return receipt

def validate(receipt,policy,now=None):
    import reconciliation
    check_parser()
    require(policy['current_quota']==reconciliation.REVIEWED_QUOTA,'unchanged reviewed window/allowance')
    require(not receipt.get('parse_error'),'invalid measured collection quota')
    values=verify_projection(receipt);expected=policy['current_quota']
    require(receipt['refresh_time_utc']==expected['refresh_time_utc'],'quota window changed')
    require(all(values[k]==Decimal(str(expected[k])) for k in ('total_allowed_s','minimum_allowed_s')),'quota allowance changed')
    now=now or dt.datetime.now(dt.timezone.utc);stamp=dt.datetime.fromisoformat(receipt['utc'])
    require(0<policy['freshness_seconds']<=120 and 0<=(now-stamp).total_seconds()<=policy['freshness_seconds'],'stale/future quota')
    require(stamp<dt.datetime.fromisoformat(receipt['refresh_time_utc']),'quota refresh passed')
    require(values['used_s']>=Decimal(str(expected['used_s'])) and values['reserved_s']>=0 and all(values[k]<=values['total_allowed_s'] for k in ('used_s','reserved_s')),'quota outside window/allowance')
    with localcontext() as context:
        context.prec=50
        require(values['used_s']+values['reserved_s']<=values['total_allowed_s'],'quota outside allowance')
    return values
