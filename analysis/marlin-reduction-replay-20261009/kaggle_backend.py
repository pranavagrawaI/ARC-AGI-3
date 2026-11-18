"""Current installed SDK boundary. No automatic push retries or credential logging."""
import datetime as dt
import json
from pathlib import Path


def bound_http():
    import requests
    send=requests.Session.send
    def bounded(session,request,**kwargs):
        kwargs['timeout']=(15,45)
        return send(session,request,**kwargs)
    requests.Session.send=bounded


class KaggleBackend:
    def __init__(self):
        bound_http()
        from kaggle.api.kaggle_api_extended import KaggleApi
        from kagglesdk import KaggleClient
        from kagglesdk.kaggle_creds import KaggleCredentials
        self.api=KaggleApi();self.creds=KaggleCredentials.load(KaggleClient())
        self.refresh_auth()

    def refresh_auth(self):
        if self.creds is None:
            self.api.authenticate();return
        expiry=self.creds._access_token_expiration
        if not self.creds._access_token or expiry is None or expiry-dt.datetime.now(dt.timezone.utc)<dt.timedelta(hours=1):
            # generate_access_token refreshes the same authorization without save()
            # touching the user's protected home directory. New credentials stay in memory.
            response=self.creds.generate_access_token()
            self.creds._access_token=response.token
            self.creds._access_token_expiration=dt.datetime.now(dt.timezone.utc)+dt.timedelta(seconds=response.expires_in)
        self.api.config_values[self.api.CONFIG_NAME_TOKEN]=self.creds._access_token
        self.api.config_values[self.api.CONFIG_NAME_USER]=self.creds.get_username()
        self.api._authenticated=True
        if self.creds.get_username()!='pranavagrawai':raise ValueError('Unexpected authenticated Kaggle owner')

    def quota(self):
        self.refresh_auth()
        with self.api.build_kaggle_client() as client:return client.kernels.kernels_api_client.get_accelerator_quota_statistics()

    def active_sessions(self,owner):
        kernels=self.api.kernels_list(mine=True,user=owner,sort_by='dateRun',page_size=20)
        active=[]
        for kernel in kernels:
            ref=getattr(kernel,'ref',None)
            if not ref:
                stamp=getattr(kernel,'last_run_time',None)
                if (getattr(kernel,'title',None)=='[Private Notebook]' and not getattr(kernel,'author',None)
                    and isinstance(stamp,dt.datetime) and stamp.replace(tzinfo=dt.timezone.utc)==dt.datetime(2010,4,1,tzinfo=dt.timezone.utc)
                    and not getattr(kernel,'current_version_number',None)):continue
                active.append({'ref':None,'status':'unknown identity'});continue
            status=self.status(ref)
            if status['status'] in {'RUNNING','QUEUED'}:active.append({'ref':ref,'status':status['status']})
            elif status['status'] not in {'COMPLETE','ERROR','CANCELED','CANCELLED'}:raise ValueError('Unknown live session status')
        return active

    def get_kernel(self,slug):
        from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
        owner,name=slug.split('/');request=ApiGetKernelRequest();request.user_name=owner;request.kernel_slug=name
        with self.api.build_kaggle_client() as client:return client.kernels.kernels_api_client.get_kernel(request)

    def require_new_slug(self,slug):
        import requests
        try:self.get_kernel(slug)
        except requests.HTTPError as error:
            if error.response is not None and error.response.status_code==404:return
            raise
        raise ValueError('Remote slug already exists; do not reuse a consumed or ambiguous run identity')

    def push(self,folder,timeout_s):
        return json.loads(str(self.api.kernels_push(str(folder),timeout=str(timeout_s))))

    def status(self,slug):
        self.refresh_auth();return json.loads(str(self.api.kernels_status(slug)))

    def verify_remote(self,slug,receipt,expected_notebook,expected_meta):
        response=self.get_kernel(slug);meta=response.metadata
        if meta.id!=receipt['kernelId'] or meta.current_version_number!=receipt['versionNumber']:
            raise ValueError('Remote latest identity differs from the exclusive accepted version')
        remote=json.loads(response.blob.source)
        def sources(nb):return [(cell['cell_type'],''.join(cell['source']) if isinstance(cell['source'],list) else cell['source']) for cell in nb['cells']]
        if sources(remote)!=sources(expected_notebook):raise ValueError('Remote notebook cells differ from frozen source')
        fields={'dataset_sources':'dataset_data_sources','model_sources':'model_data_sources','competition_sources':'competition_data_sources',
                'is_private':'is_private','enable_gpu':'enable_gpu','enable_tpu':'enable_tpu','enable_internet':'enable_internet',
                'machine_shape':'machine_shape','docker_image':'docker_image','kernel_sources':'kernel_data_sources','language':'language','kernel_type':'kernel_type'}
        for local,attr in fields.items():
            value=getattr(meta,attr)
            if isinstance(value,str) and local in {'language','kernel_type'}:value=value.lower()
            if value!=expected_meta[local]:raise ValueError('Remote launch metadata differs: '+local)
        return {'kernel_id':meta.id,'version':meta.current_version_number,'cells_verified':len(remote['cells']),'metadata_verified':list(fields),'source':remote}

    def output_page(self,slug,token):
        from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
        owner,name=slug.split('/');request=ApiListKernelSessionOutputRequest()
        request.user_name=owner;request.kernel_slug=name;self.api._set_paging(request,200,token)
        with self.api.build_kaggle_client() as client:
            response=client.kernels.kernels_api_client.list_kernel_session_output(request)
        # No signed download URLs are persisted or printed.
        return response.files or [],response.next_page_token,response.log

    def download(self,item,directory):
        import requests
        root=Path(directory).resolve();path=(root/item.file_name).resolve()
        if not path.is_relative_to(root):raise ValueError('Output filename escapes collection root')
        if path.exists():return path
        path.parent.mkdir(parents=True,exist_ok=True);partial=path.with_name(path.name+'.partial')
        with requests.get(item.url,stream=True) as response:
            response.raise_for_status()
            with partial.open('wb') as stream:
                for chunk in response.iter_content(1024*1024):stream.write(chunk)
        partial.replace(path);return path
