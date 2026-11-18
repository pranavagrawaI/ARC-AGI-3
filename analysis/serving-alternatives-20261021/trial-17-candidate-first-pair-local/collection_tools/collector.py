"""Separate bounded current-session missing-file collection; no implicit retries."""
import hashlib,json,time
from pathlib import Path,PurePosixPath

CAPS=dict(list_pages=300,page_size=100,downloads=20000,bytes=8*1024**3,wall_seconds=7200,concurrency=1)

class Stop(ValueError):pass

def require(value,message):
    if not value:raise Stop(message)

def safe_name(name):
    require(isinstance(name,str) and bool(name) and name!='.' and '\\' not in name and '\0' not in name,'unsafe file name')
    p=PurePosixPath(name)
    require(not p.is_absolute() and '..' not in p.parts and str(p)==name,'unsafe/noncanonical file path')
    return name

def pending_body_required(http,name):
    require(http.get('rid')==PurePosixPath(name).parent.name,'pending HTTP request identity')
    require(http.get('status') in (None,200),'unexpected pending HTTP status')
    require(type(http.get('received_bytes')) is int and type(http.get('persisted_bytes')) is int
            and 0<=http['received_bytes']==http['persisted_bytes'],'pending persisted-byte schema')
    if http['status'] is None:
        require(http['received_bytes']==0 and http.get('complete') is False,'body without headers')
        return False
    lengths=[value for key,value in http.get('headers',[]) if key.lower()=='content-length']
    require(len(lengths)==1 and str(lengths[0]).isdigit() and http['received_bytes']<=int(lengths[0]),'pending HTTP200 length evidence')
    return True

class Budget:
    def __init__(self,limits,clock):
        self.limits=CAPS|dict(limits or {});self.clock=clock or time.monotonic;self.start=self.clock()
        require(all(type(v) in (int,float) and 0<v<=CAPS[k] for k,v in self.limits.items()),'invalid or expanded budget')
        require(all(type(self.limits[k]) is int for k in ('list_pages','page_size','downloads','bytes','concurrency')),'integer count/byte budgets required')
        self.end=self.start+self.limits['wall_seconds']
    def check(self):require(self.clock()<self.end,'cooperative wall deadline reached')
    def remaining(self):self.check();return self.end-self.clock()

def collect(manifest,destination,api,limits=None,clock=None,identity=None):
    """Public seam: real transport or offline fixtures; one new destination only."""
    require(isinstance(identity,dict) and set(identity)=={'kernel_id','version','source_exact','status','inner_sha256'} and identity['version']==1 and identity['source_exact'] is True,'exact reviewed current identity required')
    destination=Path(destination);destination.mkdir(exist_ok=False)
    state=dict(complete=False,addressing='CURRENT_SESSION_BEFORE_AFTER_EXACT_V1_NOT_COMPLETE_ZIP',
        list_pages=0,download_attempts=0,downloaded_bytes=0,files=[],skipped_conditionals=[],pages=[])
    budget=Budget(limits,clock)
    try:
        rows={safe_name(row['name']):dict(row) for row in manifest}
        require(len(rows)==len(manifest)<=budget.limits['downloads'],'duplicate/expanded manifest')
        for name,row in rows.items():
            require(type(row['conditional']) is bool,'conditional schema')
            if row['conditional']:
                require(name.endswith('/body.bin') and str(PurePosixPath(name).with_name('http.json')) in rows,'conditional prerequisite missing')
        state['before']=api.readback(destination/'before',budget.remaining());budget.check()
        require(all(state['before'].get(k)==v for k,v in identity.items()),'before identity/status/source mismatch')
        urls={};token='';seen_tokens=set()
        while True:
            budget.check();require(state['list_pages']<budget.limits['list_pages'],'list page cap')
            require(token not in seen_tokens,'repeated page token');seen_tokens.add(token)
            state['list_pages']+=1
            page,next_token=api.list_page(token,budget.limits['page_size']);budget.check()
            require(len(page)<=budget.limits['page_size'],'page size exceeds reviewed bound')
            matched=[]
            for name,url in page:
                safe_name(name)
                if name not in rows:continue
                require(name not in urls,'duplicate requested file across listing')
                require(isinstance(url,str) and url.startswith('https://'),'invalid download addressing')
                urls[name]=url;matched.append(name)
            state['pages'].append(dict(page=state['list_pages'],items=len(page),matched=matched,
                page_token_sha256=hashlib.sha256(token.encode()).hexdigest() if token else None))
            if not next_token:break
            require(isinstance(next_token,str),'invalid pagination token');token=next_token
        require(all(name in urls for name,row in rows.items() if not row['conditional']),'unconditional required file absent from bounded listing')
        for name,row in sorted(rows.items(),key=lambda item:(item[1].get('dependency')!='INDEPENDENT_ACTUAL_RESULT',item[0])):
            if not row['conditional']:download_one(name,row,urls[name],destination,api,budget,state)
        for name,row in sorted(rows.items()):
            if not row['conditional']:continue
            prerequisite=destination/'members'/PurePosixPath(name).with_name('http.json')
            needed=pending_body_required(json.loads(prerequisite.read_bytes()),name)
            if needed:
                require(name in urls,'required pending body absent from listing')
                download_one(name,row,urls[name],destination,api,budget,state)
            else:state['skipped_conditionals'].append(dict(name=name,reason='original HTTP status None; persisted/received0; no body required'))
        state['after']=api.readback(destination/'after',budget.remaining());budget.check()
        require(all(state['after'].get(k)==v for k,v in identity.items()),'after identity/status/source mismatch')
        state['complete']=True
    except BaseException as error:
        state['error']=dict(type=type(error).__name__,reason=str(error) if isinstance(error,Stop) else 'collection operation failed; no retry')
        if hasattr(api,'error_record') and not isinstance(error,Stop):state['safe_error']=api.error_record(error)
    state['elapsed_seconds']=budget.clock()-budget.start
    receipt=destination/'COLLECTION.json';receipt.write_text(json.dumps(state,indent=2)+'\n')
    if budget.clock()>=budget.end:
        state.update(complete=False,error=dict(type='Stop',reason='cooperative deadline reached through receipt persistence'))
        receipt.write_text(json.dumps(state,indent=2)+'\n')
    return state

def download_one(name,row,url,destination,api,budget,state):
    budget.check();require(state['download_attempts']<budget.limits['downloads'],'file attempt cap')
    target=destination/'members'/safe_name(name);target.parent.mkdir(parents=True,exist_ok=True)
    require(all(not p.is_symlink() for p in [target,*target.parents] if p!=destination.parent),'destination symlink')
    require(not target.exists(),'exclusive file exists')
    state['download_attempts']+=1
    with (destination/'attempts.jsonl').open('a') as journal:journal.write(json.dumps(dict(name=name,attempt=1))+'\n')
    count=0;digest=hashlib.sha256()
    try:
        with api.download(url,budget.remaining()) as response:
            response.raise_for_status();require(response.status_code==200,'download must be HTTP200 without redirect')
            size=response.headers.get('content-length')
            require(isinstance(size,str) and size.isdigit(),'bounded download needs explicit Content-Length')
            require(int(size)<=budget.limits['bytes']-state['downloaded_bytes'],'declared size exceeds remaining byte cap')
            with target.open('xb') as stream:
                for block in response.iter_content(chunk_size=65536):
                    budget.check();state['downloaded_bytes']+=len(block);count+=len(block)
                    require(state['downloaded_bytes']<=budget.limits['bytes'] and count<=int(size),'stream exceeds declared/total byte cap')
                    stream.write(block);digest.update(block)
            budget.check();require(count==int(size),'incomplete declared file body')
        entry=dict(name=name,origin='INDEPENDENT_CURRENT_SESSION_DOWNLOAD',bytes=count,sha256=digest.hexdigest(),complete=True,attempt=1)
        state['files'].append(entry)
        require(row.get('expected_bytes') in (None,count),'expected byte mismatch; actual file retained')
        require(row.get('expected_sha256') in (None,entry['sha256']),'expected hash mismatch; actual file retained')
    except BaseException:
        state['partial']=dict(name=name,bytes_persisted=target.stat().st_size if target.exists() else 0,
            persisted_prefix_sha256=digest.hexdigest(),complete=False)
        raise
