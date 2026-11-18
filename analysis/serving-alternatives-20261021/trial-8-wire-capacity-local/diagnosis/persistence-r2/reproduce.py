"""CPU loopback + analytical reducer witness; output must be a NEW directory."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch

def main():
    parser=argparse.ArgumentParser();parser.add_argument('source');parser.add_argument('output');a=parser.parse_args()
    source=Path(a.source).resolve();out=Path(a.output).resolve();out.mkdir(exist_ok=False,parents=True)
    sys.path.insert(0,str(source));sys.path.insert(0,str(source/'tests'))
    import test_replay,test_reducer,replay,reduce_api,catalog
    class Body:
        def __init__(self,stream):self.stream=stream
        def __enter__(self):self.stream.__enter__();return self
        def __exit__(self,*args):return self.stream.__exit__(*args)
        def close(self):return self.stream.close()
        def write(self,data):time.sleep(.3);raise OSError('persistence failed after final body read')
    original=Path.open
    def opened(path,*args,**kwargs):
        stream=original(path,*args,**kwargs)
        return Body(stream) if path.name=='body.bin' and args and args[0]=='xb' else stream
    case=test_replay.ReplayTests()
    try:
        with patch.object(Path,'open',opened):root,server,obj,events,ok,elapsed=case.run_fixture('ok',(.05,.05))
        shutil.copytree(root/'out',out/'raw-replay')
        result=dict(scope='CPU_ONLY_LOOPBACK_AND_ANALYTICAL_FIXTURE_NOT_SCIENTIFIC_RATE',source=str(source),
                    source_sha256={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in ('replay.py','reduce_api.py')},
                    run_ok=ok,failed=obj.failed,elapsed_s=elapsed,replay_end=obj.end,
                    responses=sum(e['kind']=='response' for e in events),censored_pending=sum(e['kind']=='cutoff_pending' for e in events),
                    http=[json.loads(p.read_text()) for p in (root/'out').glob('*/http.json')])
    finally:case.doCleanups()
    pair=out/'pair';pair.mkdir();cat,_=test_reducer.fixture(pair,'control',0,[100,700,1000,1300])
    test_reducer.fixture(pair,'candidate',2000,[100,700,800,1000,1100,1300,1400])
    def compare():
        with patch.object(catalog,'load',return_value=cat),patch.object(reduce_api.provenance,'check',return_value={'CPU_FIXTURE_ONLY':True}):
            value=reduce_api.compare(pair)
        return {**{k:value[k] for k in ('verdict','ratio','issues')},'control_valid':value['control']['valid'],'control_issues':value['control']['issues']}
    result['ordinary_cutoff_pair']=compare()
    events=list(map(json.loads,(pair/'control/events.jsonl').read_text().splitlines()))
    rid=next(r['rid'] for r in events if r['kind']=='cutoff_pending');path=pair/'control'/rid/'http.json';receipt=json.loads(path.read_text())
    receipt.update(received_bytes=228,headers=[['Content-Length','228']],complete=False,error="OSError('persistence failed after final body read')")
    path.write_text(json.dumps(receipt));(path.parent/'body.bin').write_bytes(b'')
    result['disk_loss_pair']=compare()
    (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    inventory={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()}
    (out/'INVENTORY.json').write_text(json.dumps(inventory,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='http'},indent=2))

if __name__=='__main__':main()
