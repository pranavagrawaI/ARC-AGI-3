"""Reviewer checks of actual main orchestration; no auth or provider access."""
import json,sys
from pathlib import Path
import pytest
CANDIDATE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CANDIDATE))
import run_pair as rp
import auth

def setup_parent(monkeypatch,tmp_path,approval='APPROVED'):
    monkeypatch.setattr(rp,'HERE',tmp_path)
    monkeypatch.setattr(rp,'gates',lambda *args:'review-fixture')
    p=tmp_path/'permission.json';p.write_text(json.dumps({'task':'OR-CONSTRUCTOR-REASONING-PAIR','role':'PM','decision':approval,'credential_use':'sole-key-existing-loader'}))
    a=tmp_path/'review-fixture.json';a.write_text('{}')
    monkeypatch.setattr(sys,'argv',['run_pair.py','--execute','--review',str(a),'--release',str(a),'--credential-approval',str(p)])
    return p

def test_credential_denial_precedes_any_loader(monkeypatch,tmp_path):
    setup_parent(monkeypatch,tmp_path,'PENDING')
    def forbidden(**kwargs):raise AssertionError('auth called')
    monkeypatch.setattr(auth,'load_key',forbidden)
    with pytest.raises(rp.Stop,match='credential-use'):rp.main()
    assert not (tmp_path/'live').exists()

@pytest.mark.parametrize('mode',['timeout','unknown'])
def test_parent_denies_root_freeze_and_on_after_child_artifacts(monkeypatch,tmp_path,mode):
    setup_parent(monkeypatch,tmp_path)
    monkeypatch.setattr(auth,'load_key',lambda **kwargs:('fixture-not-a-credential','fixture'))
    calls=[];real_supervise=rp.supervise
    def supervise(command,output,hard_seconds=245):
        calls.append(command);root=Path(output);child=root/'child';child.mkdir()
        # Real subprocess writes a candidate artifact then either hangs or exits.
        script='from pathlib import Path; import json,time; p=Path('+repr(str(child))+'); (p/"FROZEN.json").write_text("{}"); (p/"RESULT.json").write_text(json.dumps({"frozen":True})); time.sleep('+('30' if mode=='timeout' else '0')+')'
        if mode=='unknown':
            ledger=root.parent/'ledger.json';d=json.loads(ledger.read_text());d['unknown_exposure_usd']='0.4';rp.write(ledger,d)
        return real_supervise([sys.executable,'-B','-c',script],output,hard_seconds=.2)
    monkeypatch.setattr(rp,'supervise',supervise)
    with pytest.raises(rp.Stop,match='pair stopped'):rp.main()
    live=tmp_path/'live';assert len(calls)==1 and not (live/'on').exists()
    assert (live/'off/child/FROZEN.json').exists() and not (live/'off/FROZEN.json').exists()
    assert json.loads((live/'off/RESULT.json').read_text())['status']=='HOLD'
    assert json.loads((live/'ledger.json').read_text())['stopped']
