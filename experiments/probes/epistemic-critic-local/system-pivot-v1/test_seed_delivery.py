import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import live_system_pivot_v2 as v2


def request(messages, tools=True):
    return SimpleNamespace(full_url='https://openrouter.ai/api/v1/chat/completions',
        data=json.dumps({'messages':messages,'tools':[{}] if tools else None}).encode())


def test_actor_request_requires_all_evidence_markers(monkeypatch):
    monkeypatch.setattr(v2,'_urlopen',lambda *_a,**_k:'sent')
    text=' '.join(v2.MARKERS)
    assert v2.checked_urlopen(request([{'content':text}]))=='sent'


def test_actor_request_fails_closed_before_network(monkeypatch):
    called=[];monkeypatch.setattr(v2,'_urlopen',lambda *_a,**_k:called.append(1))
    try:v2.checked_urlopen(request([{'content':'fresh game'}]))
    except RuntimeError as e:assert 'absent before paid inference' in str(e)
    else:raise AssertionError('missing evidence accepted')
    assert called==[]


def test_critic_request_does_not_require_actor_seed(monkeypatch):
    monkeypatch.setattr(v2,'_urlopen',lambda *_a,**_k:'sent')
    assert v2.checked_urlopen(request([{'content':'review packet'}],tools=False))=='sent'
