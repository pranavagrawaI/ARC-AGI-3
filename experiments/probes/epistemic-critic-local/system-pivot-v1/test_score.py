import pytest
import json
from score import score_arm, load_arm

STOP={'seconds':60,'operationally_valid':True,'reason':'time cap'}
def event(kind,seconds,**kw):return {'kind':kind,'seconds':seconds,'warranted':True,**kw}
def corrected():
    return [event('correction',10,origin='actor'),event('retained',20,decision_id=1),event('decision',20,decision_id=1),
      event('retained',30,decision_id=2),event('decision',30,decision_id=2),event('retained',40,decision_id=3),event('decision',40,decision_id=3)]

def test_independent_actor_correction_earns_full_credit():
    result=score_arm(corrected(),STOP)
    assert result['retained_correction']=={'correction_seconds':10,'confirmed_seconds':40,'actions_to_correction':0}

def test_origin_does_not_change_score():
    a=corrected();b=corrected();b[0]['origin']='critic'
    assert score_arm(a,STOP)==score_arm(b,STOP)

def test_agreement_without_warranted_repair_gets_no_credit():
    assert score_arm([event('correction',10,warranted=False)],STOP)['first_correction_seconds'] is None

def test_short_followup_remains_unresolved():
    r=score_arm(corrected()[:-2],STOP)
    assert r['first_correction_seconds']==10 and r['retained_correction'] is None

def test_regression_invalidates_retention():
    events=corrected();events.insert(3,event('regression',25))
    assert score_arm(events,STOP)['retained_correction'] is None

def test_false_pivot_counted_even_if_later_correction_occurs():
    events=[event('false_pivot',5)]+corrected()
    assert score_arm(events,STOP)['false_pivots']==1

def test_same_action_can_retain_a_corrected_belief():
    events=corrected()
    for e in events:
        if e['kind']=='decision':e['action']='UP'
    assert score_arm(events,STOP)['followup_status']=='confirmed'

def test_time_includes_review_delay():
    a=corrected();b=[dict(e,seconds=e['seconds']+15) for e in a]
    assert score_arm(b,STOP)['retained_correction']['confirmed_seconds']-score_arm(a,STOP)['retained_correction']['confirmed_seconds']==15

def test_unordered_or_out_of_window_events_rejected():
    with pytest.raises(ValueError):score_arm([event('correction',70)],STOP)
    with pytest.raises(ValueError):score_arm([event('correction',20),event('regression',10)],STOP)

def test_transport_failure_remains_operationally_invalid():
    r=score_arm(corrected(),dict(STOP,operationally_valid=False,reason='transport'))
    assert not r['operationally_valid']

def test_three_actions_without_evidence_are_not_retained_repair():
    events=[e for e in corrected() if e['kind']!='retained']
    assert score_arm(events,STOP)['retained_correction'] is None

def test_source_binding_uses_actual_actor_timestamp_and_rejects_invented_quote(tmp_path):
    row={'call':1,'arm':'critic','purpose':'actor','arm_elapsed_seconds':12,
         'response':{'choices':[{'message':{'content':'The previous claim was unsupported.'}}]}}
    (tmp_path/'responses.jsonl').write_text(json.dumps(row)+'\n')
    (tmp_path/'RESULTS.json').write_text(json.dumps([{'arm':'critic','seconds':30,'fatal':None,'unsettled_reservation':0,'cap':'time'}]))
    grade={'kind':'correction','actor_call':1,'evidence_quote':'previous claim was unsupported',
           'warranted':True,'rationale':'Independent evaluator examined the available observations.'}
    assert load_arm(tmp_path,'critic',[grade])['first_correction_seconds']==12
    with pytest.raises(ValueError,match='unbound'):load_arm(tmp_path,'critic',[dict(grade,evidence_quote='I solved the game')])

def test_later_statement_cannot_certify_an_earlier_decision(tmp_path):
    row={'call':1,'arm':'critic','purpose':'actor','arm_elapsed_seconds':12,
         'response':{'choices':[{'message':{'content':'Corrected displacement is retained.'}}]}}
    (tmp_path/'responses.jsonl').write_text(json.dumps(row)+'\n')
    (tmp_path/'actions.jsonl').write_text(json.dumps({'arm':'critic','step':1,'elapsed_seconds':10})+'\n')
    grade={'kind':'retained','actor_call':1,'decision_id':1,'evidence_quote':'Corrected displacement',
           'warranted':True,'rationale':'Review rationale.'}
    with pytest.raises(ValueError,match='precede'):load_arm(tmp_path,'critic',[grade])
