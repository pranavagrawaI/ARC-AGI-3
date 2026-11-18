import inspect
import pytest
from live_hook import install
from live_pilot import critic_request


@pytest.mark.parametrize('arm', ['baseline', 'critic', 'skill'])
def test_pause_cannot_commit_and_resubmission_keeps_original_gates(arm):
    committed=[]; reviews=[]; events=[]
    class Agent:
        def __init__(self): self.context={}
        def submit_actions(self, actions, rationale, predicted_path=None, *, search_id=None):
            if actions==['INVALID']: return 'REJECTED: invalid action'
            committed.extend(actions)
            return 'submitted'
    signature=inspect.signature(Agent.submit_actions)
    Agent=install(Agent,'saved notes',lambda p: reviews.append(p) or 'finding',events.append,arm)
    agent=Agent()
    agent.submit_actions(['UP'],'tentative')
    assert committed==(['UP'] if arm=='baseline' else [])
    assert len(reviews)==(1 if arm=='critic' else 0)
    assert agent.submit_actions(['INVALID'],'test')=='REJECTED: invalid action'
    assert agent.submit_actions(['LEFT'],'revised')=='submitted'
    assert committed==(['UP','LEFT'] if arm=='baseline' else ['LEFT'])
    assert inspect.signature(Agent.submit_actions)==signature


def test_critic_has_exact_actor_prefix_without_mutating_it():
    body={'messages':[{'role':'system','content':'actor system'},
                      {'role':'user','content':'observation'}],
          'tools':[{'type':'function','function':{'name':'execute_python'}}],
          'max_tokens':2048}
    reviewed=critic_request(body,{'content':'candidate'}, {'actions':['UP']})
    assert reviewed['messages'][:-1]==body['messages']
    assert reviewed['tools']==body['tools']
    assert reviewed['tool_choice']=='none'
    assert reviewed['max_tokens']==384
    assert len(body['messages'])==2


def test_real_nooa_class_accepts_declared_hook():
    import sys
    from live_pilot import RUNTIME
    sys.path.insert(0,str(RUNTIME/'chassis'))
    from solver_agent import MdArcSolverAgent
    Reviewed=install(MdArcSolverAgent,'notes',lambda p:'feedback',lambda e:None,'critic')
    assert issubclass(Reviewed,MdArcSolverAgent)
    assert inspect.signature(Reviewed.submit_actions)==inspect.signature(MdArcSolverAgent.submit_actions)


def test_isolated_critic_retains_complete_context_as_data_without_actor_tools():
    import json
    actor={'messages':[{'role':'system','content':'ACT AS THE ACTOR'}],
           'tools':[{'type':'function','function':{'name':'execute_python'}}]}
    body=critic_request(actor,{'content':'inspect'}, {},boundary=True,isolated=True)
    assert 'tools' not in body
    assert 'ACT AS THE ACTOR' not in body['messages'][0]['content']
    assert json.loads(body['messages'][1]['content'])['actor_request_as_evidence']==actor
