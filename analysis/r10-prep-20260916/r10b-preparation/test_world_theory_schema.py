import copy
import pytest
from world_theory_schema import BUCKETS,validate

def candidate():
    theory={k:[] for k in BUCKETS}
    theory.update(version=1,parent_version=0,evidence_refs=['B/session/transitions:1#abc'],revisions=[{'kind':'add','from_ids':[],'to_ids':['C1'],'evidence_refs':['B/session/transitions:1#abc']}])
    theory['mechanics']=['C1'];theory['supported_claims']=['C1']
    return {'theory':theory,'claim_catalog':{'C1':{'claim':'This probe changed a cell','status':'supported','evidence_refs':['B/session/transitions:1#abc'],'counterevidence_refs':[],'first_observed':1,'last_updated':1}}}

def check(x,previous=None):
    return validate(x,previous=previous,evidence_catalog={'B/session/transitions:1#abc'},cutoff=1)

def test_valid_cited_envelope():
    x=candidate();assert check(x)==x

@pytest.mark.parametrize('mutation',[
    lambda x:x['claim_catalog']['C1'].update(confidence=.9),
    lambda x:x['claim_catalog']['C1'].update(evidence_refs=[]),
    lambda x:x['claim_catalog']['C1'].update(evidence_refs=['A/foreign:1#abc']),
    lambda x:x['claim_catalog']['C1'].update(last_updated=2),
    lambda x:x['theory'].update(parent_version=99),
    lambda x:x['theory'].update(entities=['uncited prose']),
    lambda x:x['theory'].update(revisions=[]),
])
def test_rejects_invalid_revision(mutation):
    x=candidate();mutation(x)
    with pytest.raises(ValueError):check(x)

def test_rejection_does_not_mutate_previous():
    old=candidate();snapshot=copy.deepcopy(old);x=copy.deepcopy(old)
    x['theory'].update(version=2,parent_version=1,revisions=[])
    x['claim_catalog']['C1']['claim']='Unsupported changed meaning'
    with pytest.raises(ValueError):check(x,old)
    assert old==snapshot
