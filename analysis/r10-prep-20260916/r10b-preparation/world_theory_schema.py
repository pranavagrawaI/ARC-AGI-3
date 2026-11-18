"""R10b prepared schema contract. Validates structure/provenance, never entailment."""
from __future__ import annotations

PAYLOAD = {'claim','status','evidence_refs','counterevidence_refs','first_observed','last_updated'}
BUCKETS = {'entities','mechanics','goals_or_objectives','persistent_state','supported_claims','competing_hypotheses','contradictions','unknowns'}
STATUSES = {'supported_claims':'supported','competing_hypotheses':'tentative','contradictions':'contradicted','unknowns':'unknown'}
TOP = BUCKETS | {'version','parent_version','revisions','evidence_refs'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(envelope, *, previous, evidence_catalog, cutoff):
    """Accepted snapshots use host-issued IDs; proposed new IDs require allocation first.

    evidence_catalog is the packet-builder's same-arm/session cutoff catalog,
    not a model-supplied list. Its own provenance needs integration verification.
    """
    require(type(envelope) is dict and set(envelope)=={'theory','claim_catalog'}, 'envelope fields')
    theory, claims = envelope['theory'], envelope['claim_catalog']
    require(type(theory) is dict and set(theory)==TOP, 'theory fields')
    require(type(claims) is dict and all(isinstance(k,str) and k.startswith('C') and k[1:].isdigit() for k in claims), 'claim IDs')
    parent=previous['theory']['version'] if previous else 0
    require(type(theory['version']) is int and theory['version']==parent+1, 'version')
    require(type(theory['parent_version']) is int and theory['parent_version']==parent, 'parent')
    require(type(cutoff) is int and cutoff>=0, 'cutoff')
    def refs(values, nonempty=False):
        require(type(values) is list and (bool(values) or not nonempty), 'evidence list')
        require(all(isinstance(r,str) and r in evidence_catalog for r in values), 'unknown evidence ref')
        require(len(values)==len(set(values)), 'duplicate evidence ref')
    refs(theory['evidence_refs'])
    for cid, claim in claims.items():
        require(type(claim) is dict and set(claim)==PAYLOAD, 'exact claim payload')
        require(isinstance(claim['claim'],str) and bool(claim['claim'].strip()), 'claim text')
        require(claim['status'] in STATUSES.values(), 'status')
        refs(claim['evidence_refs'],True); refs(claim['counterevidence_refs'])
        require(set(claim['evidence_refs']+claim['counterevidence_refs'])<=set(theory['evidence_refs']), 'top-level citation coverage')
        first,last=claim['first_observed'],claim['last_updated']
        require(type(first) is int and type(last) is int and 0<=first<=last<=cutoff, 'step range')
    used=set()
    for bucket in BUCKETS:
        ids=theory[bucket]
        require(type(ids) is list and all(isinstance(cid,str) and cid in claims for cid in ids), 'bucket claim refs only')
        require(len(ids)==len(set(ids)), 'duplicate bucket ID')
        if bucket in STATUSES:
            require(all(claims[cid]['status']==STATUSES[bucket] for cid in ids), 'status bucket mismatch')
        used.update(ids)
    require(used==set(claims), 'unplaced claim')
    for status_bucket,status in STATUSES.items():
        require(set(theory[status_bucket])=={cid for cid,c in claims.items() if c['status']==status}, 'status coverage')
    revisions=theory['revisions']
    require(type(revisions) is list, 'revisions')
    old=previous['claim_catalog'] if previous else {}
    changed={cid for cid,c in claims.items() if old.get(cid)!=c}
    require(set(old)<=set(claims), 'silent deletion')
    revised=set()
    for revision in revisions:
        require(type(revision) is dict and set(revision)=={'kind','from_ids','to_ids','evidence_refs'}, 'revision fields')
        require(revision['kind'] in {'add','update','retract','replace','split','merge'}, 'revision kind')
        before,after=revision['from_ids'],revision['to_ids']
        require(type(before) is list and type(after) is list and bool(after), 'revision IDs')
        require(all(isinstance(c,str) and c in old for c in before) and all(isinstance(c,str) and c in claims for c in after), 'revision lineage refs')
        refs(revision['evidence_refs'],True)
        if revision['kind']=='add':
            require(not before and all(c not in old for c in after), 'add lineage')
        if revision['kind'] in {'update','retract'}:
            require(len(before)==1 and before==after, 'stable identity update')
        if revision['kind']=='retract':
            require(claims[after[0]]['status']=='contradicted' and bool(claims[after[0]]['counterevidence_refs']), 'retraction counterevidence')
        if revision['kind'] in {'replace','split','merge'}:
            require(bool(before) and not set(before)&set(after) and all(c not in old for c in after), 'changed meaning IDs')
            require(all(claims[c]['status']=='contradicted' and claims[c]['counterevidence_refs'] for c in before), 'retained superseded claims')
        revised.update(after); revised.update(before)
    require(changed<=revised, 'unrecorded revision')
    for cid in set(old)&set(claims):
        require(claims[cid]['first_observed']==old[cid]['first_observed'], 'first observation drift')
        require(claims[cid]['last_updated']>=old[cid]['last_updated'], 'update went backwards')
    return envelope
