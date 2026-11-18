"""Separate final transport pass; all prior packets are imported without edits."""
import hashlib,importlib.util,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
PRIOR=BASE/'trial16-residual-186-local';SECOND=BASE/'trial16-residual-186-collected-20261024'
SECOND_SHA='e1889fabcf2d3f67466a2570d4f913108bed5c937f9b83aceb30ad4d83543803'
REVIEW=BASE/'coordination-20261024/trial16-final-residual-transport-local.json'
REVIEW_SHA='fbbb4c743d1b46fc3dad0b361d9da2034c78a8ec3895812541d9a497e8566405'
GAPS_SHA='655b246d431b09fe7161bf30f1e20852ca55f0a7a8230373deec96aefa34b966'
FAILED='serving-candidate-s10-16/output/candidate/serving-candidate-s10-16-20261024-candidate-8/body.bin'
CAPS=dict(list_pages=100,page_size=100,downloads=167,bytes=512*1024**2,wall_seconds=900,concurrency=1)
POLICY=dict(files=167,new_download_attempts_max=334,attempts_per_name_this_pass=2,bytes=512*1024**2,wall_seconds=900,concurrency=1,list_pages=100,page_size=100)
sys.path.insert(0,str(PRIOR));import context as prior_context
import bridge as prior_bridge
from collector import require,safe_name
APPROVED=prior_context.APPROVED;AUDIT_R2=prior_context.AUDIT_R2
FIRST=prior_context.FIRST;FIRST_SHA=prior_context.FIRST_SHA;REPEAT75=prior_context.REPEAT
ACTUAL_RESULT=prior_context.ACTUAL_RESULT

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
def verify_local():
    prior_context.verify_local()
    pins=read(HERE/'SOURCE-PINS.json')
    for name,digest in pins['files'].items():require(sha(HERE/name)==digest,'final source pin drift: '+name)
    for name,digest in pins['references'].items():require(sha(name)==digest,'final reference drift: '+name)
    require(sha(REVIEW)==REVIEW_SHA and sha(SECOND/'COLLECTION.json')==SECOND_SHA,'final authority/second receipt drift')
