"""Separate residual-pass context. No network or approved-source mutation at import."""
import hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
APPROVED=BASE/'trial16-missing-raw-local'
AUDIT_R2=BASE/'trial16-supplemental-audit-r2-local'
FIRST=BASE/'trial16-missing-raw-collected-20261024'
REVIEW=BASE/'coordination-20261024/trial16-residual-186-local-preparation.json'
REVIEW_SHA='2d2fabe09a56cdffb682f75ef4514b8aad66dd15a93844dbdf7aabe8b4f79e37'
FIRST_SHA='4b3a6d78747792b32f5b47ab156a0ea2bf7140f108c16d18f829681c8de83a5e'
REPEAT='serving-candidate-s10-16/output/candidate/serving-candidate-s10-16-20261024-candidate-75/http.json'
ACTUAL_RESULT='serving-candidate-s10-16/output/result.json'
CAPS=dict(list_pages=100,page_size=100,downloads=186,bytes=512*1024**2,wall_seconds=600,concurrency=1)
sys.path.insert(0,str(APPROVED))
from collector import require,safe_name

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def read(path):return json.loads(Path(path).read_bytes())

def verify_local():
    import importlib.util
    spec=importlib.util.spec_from_file_location('approved_collection_run',APPROVED/'run.py')
    approved_run=importlib.util.module_from_spec(spec);spec.loader.exec_module(approved_run);approved_run.verify_local()
    for folder in (AUDIT_R2,HERE):
        pins=read(folder/'SOURCE-PINS.json')
        for name,digest in pins['files'].items():require(sha(folder/name)==digest,'source pin drift: '+name)
        for name,digest in pins['references'].items():require(sha(name)==digest,'reference pin drift: '+name)
    require(sha(REVIEW)==REVIEW_SHA and sha(FIRST/'COLLECTION.json')==FIRST_SHA,'residual/first authority drift')
