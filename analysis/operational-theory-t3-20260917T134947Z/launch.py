"""One user-authorized T3 invocation; frozen driver, external output relocation only."""
import datetime,json,os,shutil,sys
from pathlib import Path
root=Path.cwd();candidate=root/'analysis/operational-theory-t1r3-20260917';out=Path(Path('/tmp/arc-t3-operator/output-path').read_text());sys.path.insert(0,str(candidate))
from identity import verify,sha
import run_pair
from transport import exclusive
expected='992cc66ae2e8577f69f852f00a5330f8f0fb2269155c435f4d1d35ba2931c53c'
assert verify()==expected
review=candidate/'T2R3-REVIEW.json';assert sha(review)=='6c43c41d615db4c4ad1e207f96743bcccda68deab6418b971293403a221b8928'
resolution=json.loads((out/'MODEL-RESOLUTION.json').read_text());assert resolution['status']=='RESOLVED'
assert run_pair.metadata_check(out/'endpoints.raw')==resolution['metadata_sha256']
if not os.environ.get('OPENROUTER_API_KEY'):
 from dotenv import dotenv_values
 supplied=dotenv_values(root/'.env').get('OPENROUTER_API_KEY')
 if not supplied:raise RuntimeError('existing credential plumbing unavailable; no generation')
 os.environ['OPENROUTER_API_KEY']=supplied
 del supplied
# Main's sole output-root variable also locates selected metadata. Copy the pinned
# selected metadata byte-for-byte so relocation changes neither gate nor request.
(out/'metadata').mkdir()
shutil.copyfile(candidate/'metadata/selected.json',out/'metadata/selected.json')
assert sha(out/'metadata/selected.json')==sha(candidate/'metadata/selected.json')
run_pair.HERE=out
exclusive(out/'OUTPUT-RELOCATION.json',{'frozen_driver':str(candidate/'run_pair.py'),'driver_sha256':sha(candidate/'run_pair.py'),'runtime_override':'run_pair.HERE = external launch directory','purpose':'user requires all launch outputs outside frozen candidate','selected_metadata_sha256':sha(out/'metadata/selected.json'),'frozen_files_edited':False})
auth={'role':'user','decision':'AUTHORIZE_T3','task':'operational-theory-T1R3','authorized_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'authority':'Current user request to execute authorized TASK T3 exactly; preserved USER-TASK.txt','user_task_sha256':sha(out/'USER-TASK.txt'),'manifest_sha256':expected,'card_sha256':sha(candidate/'EXPERIMENT-CARD.md'),'review_sha256':sha(review),'metadata_sha256':resolution['metadata_sha256'],'models_metadata_sha256':resolution['models_sha256'],'model_id':'qwen/qwen3.8-27b','canonical_model_slug':resolution['resolved_model']['canonical_slug'],'endpoint':'reka/fp8','paired_launches':1,'order':['control','treatment'],'requests_per_arm':6,'completion_tokens_per_request':4096,'completion_tokens_per_arm':24576,'prompt_tokens_per_arm':600000,'pair_cost_ceiling_usd':'2','pair_wall_seconds':480,'arm_wall_seconds':240,'request_absolute_seconds':45,'broker_operations_per_arm':120,'retries':0,'fallbacks':False,'unknown_usage_policy':'stop further calls; retain unresolved exposure','launch_operator_sha256':sha(Path(__file__))}
exclusive(out/'T3-AUTHORIZATION.json',auth)
run_pair.gates(review,out/'T3-AUTHORIZATION.json',out/'endpoints.raw')
claims=root/'analysis/operational-theory-t3-claims';claims.mkdir(exist_ok=True)
exclusive(claims/(expected+'.json'),{'manifest_sha256':expected,'authorization_sha256':sha(out/'T3-AUTHORIZATION.json'),'launch_dir':str(out),'one_shot':True})
exclusive(out/'LAUNCH-START.json',{'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'authorization_sha256':sha(out/'T3-AUTHORIZATION.json'),'source_verified':verify()})
sys.argv=['run_pair.py','--execute','--review',str(review),'--authorization',str(out/'T3-AUTHORIZATION.json'),'--metadata',str(out/'endpoints.raw')]
code=0
try:run_pair.main()
except SystemExit as exc:code=exc.code or 0
finally:
 exclusive(out/'LAUNCH-END.json',{'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':code,'source_manifest_after':verify()})
print('Frozen driver exit:',code)
raise SystemExit(code)
