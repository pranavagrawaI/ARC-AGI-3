"""Bounded CPU all-corpus reference from exact serialized body bytes.

Pinned frontend tool classes and template-loader, local tokenizer/config-derived
image expansion. This is NOT native processor/tensor or candidate runtime proof.
"""
import ast,collections,copy,hashlib,json,logging,signal,sys,time
from pathlib import Path
from types import SimpleNamespace
from tokenizers import Tokenizer
import jinja2,pydantic,tokenizers
from reference import digest,check_prepared
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
sys.path.insert(0,str(BASE/'trial-5-serving-screen/prompt-audit'))
import audit
from candidate_tools import load_candidate_model

def main():
 signal.alarm(600);started=time.monotonic()
 prior=BASE/'trial-9-readiness-local';assets=BASE/'trial-3-matched-preparation/processor-assets'
 old=json.loads((prior/'catalog.json').read_text());new=copy.deepcopy(old)
 wheel=BASE.parent/'serving-headroom-v3-20261008/evidence/packages/sglang-0.5.19+gd00d88efc8d6-cp312-cp312-linux_x86_64.whl'
 assert digest(wheel.read_bytes())=='d0620216732b82b6efd4f8a78cc9b5f55343a4d3837fdfa3bcbb841b7718a225'
 source_dir=BASE/'trial-5-serving-screen/prompt-audit/source'
 source=(source_dir/'template_manager.py').read_text()
 node=next(n for c in ast.parse(source).body if isinstance(c,ast.ClassDef) for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='_load_jinja_template')
 future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
 ns={'detect_jinja_template_content_format':lambda text:'CPU_FORMAT_DIAGNOSTIC','logger':logging.getLogger('CPU')}
 exec(compile(ast.fix_missing_locations(ast.Module(body=[future,node],type_ignores=[])),'pinned-template-loader','exec'),ns)
 manager=SimpleNamespace();tm=SimpleNamespace(tokenizer=SimpleNamespace())
 ns['_load_jinja_template'](manager,tm,str(assets/'control/chat_template.jinja'))
 env=audit.processor_fit.template(assets/'control').environment
 templates={'control':env.from_string(tm.tokenizer.chat_template),'candidate':env.from_string((prior/'chat_template.normalized.jinja').read_text())}
 models={'control':audit.source_models((source_dir/'protocol.py').read_text()),'candidate':load_candidate_model()[0]}
 toks={a:Tokenizer.from_file(str(assets/a/'tokenizer.json')) for a in ('control','candidate')}
 audit.corpus.verify_corpus()
 recovery=BASE/'trial9-cutoff-local-20261023/evidence/partial-members/members/serving-wire-capacity9/output/control'
 subset=json.loads((BASE/'trial9-cutoff-local-20261023/evidence/partial-response-screen.json').read_text())['rows']
 by_key=collections.defaultdict(list)
 for row in subset:by_key[row['key']].append(row)
 delta=collections.Counter();rows=[];observed=[];max_joint=0
 new.update(schema='exact-http-byte-bound-prompt-reference-v2-local',supersedes_catalog_sha256=digest((prior/'catalog.json').read_bytes()),
            qualification='CPU_EXACT_WIRE_RENDER_REFERENCE; NATIVE_PROCESSOR_EQUIVALENCE_UNPROVED')
 for index,(key,record) in enumerate(old['records'].items()):
  raw=audit.corpus.canonical(audit.corpus.payload_for(record));body=json.loads(raw)
  assert audit.corpus.wire_hash(body)==record['wire_sha']
  assert audit.corpus.image_observations(body)==record['images']
  kwargs=dict(messages=audit.processor_fit.messages_for_template(body['messages']),add_generation_prompt=True,**body['chat_template_kwargs'])
  ids={};render={};counts={}
  for arm in ('control','candidate'):
   tools=audit.normalize_tools(body['tools'],models[arm]);text=templates[arm].render(tools=tools,**kwargs)
   render[arm]=digest(text.encode());ids[arm]=toks[arm].encode(text,add_special_tokens=False).ids
   assert ids[arm].count(toks[arm].token_to_id('<|image_pad|>'))==len(record['images'])
   counts[arm]=len(ids[arm])+399*len(record['images'])
  assert render['control']==render['candidate'] and ids['control']==ids['candidate']
  value=new['records'][key]
  value.update(superseded_reference_prompt=record['local_prompt'],local_prompt=counts['control'],prepared_body_sha256=digest(raw),
               render_sha=render['control'],token_ids_sha=digest(audit.corpus.canonical(ids['control'])))
  check_prepared(raw,value);delta[counts['control']-record['local_prompt']]+=1
  max_joint=max(max_joint,counts['control']+record['cap']);assert max_joint<=139264
  rows.append(dict(key=key,prepared_body_sha256=digest(raw),source_png_count=len(record['images']),source_png_sha256=record['images'],
                   counts=counts,render_sha256=render,token_ids_sha256={arm:digest(audit.corpus.canonical(ids[arm])) for arm in ids}))
  for case in by_key[key]:
   directory=recovery/case['rid'];transport=(directory/'wire.http').read_bytes();actual_raw=transport.split(b'\r\n\r\n',1)[1]
   check_prepared(actual_raw,value);assert actual_raw==raw
   response=json.loads((directory/'body.bin').read_text());usage=response['usage']
   assert usage['prompt_tokens']==counts['control']
   assert usage['prompt_tokens_details']['image_tokens']==400*len(record['images'])
   observed.append(dict(rid=case['rid'],key=key,actual_http_body_sha256=digest(actual_raw),prompt_tokens=usage['prompt_tokens'],image_tokens=usage['prompt_tokens_details']['image_tokens']))
  if (index+1)%250==0:print(index+1,round(time.monotonic()-started,3),flush=True)
 assert len(rows)==1369 and len(observed)==275
 assert all(old[k]==new[k] for k in ('streams','initial_holders','initial_queue','slots'))
 for key,record in old['records'].items():
  for field in ('wire_sha','images','cap','game','index','file','offset','extends_previous'):
   assert record[field]==new['records'][key][field]
 (HERE/'catalog.exact-wire-v2.json').write_text(json.dumps(new,indent=2,sort_keys=True)+'\n')
 (HERE/'exact-wire-records.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
 source_paths={'script':Path(__file__),'reference':HERE/'reference.py','old_catalog':prior/'catalog.json','protocol':source_dir/'protocol.py','template_manager':source_dir/'template_manager.py','control_template':assets/'control/chat_template.jinja','candidate_template':prior/'chat_template.normalized.jinja','old_audit':BASE/'trial-5-serving-screen/prompt-audit/audit.py','reference_renderer':BASE/'trial-3-matched-preparation/processor_fit.py','candidate_class_extractor':BASE/'trial-5-serving-screen/prompt-audit/candidate_tools.py','corpus_base':prior/'corpus_base.py'}
 for arm in ('control','candidate'):
  for name in ('tokenizer.json','tokenizer_config.json','preprocessor_config.json'):source_paths[arm+'/'+name]=assets/arm/name
 manifest=json.loads((BASE/'trial-5-serving-screen/prompt-audit/source-vllm/manifest.json').read_text())
 report=dict(status='LOCAL_EXACT_WIRE_REFERENCE_DERIVED_NOT_LAUNCH_AUTHORITY',records=1369,observed_control_subset=275,missing_completed_joins=266,
             old_reference_minus_new_not_used_as_rule=True,derived_new_minus_old_histogram=dict(delta),all_control_candidate_text_renders_and_ids_equal=True,
             immutable_order_wire_png_caps_unchanged=True,all275_exact_http_body_and_api_prompt_image_totals_match=True,max_joint_tokens=max_joint,
             source_sha256={name:digest(path.read_bytes()) for name,path in source_paths.items()},candidate_source_manifest=manifest,
             wheel_sha256=digest(wheel.read_bytes()),catalog_sha256=digest((HERE/'catalog.exact-wire-v2.json').read_bytes()),records_sha256=digest((HERE/'exact-wire-records.jsonl').read_bytes()),
             versions=dict(tokenizers=tokenizers.__version__,jinja2=jinja2.__version__,pydantic=pydantic.__version__),elapsed_seconds=time.monotonic()-started,
             limitations=['Text frontend serializer/template/tokenizer CPU proof; native pixels/processor/tensors unobserved.',
                          'Control runtime agreement observed only for275archive-order-selected responses;266completedjoins missing.',
                          'Candidate production thinkingtrue runtime and fullpaired capacity still unproved.','No old trial reclassification; original accounting gate remains unchanged.'],observed=observed)
 (HERE/'ALL-CORPUS-PROOF.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ('observed','source_sha256','candidate_source_manifest')},indent=2))
if __name__=='__main__':main()
