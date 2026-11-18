"""CPU source-bound render/tokenizer probes. No weights/engine/API/GPU."""
import ast,hashlib,json,sys,typing,logging,time,signal,collections
from pathlib import Path
from types import SimpleNamespace
import jinja2,tokenizers
from tokenizers import Tokenizer
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
sys.path.insert(0,str(BASE/'trial-5-serving-screen/prompt-audit'))
import audit

def sha(raw):return hashlib.sha256(raw).hexdigest()
def main():
 signal.alarm(180);started=time.monotonic()
 assets=BASE/'trial-3-matched-preparation/processor-assets/control'
 template=(assets/'chat_template.jinja').read_text()
 # Exact pinned loader method; only format detection is stubbed (its result
 # selects a branch, not loaded template bytes).
 source=(BASE/'trial-5-serving-screen/prompt-audit/source/template_manager.py').read_text()
 node=next(n for c in ast.parse(source).body if isinstance(c,ast.ClassDef) for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='_load_jinja_template')
 future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
 namespace={'detect_jinja_template_content_format':lambda text:'CPU_FORMAT_DIAGNOSTIC','logger':logging.getLogger('CPU')}
 exec(compile(ast.fix_missing_locations(ast.Module(body=[future,node],type_ignores=[])),'pinned-template-loader','exec'),namespace)
 manager=SimpleNamespace();tm=SimpleNamespace(tokenizer=SimpleNamespace())
 namespace['_load_jinja_template'](manager,tm,str(assets/'chat_template.jinja'))
 loaded=tm.tokenizer.chat_template
 tool=audit.source_models((BASE/'trial-5-serving-screen/prompt-audit/source/protocol.py').read_text())
 tok=Tokenizer.from_file(str(assets/'tokenizer.json'))
 env=audit.processor_fit.template(assets).environment
 cat=json.loads((BASE/'trial-9-readiness-local/catalog.json').read_text())
 root=BASE/'trial9-cutoff-local-20261023/evidence/partial-members/members/serving-wire-capacity9/output/control'
 screen=json.loads((BASE/'trial9-cutoff-local-20261023/evidence/partial-response-screen.json').read_text())
 rows=[]
 tpl=env.from_string(loaded)
 counts=collections.Counter();tool_shapes=set();observed_response_ids=set()
 for index,row in enumerate(screen['rows']):
  body=json.loads((root/row['rid']/'wire.json').read_text())['body'];record=cat['records'][row['key']]
  source_body=audit.corpus.payload_for(record)
  assert audit.corpus.canonical(source_body)==audit.corpus.canonical(body)
  kwargs=dict(messages=audit.processor_fit.messages_for_template(body['messages']),tools=audit.normalize_tools(body['tools'],tool),add_generation_prompt=True,**body['chat_template_kwargs'])
  variants={}
  for label,text in ([('audit_original',template),('pinned_loader',loaded)] if index<6 else []):
   for thinking in (True,False):
    rendered=env.from_string(text).render(**(kwargs|{'enable_thinking':thinking}))
    ids=tok.encode(rendered,add_special_tokens=False).ids
    decoded=tok.decode(ids,skip_special_tokens=False);reencoded=tok.encode(decoded,add_special_tokens=False).ids
    variants[label+str(thinking)]=dict(render_sha256=sha(rendered.encode()),render_tail=repr(rendered[-100:]),raw_text_tokens=len(ids),config_image_expanded_count=len(ids)+399*len(record['images']),
                                      final_ids=ids[-15:],final_tokens=[tok.id_to_token(i) for i in ids[-15:]],decode_reencode_ids_equal=ids==reencoded,decode_reencode_count_delta=len(reencoded)-len(ids))
  tools_comparison={}
  renders={}
  experiments=[('source_insertion_order',source_body),('actual_wire',body)]
  if index<6:experiments += [('canonical_roundtrip',json.loads(audit.corpus.canonical(source_body))),('actual_with_source_tools',body|{'tools':source_body['tools']})]
  for label,request in experiments:
   args=dict(messages=audit.processor_fit.messages_for_template(request['messages']),tools=audit.normalize_tools(request['tools'],tool),add_generation_prompt=True,**request['chat_template_kwargs'])
   rendered=tpl.render(**args);ids=tok.encode(rendered,add_special_tokens=False).ids;renders[label]=rendered
   tools_comparison[label]=dict(render_sha256=sha(rendered.encode()),expanded_count=len(ids)+399*len(record['images']))
   if index<6:tools_comparison[label]['normalized_tools']=args['tools']
  left=renders['source_insertion_order'];right=renders['actual_wire'];first=next((i for i,(a,b) in enumerate(zip(left,right)) if a!=b),None)
  observed=json.loads((root/row['rid']/'body.bin').read_text())['usage']
  assert tools_comparison['source_insertion_order']['expanded_count']==record['local_prompt']
  assert tools_comparison['actual_wire']['expanded_count']==observed['prompt_tokens']
  assert len(record['images'])==len(audit.corpus.image_observations(body))
  assert audit.corpus.image_observations(body)==record['images']
  assert observed['prompt_tokens_details']['image_tokens']==400*len(record['images'])
  counts[str(observed['prompt_tokens']-record['local_prompt'])]+=1
  tool_shapes.add(json.dumps(body['tools'],ensure_ascii=False,separators=(',',':')))
  rows.append(dict(rid=row['rid'],key=row['key'],images=len(record['images']),local_prompt=record['local_prompt'],observed_usage=observed,wire_kwargs=body['chat_template_kwargs'],variants=variants,
                   canonical_json_equal=True,tools_comparison=tools_comparison,first_render_difference=first,
                   source_difference_context=left[max(0,first-80):first+350] if first is not None else None,
                   actual_difference_context=right[max(0,first-80):first+350] if first is not None else None))
 report=dict(scope='CPU_RENDER_TEXT_TOKENIZER_ONLY_NOT_NATIVE_PROCESSOR',versions=dict(tokenizers=tokenizers.__version__,jinja2=jinja2.__version__),pinned_loader_changes_template_bytes=loaded!=template,
             joined_subset=len(rows),missing_joins=266,total_response_events=541,observed_api_minus_old_reference=dict(counts),unique_actual_tools_insertion_order=len(tool_shapes),
             observed_api_matches_actual_wire_render=len(rows),elapsed_seconds=time.monotonic()-started,
             hashes=dict(template=sha(template.encode()),loaded=sha(loaded.encode()),loader_source=sha(source.encode()),tokenizer=sha((assets/'tokenizer.json').read_bytes()),script=sha(Path(__file__).read_bytes())),rows=rows)
 (HERE/'render-probe.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
