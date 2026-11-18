"""Read-only trial7 actual diagnostic requests vs production-setting counterfactual."""
import hashlib,json,sys,time
from pathlib import Path
from tokenizers import Tokenizer
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
sys.path.insert(0,str(BASE/'trial-5-serving-screen/prompt-audit'))
import audit
from candidate_tools import load_candidate_model

def main():
 started=time.monotonic();model,_=load_candidate_model()
 assets=BASE/'trial-3-matched-preparation/processor-assets/candidate'
 tok=Tokenizer.from_file(str(assets/'tokenizer.json'))
 template=audit.processor_fit.template(assets).environment.from_string((BASE/'trial-9-readiness-local/chat_template.normalized.jinja').read_text())
 root=BASE/'trial-7-serving-feasibility-guard/remote-readback/archive-extracted/serving-candidate-feas7/output/candidate-functional'
 records=[]
 for case in sorted(root.iterdir()):
  if not case.is_dir():continue
  raw=(case/'request.json').read_bytes();request=json.loads(raw);response=json.loads((case/'response.bin').read_text())
  wire=json.loads((case/'wire.json').read_text());assert hashlib.sha256(raw).hexdigest()==wire['body_sha256']
  images=audit.corpus.image_observations(request);tools=audit.normalize_tools(request['tools'],model)
  kwargs=dict(messages=audit.processor_fit.messages_for_template(request['messages']),tools=tools,add_generation_prompt=True,**request['chat_template_kwargs'])
  assert kwargs['enable_thinking'] is False
  computed={}
  for thinking in (False,True):
   text=template.render(**(kwargs|{'enable_thinking':thinking}));ids=tok.encode(text,add_special_tokens=False).ids
   computed[str(thinking)]=dict(prompt=len(ids)+399*len(images),render_sha256=hashlib.sha256(text.encode()).hexdigest())
  assert computed['False']['prompt']==response['usage']['prompt_tokens']
  assert response['usage']['prompt_tokens_details']['multimodal_tokens']['image']==400*len(images)
  records.append(dict(case=case.name,wire_sha256=wire['body_sha256'],raw_kwargs=request['chat_template_kwargs'],cap=request['max_tokens'],images=len(images),usage=response['usage'],cpu_thinking_counterfactual=computed))
 result=dict(scope='EXACT_DIAGNOSTIC_WIRE_CPU_RENDER_ONLY; TRUE_COUNTERFACTUAL_NOT_RUNTIME',rows=records,elapsed_seconds=time.monotonic()-started)
 (HERE/'diagnostic-probe.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
