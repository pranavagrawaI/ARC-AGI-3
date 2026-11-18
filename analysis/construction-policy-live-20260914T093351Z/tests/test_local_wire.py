import ast,base64,errno,fcntl,hashlib,json,os,re,signal,subprocess,sys,tempfile,textwrap,threading,time,types,unittest
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from unittest import mock

HERE=Path(__file__).resolve().parents[1];PACKAGE=HERE.parents[1]/'deployment/kaggle'/HERE.name
sys.path.insert(0,str(HERE))
from constructor import ConstructionResult
from integration import FixtureTransport,IntegrationHost,load_case
from integration_boundary import SealedBoundary,execute_inspection
from constructor import SYSTEM,TOOLS,construction_messages
from local_transport import LocalHTTPTransport,MAX_RESPONSE_BYTES
from request_contract import LOCAL_MODEL,logical_to_local_request,validate_local_request
from test_integration import CANDIDATE,WORK,call,response


def logical(): return IntegrationHost(load_case(),FixtureTransport([]),Path(tempfile.mkdtemp())).next_request([])
def local_response(tool_call):
 return {'id':'chatcmpl-sglang','object':'chat.completion','created':123,'model':LOCAL_MODEL,'system_fingerprint':'sglang','choices':[{'index':0,'finish_reason':'tool_calls','logprobs':None,'message':{'role':'assistant','content':None,'reasoning_content':None,'tool_calls':[{**tool_call,'index':0}]}}],'usage':{'prompt_tokens':100,'completion_tokens':30,'total_tokens':130,'completion_tokens_details':{'reasoning_tokens':0}}}

class ScriptedServer:
 def __init__(self,responses,delay=0):
  self.requests=[];outer=self
  class Handler(BaseHTTPRequestHandler):
   def do_POST(self):
    n=int(self.headers['Content-Length']);raw=self.rfile.read(n);outer.requests.append(raw)
    if delay: time.sleep(delay)
    item=responses[len(outer.requests)-1]
    if isinstance(item,tuple): status,rawout=item
    else: status,rawout=200,json.dumps(item,separators=(',',':')).encode()
    self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(rawout)));self.end_headers()
    try:self.wfile.write(rawout)
    except (BrokenPipeError,ConnectionResetError):pass
   def log_message(self,*args): pass
  self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler);self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
 @property
 def base(self): return f'http://127.0.0.1:{self.server.server_port}/v1'
 def close(self): self.server.shutdown();self.server.server_close();self.thread.join()

class LocalWireTests(unittest.TestCase):
 def exact_v15_reconcile(self):
  repo=HERE.parents[1];notebook=json.loads((repo/'deployment/kaggle/submission-20260912-v15/submission.ipynb').read_text())
  assets=''.join(notebook['cells'][1]['source'])
  match=re.search(r"Path\('/tmp/official_launcher_runtime\.py'\)\.write_bytes\(base64\.b64decode\('([^']+)'\)\)",assets)
  if match is None:self.fail('v15 embedded official launcher missing')
  source=base64.b64decode(match.group(1)).decode();tree=ast.parse(source)
  selected=[]
  for node in tree.body:
   if isinstance(node,(ast.Assign,ast.AnnAssign)) and any(getattr(target,'id',None)=='SOLVER_SESSION_SCHEMAS' for target in ([node.target] if isinstance(node,ast.AnnAssign) else node.targets)):selected.append(node)
   if isinstance(node,ast.FunctionDef) and node.name in {'expected_launcher_solver_requests','reconcile_launcher_request_count'}:selected.append(node)
  module=types.ModuleType('exact_v15_reconciliation');module.__dict__['Any']=__import__('typing').Any
  exec(compile(ast.Module(body=selected,type_ignores=[]),'<exact-v15-embedded-supervisor>','exec'),module.__dict__)
  module.reconcile_launcher_request_count._exact_source=source
  return module.reconcile_launcher_request_count
 def test_actual_v15_embedded_supervisor_contract_is_the_derived_source(self):
  repo=HERE.parents[1];notebook=json.loads((repo/'deployment/kaggle/submission-20260912-v15/submission.ipynb').read_text())
  assets=''.join(notebook['cells'][1]['source'])
  match=re.search(r"Path\('/tmp/official_launcher_runtime\.py'\)\.write_bytes\(base64\.b64decode\('([^']+)'\)\)",assets)
  if match is None:self.fail('v15 embedded official launcher missing')
  source=base64.b64decode(match.group(1))
  self.assertEqual(hashlib.sha256(source).hexdigest(),'fa7a1519595ba7f7306b2071931dc70c6784738da1f720398b0db5b1c8348afd')
  text=source.decode();self.assertIn('models = readiness["models"].get("data", [])',text)
  self.assertIn('launcher_result_path = output_dir / "official-launcher-result.json"',text)
  for option in ('"--base-url"','"--model"','"--workspace-root"','"--output"','"--timeout-s"'):self.assertIn(option,text)
  self.assertIn('if launcher_result.get("status") != "passed":',text)
 def test_adapter_is_separate_exact_and_rejects_drift(self):
  body=logical();wire=logical_to_local_request(body,local_model=LOCAL_MODEL)
  self.assertEqual(set(wire),{'model','messages','tools','tool_choice','max_tokens','reasoning','chat_template_kwargs'})
  self.assertNotIn('provider',wire);self.assertEqual(wire['messages'],body['messages']);self.assertEqual(wire['tools'],body['tools'])
  self.assertEqual(wire['reasoning'],{'enabled':False});self.assertEqual(wire['chat_template_kwargs'],{'enable_thinking':False})
  for mutation,pattern in [(lambda x:x.pop('chat_template_kwargs'),'local wire fields'),(lambda x:x.update(provider={}), 'local wire fields'),(lambda x:x.update(model='wrong'),'identity')]:
   bad=json.loads(json.dumps(wire));mutation(bad)
   with self.assertRaisesRegex(ValueError,pattern): validate_local_request(bad,expected_model=LOCAL_MODEL)
  with self.assertRaisesRegex(ValueError,'identity'): logical_to_local_request(body,local_model='wrong')
 def test_real_http_continuation_and_exact_official_launcher_cli(self):
  rs=[local_response({'id':'call_abc123','type':'function','function':{'name':'execute_python','arguments':json.dumps({'code':"import json;print(len(json.load(open('/evidence/case.json'))['visible_transitions']))"})}}),local_response({'id':'opaque-Z9','type':'function','function':{'name':'execute_python','arguments':json.dumps({'code':"import json;print(json.load(open('/evidence/case.json'))['frontier'])"})}}),local_response({'id':'submit.uuid/7','type':'function','function':{'name':'submit_candidate','arguments':json.dumps({'work_item':WORK,'candidate_source':CANDIDATE})}})]
  server=ScriptedServer(rs)
  try:
   with tempfile.TemporaryDirectory() as directory:
    root=Path(directory);output=root/'official-launcher-result.json';workspace=root/'solver-workspaces'
    env=dict(os.environ,CONSTRUCTION_ASSET_ROOT=str(HERE));env.pop('CONSTRUCTION_FIXTURE',None)
    command=[sys.executable,str(PACKAGE/'remote_entry.py'),'--base-url',server.base,'--model',LOCAL_MODEL,'--workspace-root',str(workspace),'--output',str(output),'--timeout-s','30']
    completed=subprocess.run(command,capture_output=True,text=True,timeout=60,env=env)
    self.assertEqual(completed.returncode,0,completed.stderr+completed.stdout)
    launcher=json.loads(output.read_text());self.assertEqual(launcher['status'],'passed');self.assertIsNone(launcher['gameplay_result'])
    self.assertEqual(launcher['candidate_classification'],'INTEGRATION_VALID_CANDIDATE_SUPPORTED')
    self.assertEqual(len(server.requests),3)
    wires=[json.loads(raw) for raw in server.requests]
    for raw,wire in zip(server.requests,wires):
     self.assertEqual(raw,json.dumps(wire,separators=(',',':'),ensure_ascii=False).encode());validate_local_request(wire)
     self.assertNotIn('provider',wire);self.assertNotIn('credential',raw.decode().lower());self.assertNotIn('evaluator_payload',raw.decode().lower());self.assertNotIn('/work/suffix',raw.decode().lower());self.assertNotIn('/work/verifier',raw.decode().lower())
    latest=[next(message for message in reversed(wire['messages']) if message['role']=='assistant') for wire in wires[1:]]
    self.assertEqual([message['tool_calls'][0]['id'] for message in latest],['call_abc123','opaque-Z9'])
    self.assertEqual(set(latest[0]),{'role','content','tool_calls'});self.assertNotIn('index',latest[0]['tool_calls'][0])
    evidence=workspace/'construction-evidence';self.assertEqual((evidence/'local-wire/00.body').read_bytes(),server.requests[0]);self.assertTrue((evidence/'raw-http/02.body').is_file())
    events=[json.loads(x)['event'] for x in (evidence/'lifecycle.jsonl').read_text().splitlines()];self.assertIn('request_attempt_persisted',events);self.assertIn('local_wire_persisted',events);self.assertIn('terminal',events)
    reconcile=self.exact_v15_reconcile()
    warm={'chat_completion_accesses':1,'successful_chat_completion_accesses':1}
    post={'chat_completion_accesses':1+len(server.requests),'successful_chat_completion_accesses':1+len(server.requests),'successful_freeze_gc_accesses':1,'ready_marker_count':1}
    source=reconcile._exact_source;start=source.index('            if completed.returncode != 0 or not launcher_result_path.is_file():');end=source.index('            result = {',start)
    segment=textwrap.dedent(source[start:end]);scope={'json':json,'reconcile_launcher_request_count':reconcile,'wait_for_chat_completion_count':lambda *a,**k:post,'write_json':lambda path,value:path.write_text(json.dumps(value))}
    exec(compile('def exact_unmodified_post_probe(completed,launcher_result_path,output_dir,process,warmup_facts):\n'+textwrap.indent(segment,' '),'<exact-v15-post-probe-branch>','exec'),scope)
    output_dir=root/'supervisor';output_dir.mkdir();scope['exact_unmodified_post_probe'](completed,output,output_dir,object(),warm)
    facts=json.loads((output_dir/'request-count-facts.json').read_text())
    self.assertTrue(facts['whole_model_request_count_verified']);self.assertEqual(facts['expected_launcher_solver_requests'],3)
  finally: server.close()
 def test_http_failure_malformed_timeout_and_oversize_are_durable_no_retry(self):
  scenarios=[([(503,b'no')],1,'HTTP Error 503'),([(200,b'{')],1,'malformed'),([(200,b'x'*(MAX_RESPONSE_BYTES+1))],1,'byte cap')]
  for responses,timeout,pattern in scenarios:
   server=ScriptedServer(responses)
   try:
    with tempfile.TemporaryDirectory() as directory:
     t=LocalHTTPTransport(base_url=server.base,model=LOCAL_MODEL,output=Path(directory),timeout_s=timeout)
     with self.assertRaisesRegex(Exception,pattern): t(logical())
     self.assertEqual(len(server.requests),1);self.assertTrue(any((Path(directory)/'http-failures').iterdir()))
   finally: server.close()
  server=ScriptedServer([local_response(call('x','request_evidence',{'obstacle':'x','needed_observation':'y','decision_relevance':'z'}))],delay=.2)
  try:
   with tempfile.TemporaryDirectory() as directory:
    t=LocalHTTPTransport(base_url=server.base,model=LOCAL_MODEL,output=Path(directory),timeout_s=.02)
    with self.assertRaises(Exception): t(logical())
    self.assertEqual(len(server.requests),1);self.assertTrue(any((Path(directory)/'http-failures').iterdir()))
  finally: server.close()
 def test_host_transport_failure_terminal_and_single_owner(self):
  class Boom:
   def __init__(self):self.calls=0
   def __call__(self,body):self.calls+=1;raise OSError('offline')
  with tempfile.TemporaryDirectory() as directory:
   out=Path(directory)/'e';boom=Boom();host=IntegrationHost(load_case(),boom,out)
   with self.assertRaises(OSError):host.run()
   self.assertEqual(boom.calls,1);self.assertEqual(json.loads((out/'RESULT.json').read_text())['failure_type'],'OSError')
   with self.assertRaises(FileExistsError):host.run()
   self.assertEqual(boom.calls,1)
 def test_boundary_baseline_budgets_args_and_advertised_inspection_path(self):
  boundary=SealedBoundary();self.assertGreater(boundary.deadline-time.monotonic(),14);self.assertEqual(boundary.call_seconds,3);self.assertEqual(boundary.inspection_seconds,8)
  value=boundary.invoke('def initialize(o): return len(o["x"])\ndef predict(s,a):return []\ndef assimilate(s,o,v):return s\n','initialize',[{'x':'a'*17000}])
  self.assertEqual(value,17000)
  inspected=execute_inspection("import json;print(json.load(open('/evidence/case.json'))['frontier'])",load_case());self.assertEqual(inspected['stdout'].strip(),'5')
  self.assertEqual(inspected,{'status':'completed','returncode':0,'stdout':'5\n','stderr':'','output_truncated':False})
  packet=json.loads(construction_messages(load_case(),None)[1]['content'])
  self.assertEqual(packet['visible_case']['full_evidence']['path'],'/evidence/case.json')
  self.assertIn('/evidence/case.json',SYSTEM);self.assertIn('/evidence/case.json',TOOLS[0]['function']['description'])
  advertised=json.dumps([SYSTEM,TOOLS,packet],sort_keys=True);self.assertNotIn('/work/prefix.json',advertised)
 def test_inspection_pipe_fds_close_when_launch_raises_exception_or_interrupt(self):
  real_pipe,real_close=os.pipe,os.close
  for raised in (OSError('process launch failed'),KeyboardInterrupt()):
   with self.subTest(raised=type(raised).__name__):
    acquired=[];closed=[];owned=set()
    def tracked_pipe():
     pair=real_pipe();acquired.extend(pair);owned.update(pair);return pair
    def tracked_close(fd):
     if fd in owned:closed.append(fd);owned.remove(fd)
     real_close(fd)
    def fake_rootfs(root,*args,**kwargs):
     evidence=root/'evidence';evidence.mkdir(parents=True);(evidence/'case.json').write_text('{}');return {}
    with mock.patch('integration_boundary.cb.rootfs',side_effect=fake_rootfs),mock.patch('integration_boundary.cb.validate_tree'),mock.patch('integration_boundary.cb.bounded',side_effect=raised),mock.patch('integration_boundary.os.pipe',side_effect=tracked_pipe),mock.patch('integration_boundary.os.close',side_effect=tracked_close):
     with self.assertRaises(type(raised)):
      SealedBoundary().inspect('print(1)',{})
    self.assertEqual(len(acquired),2)
    self.assertEqual(len(closed),2)
    self.assertCountEqual(closed,acquired)
    for fd in acquired:
     with self.assertRaises(OSError) as caught:fcntl.fcntl(fd,fcntl.F_GETFD)
     self.assertEqual(caught.exception.errno,errno.EBADF)
 def test_actual_sealed_inspection_preserves_stderr_and_rejects_aggregate_output_flood(self):
  receipt=execute_inspection("import sys;print('visible');print('diagnostic',file=sys.stderr)",load_case())
  self.assertEqual(receipt['status'],'completed');self.assertEqual(receipt['returncode'],0)
  self.assertEqual(receipt['stdout'],'visible\n');self.assertEqual(receipt['stderr'],'diagnostic\n');self.assertFalse(receipt['output_truncated'])
  failed=execute_inspection("import sys;print('bad',file=sys.stderr);raise ValueError('no')",load_case());self.assertEqual(failed['status'],'failed');self.assertNotEqual(failed['returncode'],0);self.assertIn('bad',failed['stderr'])
  truncated=execute_inspection("print('x'*17000,end='')",load_case());self.assertEqual(len(truncated['stdout']),16000);self.assertTrue(truncated['output_truncated'])
  with self.assertRaisesRegex(RuntimeError,'output limit'):
   execute_inspection("import os;os.write(1,b'x'*21000);os.write(2,b'diagnostic')",load_case())
 def test_remote_transport_failure_reports_infrastructure_separately(self):
  server=ScriptedServer([(503,b'no')])
  try:
   with tempfile.TemporaryDirectory() as directory:
    root=Path(directory);output=root/'official-launcher-result.json';env=dict(os.environ,CONSTRUCTION_ASSET_ROOT=str(HERE));env.pop('CONSTRUCTION_FIXTURE',None)
    completed=subprocess.run([sys.executable,str(PACKAGE/'remote_entry.py'),'--base-url',server.base,'--model',LOCAL_MODEL,'--workspace-root',str(root/'workspaces'),'--output',str(output),'--timeout-s','30'],capture_output=True,text=True,timeout=60,env=env)
    self.assertEqual(completed.returncode,1);self.assertEqual(len(server.requests),1)
    launcher=json.loads(output.read_text());self.assertEqual(launcher['status'],'failed');self.assertIsNone(launcher['candidate_classification']);self.assertIsNone(launcher['gameplay_result']);self.assertNotIn('solver_sessions',launcher)
  finally:server.close()
 def test_fresh_suffix_boundary_after_delayed_construction_and_irreversible_attempt(self):
  ready=ConstructionResult(status='CANDIDATE_READY',candidate_source=CANDIDATE,evaluation={'visible':{'status':'passed'},'coverage':{'missing_required':[]}})
  seen=[]
  def fake_run(*args,**kwargs): time.sleep(.03);return ready
  def fake_eval(source,case,include_suffix,boundary): seen.append(boundary.deadline-time.monotonic());return {'suffix':{'compatible_depth':2},'rpc_methods':[]}
  with tempfile.TemporaryDirectory() as directory,mock.patch('integration.run_construction',fake_run),mock.patch('integration.evaluate_candidate',fake_eval):
   out=Path(directory)/'e';result=IntegrationHost(load_case(),FixtureTransport([]),out).run();self.assertGreater(seen[0],14);self.assertEqual(result['suffix_attempts'],1);self.assertTrue((out/'SUFFIX-ATTEMPT.json').exists())
 def test_caught_interrupt_after_suffix_attempt_is_terminal_but_sigkill_is_marker_only(self):
  ready=ConstructionResult(status='CANDIDATE_READY',candidate_source=CANDIDATE,evaluation={'visible':{'status':'passed'},'coverage':{'missing_required':[]}})
  with tempfile.TemporaryDirectory() as directory,mock.patch('integration.run_construction',return_value=ready),mock.patch('integration.evaluate_candidate',side_effect=KeyboardInterrupt):
   out=Path(directory)/'caught'
   with self.assertRaises(KeyboardInterrupt):IntegrationHost(load_case(),FixtureTransport([]),out).run()
   self.assertTrue((out/'SUFFIX-ATTEMPT.json').is_file());result=json.loads((out/'RESULT.json').read_text());self.assertEqual(result['failure_type'],'KeyboardInterrupt')
   terminal=json.loads((out/'lifecycle.jsonl').read_text().splitlines()[-1]);self.assertEqual(terminal['event'],'terminal')
  with tempfile.TemporaryDirectory() as directory:
   out=Path(directory)/'killed'
   script="""import sys,time\nfrom pathlib import Path\nsys.path.insert(0,sys.argv[1])\nimport integration\nfrom constructor import ConstructionResult\nsource=%r\nintegration.run_construction=lambda *a,**k:ConstructionResult(status='CANDIDATE_READY',candidate_source=source,evaluation={})\nintegration.evaluate_candidate=lambda *a,**k:time.sleep(30)\nintegration.IntegrationHost(integration.load_case(),integration.FixtureTransport([]),Path(sys.argv[2])).run()\n""" % CANDIDATE
   child=subprocess.Popen([sys.executable,'-c',script,str(HERE),str(out)])
   deadline=time.monotonic()+10
   while not (out/'SUFFIX-ATTEMPT.json').is_file() and child.poll() is None and time.monotonic()<deadline:time.sleep(.02)
   self.assertTrue((out/'SUFFIX-ATTEMPT.json').is_file());os.kill(child.pid,signal.SIGKILL);self.assertEqual(child.wait(timeout=5),-signal.SIGKILL)
   self.assertFalse((out/'RESULT.json').exists());events=[json.loads(x)['event'] for x in (out/'lifecycle.jsonl').read_text().splitlines()];self.assertNotIn('terminal',events)

if __name__=='__main__':unittest.main()
