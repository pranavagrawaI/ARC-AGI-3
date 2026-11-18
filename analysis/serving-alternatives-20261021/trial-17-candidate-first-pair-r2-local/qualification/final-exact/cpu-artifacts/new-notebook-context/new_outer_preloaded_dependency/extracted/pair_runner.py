"""Candidate-first fixed-order package contrast; isolated whole-group/native ownership."""
import hashlib,json,os,socket,subprocess,time
from pathlib import Path
from urllib.parse import urlsplit
import candidate_runner,candidate_resources,resource_observe,capture,runtime_scope,handoff,primary_bundle,phase_tools
from native_support import NativeSupport
from phase_tools import require
from core import shutdown
HERE=Path(__file__).resolve().parent

class Runner(NativeSupport,candidate_runner.Runner):
 def __init__(self,config,output):
  super().__init__(config,output)
  self.native=None;self.native_binding=None;self.lease=None;self.expected=None;self.native_handles=[]
  self.candidate_cleanup_attempted=False;self.native_release_attempted=False;self.control_admission=None

 def bind_server(self):
  identity=super().bind_server()
  if not (self.output/'candidate-server-binding.json').exists():
   bound=dict(arm='candidate',identity=identity,session_id=self.config['run_id'],gpu_uuid=self.gpu_uuid,registered_native_binding=None)
   (self.output/'candidate-server-binding.json').write_text(json.dumps(bound,indent=2)+'\n')
   self.emit('capacity_server_bound',**bound)
  return identity

 def control_gate(self,end):
  require(self.control_admission is not None,'no reverse fresh admission')
  living=shutdown.owned(self.binding);self.check_observer()
  require(not living and not capture.RETAINED and all(g==self.observer.pid for g in self.groups),'candidate/phase/query ownership before control')
  release=json.loads((self.output/'candidate-release.json').read_text());fresh=self.control_admission
  require(candidate_resources.accepted(fresh['sample'],release['absolute_deadline']),'fresh five-probe nonhost admission')
  now=time.monotonic();require(now<end,'control.prepare phase expired')
  handoff.clocks(release,fresh['sample'],fresh['admitted_at'],now,self.work_end,self.config['final_resource_seconds'])

 def reverse_handoff(self):
  self.candidate_cleanup_attempted=True
  self.finish_candidate()
  release=json.loads((self.output/'candidate-release.json').read_text());end=release['absolute_deadline']
  self.boundary('candidate.released.before_fresh')
  require(time.monotonic()<end,'reverse handoff boundary consumed original180')
  paths={key:Path(value) for key,value in self.scope['paths'].items()}
  adapter=candidate_resources.Adapters(self.output/'query-artifacts',self.binding,self.gpu_uuid,paths,8001,end,self.scope)
  # Exactly one fresh observation, including raw verification, inside original cleanup180.
  fresh=resource_observe.observe(self.binding,adapter.probes(),end,self.config['admission']['fresh_probe_seconds'])
  self.verify_references(fresh,end)
  require(candidate_resources.accepted(fresh,end),'sole fresh reverse admission rejected')
  receipt=dict(sample=fresh,policy='FIRST_FRESH_NONHOST',no_numeric_host_admission=True,
    original140_result=fresh['probes']['cgroup']['eligible'],admitted_at=time.monotonic())
  self.control_admission=receipt
  (self.output/'control-admission.json').write_text(json.dumps(receipt,indent=2)+'\n')
  self.emit('control_admitted',**receipt)
  # No boundary/host/file enumeration between fresh and actual prepare dispatch.
  self.control_phase()

 def control_phase(self):
  a=self.config['control'];package=self.output/'control-package';package.mkdir(exist_ok=True)
  self.stage='control.prepare';self.command('control.prepare',a['prepare']);self.boundary('control.prepare.after')
  self.stage='control.identity';self.boundary('control.identity.before');self.command('control.identity',a['identity']);self.boundary('control.identity.after')
  self.runtime_contract()
  with socket.socket() as check:
   check.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);check.bind(('127.0.0.1',urlsplit(a['ready_url']).port))
  self.stage='control.startup';startup_end=self.deadline(a['server']['seconds'])
  parent=self.output/'native';parent.mkdir(exist_ok=True)
  self.lease=runtime_scope.create_native_output(parent.resolve(),self.config['run_id'])
  stdout=(self.output/'control.server.stdout').open('xb');self.native_handles.append(stdout)
  with runtime_scope.open_native_stderr(self.lease) as stderr:
   launched=time.monotonic()
   self.native=subprocess.Popen(a['server']['argv'],cwd=self.config['cwd'],env=self.native_env(startup_end),stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
  self.emit('native_spawn',pid=self.native.pid,argv=a['server']['argv'],startup_deadline=startup_end)
  registration_end=min(startup_end,launched+10);receipt=package/'control-frontend-registration.json'
  while not receipt.exists():
   require(self.native.poll() is None,'wrapper died before registration');require(time.monotonic()<registration_end,'native registration deadline');time.sleep(.01)
  runtime_scope.record_wrapper(self.native,self.lease,launched)
  (self.output/'native-lease.json').write_text(json.dumps(self.lease,indent=2)+'\n')
  self.ready('control',a,self.native,startup_end);self.boundary('control.health200');self.bind_native()
  identity=shutdown.read_process(self.native.pid)['identity']
  require(identity==self.native_binding['members'][self.native.pid],'registered control wrapper drift')
  bound=dict(arm='control',identity=identity,session_id=self.config['run_id'],gpu_uuid=self.gpu_uuid,registered_native_binding=self.native_binding)
  (self.output/'control-server-binding.json').write_text(json.dumps(bound,indent=2)+'\n');self.emit('capacity_server_bound',**bound)
  paths,_=runtime_scope.derive(Path('/proc/self/cgroup').read_text(),Path('/proc/self/mountinfo').read_text())
  scope=runtime_scope.bind(self.native_binding,paths,self.lease,self.config['run_id'])
  (self.output/'release-scope.json').write_text(json.dumps(scope,indent=2)+'\n')
  for phase in ('replay','qualify'):
   self.stage='control.'+phase;self.boundary(self.stage+'.before');spec=dict(a[phase])
   if phase=='replay':spec['env']=dict(spec.get('env',{}),SCREEN_DRAIN_SECONDS=str(a['drain_seconds']),SCREEN_SESSION_ID=self.config['run_id'],SCREEN_GPU_UUID=self.gpu_uuid,SCREEN_SERVER_PGID=str(self.native.pid),SCREEN_SERVER_START=str(identity['starttime']))
   self.command(self.stage,spec,self.native);self.boundary(self.stage+'.after')
  self.emit('arm_pass',arm='control')

 def finish_control(self):
  if self.native is None:return
  self.native_release_attempted=True
  require(self.native_binding is not None,'unbound native cannot qualify release')
  start=time.monotonic();end=min(self.work_end,start+self.config['final_resource_seconds'])
  self.emit('control_cleanup_start',cleanup_start=start,phase_deadline=end,server_pgid=self.native.pid)
  scope=json.loads((self.output/'release-scope.json').read_text());paths={key:Path(value) for key,value in scope['paths'].items()}
  adapter=self.make_adapter({'gpu':capture.GPU_ARGV,'apps':capture.APPS_ARGV},paths,Path(self.lease['created']['log_path']),scope,end)
  before=shutdown.observe(self.native_binding,adapter.probes(),end,self.config['release']['probe_seconds'])
  (self.output/'pre-release.json').write_text(json.dumps(before,indent=2)+'\n');self.verify_references(before,end)
  self.emit('release_start',deadline=end)
  spec=self.config['release']
  result=shutdown.release(self.native_binding,adapter.probes(),end=end,graceful_seconds=spec['graceful_seconds'],fallback_reserve=spec['fallback_reserve'],probe_seconds=spec['probe_seconds'],poll=spec['poll'],admission_policy='FIRST_FRESH_NONHOST')
  (self.output/'release.json').write_text(json.dumps(result,indent=2)+'\n');self.emit('release_result',**result)
  self.verify_references(result,end)
  require(result['pass'] and not result['fallback_used'],'final native graceful/resources incomplete; fallback never qualifies')
  require(not capture.RETAINED and not shutdown.owned(self.native_binding),'native/query ownership unresolved')
  require(self.native.poll() is not None and time.monotonic()<=end,'native wrapper not reaped or final180 late')
  self.boundary('control.released',finalizing=True)

 def run(self):
  result=dict(run_id=self.config['run_id'],purpose=self.config['purpose'],pass_=False,qualification='FEASIBILITY_ONLY',ratio=None,adoption=False)
  errors=[];primary=None
  try:
   self.hardware_and_queries();self.start_observer()
   self.boundary('corpus.prepare.before');self.command('corpus.prepare',self.config['corpus_prepare']);self.boundary('corpus.prepare.after')
   self.candidate_phase();self.emit('arm_pass',arm='candidate');self.reverse_handoff();self.finish_control()
  except BaseException as error:
   primary=error;result.update(error=repr(error),failed_stage=self.stage)
  finally:
   if self.server is not None and not self.candidate_cleanup_attempted:
    self.candidate_cleanup_attempted=True
    try:self.finish_candidate()
    except BaseException as error:errors.append(repr(error))
   if self.native is not None and not self.native_release_attempted:
    try:self.finish_control()
    except BaseException as error:errors.append(repr(error))
   # OOM stops workload immediately but does not suppress available final evidence.
   if self.observer_spec is not None:
    try:self.boundary('final_cleanup',finalizing=True)
    except BaseException as error:errors.append(repr(error))
    try:self.stop_observer(finalizing=True);self.measurements_finalized=True
    except BaseException as error:errors.append(repr(error))
   errors.extend(self.cleanup_native());errors.extend(self.emergency())
  result.update(cleanup_errors=errors,phase_cleanup_errors=self.cleanup_failures,surviving_phase_groups=list(self.groups),
   native_pgid=self.native.pid if self.native else None,surviving_native_group=phase_tools.group_members(self.native.pid) if self.native else [],
   retained_queries=list(capture.RETAINED),query_cleanup=self.query_cleanup,candidate_health200=self.candidate_ready,host_boundaries=self.boundary_count)
  try:
   verify_start=time.monotonic();verify_end=self.deadline(self.config['verify_seconds'])
   require(primary is None and not errors and not self.cleanup_failures and not self.groups and not capture.RETAINED and not result['surviving_native_group'],'primary error or cleanup incomplete')
   self.emit('final_verification_pass',start=verify_start,deadline=verify_end)
   require(time.monotonic()<=verify_end,'final lifecycle verification deadline')
   self.command('pair.reduce',self.config['paired_reduce'])
   pair=json.loads((self.output/'paired-result.json').read_text());result.update(qualification=pair['verdict'],ratio=pair['ratio'],pass_=not pair['issues'])
  except BaseException as error:result['verification_error']=repr(error)
  if time.monotonic()>self.work_end:result.update(pass_=False,qualification='FEASIBILITY_ONLY',ratio=None,late_work=True)
  result['elapsed_seconds']=time.monotonic()-self.started;self.emit('terminal',**result)
  (self.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');self.trace.close()
  if self.root_pidfd is not None:os.close(self.root_pidfd)
  for handle in self.handles+self.native_handles:handle.close()
  try:
   publication_end=self.deadline(self.config['primary_seconds']);publication=primary_bundle.write(self.output,self.config,publication_end);returned_at=time.monotonic()
   require(publication['start']<=publication['end']<=publication['persisted_end']<=returned_at<=publication_end<=self.work_end,'primary publication returned late')
   receipt_bytes=(self.output/'primary-bundle-receipt.json').read_bytes()
   completion=dict(schema='PRIMARY_CALLER_COMPLETION_V2',helper_persisted_end=publication['persisted_end'],returned_at=returned_at,deadline=publication_end,receipt_sha256=hashlib.sha256(receipt_bytes).hexdigest())
   (self.output/'primary-publication-completion.json').write_text(json.dumps(completion,indent=2)+'\n')
   require(time.monotonic()<=publication_end,'primary completion persistence returned late')
  except BaseException as error:
   result.update(pass_=False,qualification='FEASIBILITY_ONLY',ratio=None,primary_bundle_error=repr(error));(self.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
  return result
