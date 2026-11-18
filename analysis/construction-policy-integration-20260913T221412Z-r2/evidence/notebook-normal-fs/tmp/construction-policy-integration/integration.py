"""Host-only orchestration with durable, single-owner evidence."""
import copy, hashlib, json, os, time
from dataclasses import asdict
from pathlib import Path
from constructor import ConstructionLimits, ConstructionResult, run_construction
from integration_boundary import SealedBoundary, evaluate_candidate
from real_trace import load_case

class ProtocolError(RuntimeError): pass

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')

def append(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(value, sort_keys=True)+'\n'); stream.flush(); os.fsync(stream.fileno())

def create_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        raw = (json.dumps(value, indent=2, sort_keys=True)+'\n').encode()
        os.write(fd, raw); os.fsync(fd)
    finally:
        os.close(fd)

class FixtureTransport:
    """Provider-shaped deterministic responses. It never interprets tool calls."""
    def __init__(self, responses): self.responses=copy.deepcopy(responses);self.index=0;self.executed_tools=0
    def __call__(self, body):
        if self.index>=len(self.responses): raise ProtocolError('fixture responses exhausted after malformed arguments')
        response=copy.deepcopy(self.responses[self.index]);self.index+=1;return response

class IntegrationHost:
    def __init__(self, case, transport, output, limits=None):
        self.case,self.transport,self.output=case,transport,Path(output);self.requests=0;self.limits=limits
    def _claim(self):
        self.output.mkdir(parents=True, exist_ok=True)
        create_once(self.output/'RUN-OWNER.json', {'schema':'construction-run-owner-v1','pid':os.getpid()})
        append(self.output/'lifecycle.jsonl', {'event':'run_owned'})
    def next_request(self, messages):
        captured=[]
        class Stop(Exception): pass
        def transport(body): captured.append(body);raise Stop()
        try: run_construction(self.case,transport=transport,limits=ConstructionLimits(max_model_requests=1),result=ConstructionResult(status='WORKING'))
        except Stop: pass
        return captured[0]
    def _transport(self, body):
        from request_contract import validate_request
        validate_request(body); index=self.requests;self.requests+=1
        write_json(self.output/'requests'/f'{index:02d}.json',body)
        append(self.output/'lifecycle.jsonl',{'event':'request_attempt_persisted','index':index})
        started=time.monotonic()
        try:
            response=self.transport(body)
        except Exception as error:
            latency=round(time.monotonic()-started,6)
            failure={'event':'request_failed','index':index,'failure_type':type(error).__name__,'message':str(error),'latency_seconds':latency}
            write_json(self.output/'failures'/f'{index:02d}.json',failure);append(self.output/'lifecycle.jsonl',failure)
            raise
        latency=round(time.monotonic()-started,6)
        write_json(self.output/'responses'/f'{index:02d}.json',response)
        append(self.output/'lifecycle.jsonl',{'event':'response_received','index':index,'latency_seconds':latency})
        return response
    def run(self):
        self._claim()
        result=ConstructionResult(status='WORKING')
        def tool_event(event):
            append(self.output/'tool-events.jsonl',event)
            append(self.output/'lifecycle.jsonl',{'event':'tool_dispatch','tool':event['call']['function']['name'],'id':event['call']['id']})
        try:
            result=run_construction(self.case,transport=self._transport,event_sink=tool_event,progress_sink=lambda value:write_json(self.output/'progress.json',asdict(value)),result=result,limits=self.limits)
            write_json(self.output/'transcript.json',result.transcript)
            suffix=None
            if result.status=='CANDIDATE_READY':
                source=result.candidate_source
                if not isinstance(source,str): raise RuntimeError('ready candidate missing source')
                digest=hashlib.sha256(source.encode()).hexdigest();path=self.output/'frozen_candidate.py';path.write_text(source);path.chmod(0o444)
                write_json(self.output/'FROZEN.json',{'model_sha256':digest,'source_path':'frozen_candidate.py','visible_evaluation':result.evaluation,'sealed_suffix_sha256':self.case.sealed_suffix_sha256})
                append(self.output/'lifecycle.jsonl',{'event':'frozen','model_sha256':digest})
                create_once(self.output/'SUFFIX-ATTEMPT.json',{'model_sha256':digest,'state':'started'})
                append(self.output/'lifecycle.jsonl',{'event':'suffix_attempt','model_sha256':digest})
                # A new baseline-compatible 15s shared/3s per-call boundary starts
                # only after construction and freeze; constructor time cannot age it.
                sealed_full=evaluate_candidate(source,self.case,include_suffix=True,boundary=SealedBoundary())
                suffix=sealed_full['suffix']
                write_json(self.output/'sealed_suffix_full_evaluation.json',sealed_full);write_json(self.output/'sealed_suffix_evaluation.json',suffix)
                append(self.output/'lifecycle.jsonl',{'event':'suffix_assessment','model_sha256':digest})
                incumbent={'compatible_depth':1};write_json(self.output/'incumbent_suffix_evaluation.json',incumbent)
                classification='INTEGRATION_VALID_CANDIDATE_SUPPORTED' if suffix.get('compatible_depth',0)>incumbent['compatible_depth'] else 'INTEGRATION_VALID_CANDIDATE_NOT_SUPPORTED'
            elif result.status=='NEEDS_EVIDENCE': classification='INTEGRATION_VALID_NEEDS_EVIDENCE'
            elif result.status=='BUDGET_EXHAUSTED': classification='INTEGRATION_VALID_BUDGET_EXHAUSTED'
            else: classification='INTEGRATION_INVALID'
            events=[json.loads(line) for line in (self.output/'lifecycle.jsonl').read_text().splitlines()]
            suffix_attempts=sum(event.get('event')=='suffix_attempt' for event in events)
            summary={'classification':classification,'construction_status':result.status,'model_requests':result.model_requests,'inspections':result.inspections,'submissions':result.submissions,'usage':result.usage,'suffix_assessments':sum(event.get('event')=='suffix_assessment' for event in events),'suffix_attempts':suffix_attempts,'sealed_suffix_evaluation':suffix,'scope':'one fixed LS20 construction episode; no competence claim'}
            write_json(self.output/'RESULT.json',summary);append(self.output/'lifecycle.jsonl',{'event':'terminal','classification':classification})
            return summary
        except Exception as error:
            write_json(self.output/'transcript.json',result.transcript)
            summary={'classification':'INTEGRATION_INVALID','construction_status':'FAILED','failure_type':type(error).__name__,'failure':str(error),'model_requests':result.model_requests,'inspections':result.inspections,'submissions':result.submissions,'usage':result.usage,'suffix_attempts':int((self.output/'SUFFIX-ATTEMPT.json').exists()),'scope':'one fixed LS20 construction episode; no competence claim'}
            write_json(self.output/'RESULT.json',summary);append(self.output/'lifecycle.jsonl',{'event':'terminal','classification':'INTEGRATION_INVALID','failure_type':type(error).__name__})
            raise
