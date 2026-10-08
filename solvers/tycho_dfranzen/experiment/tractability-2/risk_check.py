"""Same bounded AST/trace diagnostic as prior probe; no threshold or canonical-gate claim."""
import argparse
import ast
import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import trace
import unittest

p=argparse.ArgumentParser(); p.add_argument('probe',type=Path); p.add_argument('clock',type=Path); p.add_argument('output',type=Path); args=p.parse_args()
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
probe=load('measured_probe',args.probe); clock=load('measured_clock',args.clock)
class Checks(unittest.IsolatedAsyncioTestCase):
 async def test_receipt_and_guard(self):
  with tempfile.TemporaryDirectory() as d:
   class Client:
    async def acall(self,**kw):
     return SimpleNamespace(raw_response={},usage={'completion_tokens':20},tool_calls=[],content='',reasoning='')
   r=probe.RequestRuntime.__new__(probe.RequestRuntime)
   r.client=Client(); r.request={'request_body':{'parameters':{'max_tokens':12288},'model':'fixture'}}
   r.completion_tokens=0; r.unknown_reserved_tokens=0; r.cap=25000; r.final_tokens_remaining=4096; r.calls=0
   r.messages=[{'role':'user','content':'fixture'}]; r.observation_index=0; r.event_manager=SimpleNamespace(event=SimpleNamespace(prompt=''))
   r.started=__import__('time').monotonic(); r.destination=Path(d); r.projection=SimpleNamespace(calibrate=lambda n:None)
   await r.generate(tools=[])
   self.assertEqual(r.completion_tokens,20); self.assertEqual(r.calls,1)
   r.cap=1
   with self.assertRaises(probe.TokenBudgetExhausted): await r.generate(tools=[])
   self.assertEqual(r.calls,1); self.assertEqual(r.unknown_reserved_tokens,0)
 async def test_clock_success_and_error(self):
  from time import monotonic
  for error in (None,ValueError('fixture')):
   class Raw:
    async def generate(self,**kw):
     if error: raise error
     return 'receipt'
   raw=Raw(); raw.agent=SimpleNamespace(_tool_surface=SimpleNamespace(_trace_path=None)); raw.event_manager=SimpleNamespace(items=lambda:[])
   deadline=SimpleNamespace(total_seconds=10,started_at=monotonic(),config=SimpleNamespace(host_return_reserve_seconds=1))
   r=clock.DecisionRuntime(raw,deadline)
   if error:
    with self.assertRaises(ValueError): await r.generate(tools=[])
   else: self.assertEqual(await r.generate(tools=[]),'receipt')
tracer=trace.Trace(count=True,trace=False); result=tracer.runfunc(unittest.TextTestRunner().run,unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
if not result.wasSuccessful(): raise SystemExit('Invalid diagnostic')
rows=[]
for path in (args.probe,args.clock):
 for node in ast.walk(ast.parse(path.read_text())):
  if not isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) or node.name!='generate':continue
  statements={n.lineno for body in node.body for n in ast.walk(body) if isinstance(n,ast.stmt)}
  seen={line for (file,line),count in tracer.results().counts.items() if count and Path(file).resolve()==path.resolve()}
  cc=1
  for n in ast.walk(node):
   if isinstance(n,(ast.If,ast.For,ast.AsyncFor,ast.While,ast.ExceptHandler,ast.IfExp)):cc+=1
   elif isinstance(n,ast.BoolOp):cc+=len(n.values)-1
   elif isinstance(n,ast.comprehension):cc+=1+len(n.ifs)
  cov=len(statements&seen)/len(statements); rows.append({'file':str(path),'function':node.name,'decisions':cc,'coverage':cov,'crap':cc**2*(1-cov)**3+cc})
args.output.write_text(json.dumps({'measurement':'Same in-process AST statement trace and explicit decision count, fixed two-check suite, no threshold; subprocess behavior excluded','rows':rows},indent=2)+'\n')
