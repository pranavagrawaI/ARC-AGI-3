"""Execute the exact pinned native health function with CPU dependency stubs.
This establishes source branch semantics, not the unobserved remote state.
"""
import ast
import asyncio
from enum import Enum
import hashlib
import logging
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import uuid

ROOT=Path(__file__).resolve().parents[1]

class Response:
 def __init__(self,status_code):self.status_code=status_code

class NativeHealth(unittest.IsolatedAsyncioTestCase):
 def prepare(self,state='Starting',graceful=False,responds=True):
  text=(ROOT/'evidence/pinned-http_server.py').read_text()
  self.assertEqual(hashlib.sha256(text.encode()).hexdigest(),'833d5da3572305a96df3607833256ee31f6e7a03d7ea91c260a8bde076846ca6')
  classes=ast.parse((ROOT/'evidence/pinned-tokenizer_manager.py').read_text())
  status=next(x for x in classes.body if isinstance(x,ast.ClassDef) and x.name=='ServerStatus')
  ns={'Enum':Enum};exec(compile(ast.Module(body=[status],type_ignores=[]),'<pinned-ServerStatus>','exec'),ns)
  clock=[100.]
  manager=NS(server_status=getattr(ns['ServerStatus'],state),gracefully_exit=graceful,is_generation=True,last_receive_tstamp=0,rid_to_state={})
  async def generate(*args):
   if responds:manager.last_receive_tstamp=clock[0]+.01
   yield None
  manager.generate_request=generate
  async def sleep(seconds):
   clock[0]+=seconds
   await asyncio.sleep(0)
  env=NS(SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=NS(get=lambda:True))
  ns.update(Response=Response,Request=object,_global_state=NS(tokenizer_manager=manager),logger=logging.getLogger('pinned-health'),envs=env,
            asyncio=NS(create_task=asyncio.create_task,sleep=sleep),time=NS(time=lambda:clock[0],strftime=lambda *a:'stub',localtime=lambda t:t),
            HEALTH_CHECK_TIMEOUT=20,HEALTH_CHECK_RID_PREFIX='cpu-health',uuid=uuid,GenerateReqInput=lambda **kw:NS(**kw),
            get_disagg=lambda:NS(disaggregation_mode='null'),DisaggregationMode=NS(NULL=NS(value='null')))
  function=next(x for x in ast.parse(text).body if isinstance(x,ast.AsyncFunctionDef) and x.name=='health_generate');function.decorator_list=[]
  exec(compile(ast.Module(body=[function],type_ignores=[]),'<pinned-health_generate>','exec'),ns)
  return ns['health_generate'],manager,clock
 async def test_starting_is_503_before_generation(self):
  fn,manager,clock=self.prepare();response=await fn(NS(url=NS(path='/health')))
  self.assertEqual(response.status_code,503);self.assertEqual(clock[0],100.)
 async def test_graceful_exit_is_same_empty503(self):
  fn,manager,clock=self.prepare(state='Up',graceful=True);response=await fn(NS(url=NS(path='/health')))
  self.assertEqual(response.status_code,503);self.assertEqual(clock[0],100.)
 async def test_generation_heartbeat_can_return200(self):
  fn,manager,clock=self.prepare(state='Up');response=await fn(NS(url=NS(path='/health')))
  self.assertEqual(response.status_code,200);self.assertEqual(manager.server_status.name,'Up')
 async def test_no_heartbeat_times_out20_and_returns503(self):
  fn,manager,clock=self.prepare(state='Up',responds=False);response=await fn(NS(url=NS(path='/health')))
  self.assertEqual(response.status_code,503);self.assertEqual(clock[0]-100.,20.);self.assertEqual(manager.server_status.name,'UnHealthy')
 def test_source_defaults_are_enabled_and20_not_actual_remote_observations(self):
  env=(ROOT/'evidence/pinned-environ.py').read_text();http=(ROOT/'evidence/pinned-http_server.py').read_text()
  self.assertIn('SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION = EnvBool(True)',env)
  self.assertIn('HEALTH_CHECK_TIMEOUT = int(os.getenv("SGLANG_HEALTH_CHECK_TIMEOUT", 20))',http)
