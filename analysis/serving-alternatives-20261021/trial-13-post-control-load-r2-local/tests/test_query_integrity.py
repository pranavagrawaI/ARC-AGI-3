"""Real guarded CPU query receipts at observe/release boundaries; no GPU commands."""
import importlib.util,json,multiprocessing.process,os,socket,sys,time,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import adapters,capture
from core import shutdown
BASE=HERE.parent/'release-adapters-r6-test-local-20261023'
spec=importlib.util.spec_from_file_location('query_integrity_fixtures',BASE/'test_adapters.py')
fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)
fixtures.ROOT=Path(json.loads((BASE/'ACCEPTED-CORE.json').read_text())['directory'])

class QueryIntegrityTests(unittest.TestCase):
 setUp=fixtures.AdapterTests.setUp
 tearDown=fixtures.AdapterTests.tearDown
 persist=fixtures.AdapterTests.persist
 start=fixtures.AdapterTests.start
 build=fixtures.AdapterTests.build

 def adapter(self,stubborn=False):
  _,binding,_,_,log=self.start(stubborn=stubborn)
  with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
  return binding,self.build(binding,log,port)

 def rejection(self):
  actual=capture.command;validate=capture.validate_containment;dispatches=[];state={'rejected':False}
  def invalid_once(*args,**kw):
   if not state['rejected']:
    state['rejected']=True;raise ValueError('injected containment identity mismatch')
   return validate(*args,**kw)
  def captured(*args,**kw):
   dispatches.append(list(args[0]));receipt=actual(*args,**kw);self.results.append(dict(command=receipt));return receipt
  return dispatches,invalid_once,captured

 def assert_rejected_receipt(self):
  receipt=next(r['command'] for r in self.results if 'command' in r)
  self.assertFalse(receipt['ok']);self.assertFalse(receipt['containment']['verified'])
  self.assertTrue(receipt['cleanup_verified']);self.assertTrue(receipt['child_reaped'])
  self.assertIsNone(receipt['execution_authorized_at']);self.assertEqual(capture.RETAINED,{})

 def test_rejected_containment_clean_cleanup_stops_before_apps_and_worker(self):
  binding,adapter=self.adapter();dispatches,rejection,captured=self.rejection()
  with patch.object(capture,'validate_containment',side_effect=rejection),patch.object(capture,'command',side_effect=captured),patch.object(multiprocessing.process.BaseProcess,'start',side_effect=AssertionError('worker admitted after rejected query')) as worker:
   with self.assertRaises(adapters.ControllerQueryFailure):shutdown.observe(binding,adapter.probes(),time.monotonic()+2,1)
  self.assert_rejected_receipt();self.assertEqual(dispatches,[capture.GPU_ARGV]);worker.assert_not_called()

 def test_release_integrity_failure_has_no_future_dispatch(self):
  binding,adapter=self.adapter();dispatches,rejection,captured=self.rejection()
  with patch.object(capture,'validate_containment',side_effect=rejection),patch.object(capture,'command',side_effect=captured),patch.object(multiprocessing.process.BaseProcess,'start',side_effect=AssertionError('receipt worker admitted')) as worker:
   result=shutdown.release(binding,adapter.probes(),end=time.monotonic()+3,graceful_seconds=2,fallback_reserve=.5,probe_seconds=1,poll=.05)
  self.results.append(dict(release=result));self.assert_rejected_receipt()
  self.assertFalse(result['pass']);self.assertEqual(result['samples'],[])
  self.assertEqual(result['controller_query_failure']['kind'],'unclean_or_failed_query')
  self.assertEqual(dispatches,[capture.GPU_ARGV]);worker.assert_not_called()

 def test_successful_ineligible_resources_still_poll(self):
  binding,adapter=self.adapter(stubborn=True)
  gpu=self.root/'gpu';name,_,uuid=gpu.read_text().strip().split(',');gpu.write_text(f'{name},5000,{uuid}\n')
  actual=capture.command;dispatches=[]
  def captured(*args,**kw):
   if args[0]==capture.GPU_ARGV and len(dispatches)>=4:raise shutdown.DeadlineReached('controlled pre-execution stop after two successful samples')
   dispatches.append(list(args[0]));receipt=actual(*args,**kw)
   self.results.append(dict(command=receipt))
   if args[0]==capture.GPU_ARGV:gpu.write_text(f'{name},0,{uuid}\n')
   return receipt
  with patch.object(capture,'command',side_effect=captured):
   result=shutdown.release(binding,adapter.probes(),end=time.monotonic()+3,graceful_seconds=1.8,fallback_reserve=.7,probe_seconds=.7,poll=.05)
  self.results.append(dict(release=result))
  values=[s['probes']['gpu'] for s in result['samples'] if 'command' in s['probes']['gpu']]
  self.assertTrue(any(not v['eligible'] and v.get('used_mib')==5000 for v in values),result)
  self.assertTrue(any(v['eligible'] and v.get('used_mib')==0 for v in values),result)
  self.assertGreaterEqual(dispatches.count(capture.GPU_ARGV),2)
  self.assertTrue(all(r['command']['ok'] and r['command']['cleanup_verified'] for r in self.results if 'command' in r))

if __name__=='__main__':unittest.main()
