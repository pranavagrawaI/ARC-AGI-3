"""Real local child cleanup must complete inside the original command/startup end."""
import json,socket,subprocess,sys,tempfile,time,unittest
from pathlib import Path
import pair_runner
HERE=Path(__file__).resolve().parents[1]
class Completion(unittest.TestCase):
 def runner(self,root):
  cfg=json.loads((HERE/'packet-spec.json').read_text())['config'];cfg.update(mode='CPU_FIXTURE_ONLY',cwd=str(HERE),total_seconds=20,cleanup_seconds=3)
  return pair_runner.Runner(cfg,root/'output')
 def test_command_return_after_late_cleanup_stops_following_work(self):
  with tempfile.TemporaryDirectory() as td:
   r=self.runner(Path(td));original=r.cleanup
   def delayed(*args,**kwargs):original(*args,**kwargs);time.sleep(.25)
   try:
    r.cleanup=delayed
    with self.assertRaisesRegex(ValueError,'completion after cleanup'):r.command('CPU.phase',dict(argv=[sys.executable,'-c','pass'],seconds=.2))
    self.assertEqual(r.groups,{})
   finally:r.cleanup=original;r.cleanup(deadline=r.end);r.trace.close()
 def test_health200_cleanup_cannot_escape_original_startup_deadline(self):
  with tempfile.TemporaryDirectory() as td:
   r=self.runner(Path(td));original=r.cleanup
   server=r.start('candidate.server',dict(argv=[sys.executable,str(HERE/'tests/cpu_candidate.py'),'serve'],seconds=3),r.work_end)
   try:
    until=time.monotonic()+2
    while True:
     try:
      with socket.create_connection(('127.0.0.1',8001),timeout=.1):break
     except ConnectionRefusedError:
      if time.monotonic()>until:raise
      time.sleep(.01)
    end=time.monotonic()+.5
    def delayed(groups=None,**kwargs):
     original(groups,**kwargs)
     if groups and server.pid not in groups:time.sleep(max(0,end-time.monotonic())+.02)
    r.cleanup=delayed
    with self.assertRaisesRegex(ValueError,'startup completion after cleanup'):r.ready('candidate',r.config['candidate'],server,end)
   finally:r.cleanup=original;r.cleanup(deadline=r.end);r.trace.close()
