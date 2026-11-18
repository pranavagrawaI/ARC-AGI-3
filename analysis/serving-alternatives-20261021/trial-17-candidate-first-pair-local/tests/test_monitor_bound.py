"""No real monitor starts: fake subprocess boundary verifies one-shot timeout contract."""
import datetime as dt,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import monitor
class Monitor(unittest.TestCase):
 def run_case(self,offset):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);attempt=dt.datetime(2026,10,24,tzinfo=dt.timezone.utc);end=attempt+dt.timedelta(seconds=10800);now=[end+dt.timedelta(seconds=offset)]
   (root/'launch-attempt.json').write_text(json.dumps(dict(utc=attempt.isoformat())))
   real=dt.datetime
   class Clock(real):
    @classmethod
    def now(cls,tz=None):return now[0]
   calls=[]
   def run(command,**kwargs):
    calls.append(kwargs['timeout']);self.assertLessEqual(kwargs['timeout'],1);now[0]=end+dt.timedelta(seconds=.01)
    raise monitor.subprocess.TimeoutExpired(command,kwargs['timeout'],output=b'partial',stderr=b'')
   with patch.object(monitor,'HERE',root),patch.object(monitor,'accepted',return_value=('FIXTURE_ONLY',999,1)),patch.object(monitor.dt,'datetime',Clock),patch.object(monitor.subprocess,'run',side_effect=run),patch.object(monitor.time,'sleep',side_effect=AssertionError('no sleep past bound')):
    self.assertEqual(monitor.main(),0)
    with self.assertRaises(FileExistsError):monitor.main()
   stopped=json.loads((root/'remote-readback/monitor-stop.json').read_bytes());self.assertEqual(stopped['reason'],'DEADLINE_UNRESOLVED');self.assertFalse(stopped['restart_authorized'])
   self.assertEqual(len(calls),1 if offset<0 else 0)
 def test_last_poll_timeout_is_clipped_to_absolute_bound(self):self.run_case(-1)
 def test_no_poll_is_started_at_absolute_bound(self):self.run_case(0)
