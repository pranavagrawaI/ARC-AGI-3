import copy,unittest
import handoff
class Handoff(unittest.TestCase):
 def receipt(self):return dict(start=10.,end=14.,absolute_deadline=190.,pass_=True,error=None,server_pgid=80)
 def fresh(self):return dict(start=14.,end=15.,ipc_deadline=40.,eligible=True)
 def test_shared_work_clipped_deadline_and_late_dispatch(self):
  receipt=self.receipt();fresh=self.fresh()
  handoff.clocks(receipt,fresh,15.1,17.,200.,180)
  for end,dispatch in ((191.,17.),(190.,17.0001),(190.,14.99)):
   changed=dict(receipt,absolute_deadline=end)
   with self.assertRaises(ValueError):handoff.clocks(changed,fresh,15.1,dispatch,200.,180)
  with self.assertRaises(ValueError):handoff.clocks(receipt,fresh,15.1,17.,100.,180)
 def test_reversed_capture_and_identity(self):
  candidate=dict(start=1.,end=1501.,server_pgid='80',server_start='99',boot_id='b',session_id='s',gpu_uuid='g')
  control=dict(candidate,start=1520.,end=3020.,server_pgid='90',server_start='199')
  handoff.pair(candidate,control)
  for changed in (dict(control,start=100.),dict(control,server_pgid='80',server_start='99'),dict(control,gpu_uuid='other')):
   with self.assertRaises(ValueError):handoff.pair(candidate,changed)
