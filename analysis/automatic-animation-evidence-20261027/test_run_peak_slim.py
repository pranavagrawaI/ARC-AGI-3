import datetime as dt,json,shutil,tempfile,unittest
from pathlib import Path
import run_pair as pair
import run_peak as peak
import run_peak_slim as slim

UTC=dt.timezone.utc

class SlimBinding(unittest.TestCase):
 def test_current_slim_package_binds(self):
  self.assertEqual(len(slim.verify_slim()),64)

 def test_tampered_slim_notebook_or_body_is_refused(self):
  for target in ('kernel/m2-animation-peak-slim-20261031.ipynb','dataset/cell06.txt'):
   with tempfile.TemporaryDirectory() as d:
    copy=Path(d)/'slim';shutil.copytree(slim.SLIM,copy)
    with (copy/target).open('a') as f:f.write('\n# tamper\n')
    saved=(slim.SLIM,slim.KERNEL)
    try:
     slim.SLIM,slim.KERNEL=copy,copy/'kernel'
     with self.assertRaisesRegex(RuntimeError,'slim binding'):slim.verify_slim()
    finally:slim.SLIM,slim.KERNEL=saved

class D22Release(unittest.TestCase):
 def test_release_conditions(self):
  before=dt.datetime(2026,10,31,9,0,tzinfo=UTC);after=dt.datetime(2026,10,31,12,1,tzinfo=UTC)
  with tempfile.TemporaryDirectory() as d:
   f=Path(d);(f/'evidence').mkdir()
   self.assertIsNone(slim.d22_released(before,f))
   self.assertIsNotNone(slim.d22_released(after,f))  # window closed, never pushed
   (f/'launch-attempt.json').write_text('{}')
   self.assertIsNone(slim.d22_released(after,f))  # D22 pushed: wait for its terminal state
   (f/'terminal.json').write_text('{}')
   self.assertEqual(slim.d22_released(before,f),'D22 smoke terminal')
  with tempfile.TemporaryDirectory() as d:
   f=Path(d);(f/'evidence').mkdir();(f/'evidence/gpu-release-to-peak.json').write_text('{}')
   self.assertEqual(slim.d22_released(before,f),'D22 explicit release')

class Wiring(unittest.TestCase):
 def setUp(self):
  self.saved=(peak.RUN,pair.RUN,peak.TITLE_SLUG,pair.source_readback,peak.push_once,peak.collect_bounded,peak.PREPARED,slim.d22_released)
 def tearDown(self):
  peak.RUN,pair.RUN,peak.TITLE_SLUG,pair.source_readback,peak.push_once,peak.collect_bounded,peak.PREPARED,slim.d22_released=self.saved

 def test_configure_aims_every_seam_at_the_slim_run(self):
  slim.configure()
  self.assertEqual(peak.PREPARED,slim.KERNEL);self.assertEqual(peak.RUN,slim.RUN);self.assertEqual(pair.RUN,slim.RUN)
  self.assertIs(pair.source_readback,slim.source_readback);self.assertIs(peak.collect_bounded,slim.collect_bounded)
  meta=json.loads((peak.PREPARED/'kernel-metadata.json').read_text())
  self.assertEqual(meta['id'],slim.KERNEL_ID);self.assertIn('pranavagrawai/m2-animation-peak-cell-bodies-20261031',meta['dataset_sources'])
  self.assertEqual(peak.TITLE_SLUG,slim.KERNEL_ID)

 def test_push_refused_while_d22_holds_slot(self):
  calls=[]
  peak.push_once=lambda folder:calls.append(folder)
  slim.configure();slim.d22_released=lambda now=None,folder=None:None
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaisesRegex(RuntimeError,'D22 still owns'):peak.push_once(Path(d))
  self.assertEqual(calls,[])

 def test_push_proceeds_after_release_and_records_handoff(self):
  calls=[]
  peak.push_once=lambda folder:calls.append(folder)
  slim.configure();slim.d22_released=lambda now=None,folder=None:'D22 smoke terminal'
  with tempfile.TemporaryDirectory() as d:
   peak.push_once(Path(d));self.assertEqual(json.loads((Path(d)/'slot-handoff.json').read_text())['reason'],'D22 smoke terminal')
  self.assertEqual(len(calls),1)

if __name__=='__main__':unittest.main()
