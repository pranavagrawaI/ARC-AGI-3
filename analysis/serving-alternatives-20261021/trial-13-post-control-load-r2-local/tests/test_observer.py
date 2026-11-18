import copy,json,os,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import host_observer,release_runner
from core import shutdown
class Observer(unittest.TestCase):
 def test_live_readonly_scope_clocks_counters_and_no_peak_reset(self):
  spec=host_observer.bind('CPU_OBSERVER');row=host_observer.snapshot(spec,'boundary')
  self.assertTrue(row['raw']['current'].strip().isdigit());self.assertTrue({'oom','oom_kill'}<=row['oom_events'].keys())
  self.assertEqual(row['scope'],spec['observer_scope']);self.assertTrue(row['start']<=row['end'])
  self.assertTrue(all(row['start']<=v['start']<=v['end']<=row['end'] for v in row['file_clocks'].values()))
  self.assertIn('memory.peak cumulative and never reset; may include earlier control',row['limitations'])
 def test_unknown_observer_and_changed_input_path_fail(self):
  spec=host_observer.bind('CPU_OBSERVER')
  for mutate in (lambda v:v['controller'].update(starttime=-1),lambda v:v['paths'].update(current='/tmp/unrelated-current'),lambda v:v['observer_scope'].update(cgroup='0::/other')):
   value=copy.deepcopy(spec);mutate(value)
   with self.assertRaises((ValueError,OSError)):host_observer.snapshot(value,'bad')
 def test_diagnostic_process_unavailable_is_explicit_not_ownership(self):
  spec=host_observer.bind('CPU_OBSERVER');original=shutdown.read_process;owner=spec['controller']['pid']
  def diagnostic(pid):
   if pid!=owner:raise shutdown.UnsafeIdentity('diagnostic exec/exit fixture')
   return original(pid)
  with patch.object(shutdown,'read_process',side_effect=diagnostic):row=host_observer.snapshot(spec,'partial_process')
  self.assertTrue(any('diagnostic_unavailable' in p for p in row['processes']));self.assertTrue(row['oom_events'])
 def test_required_input_failure_is_not_zero_usage(self):
  spec=host_observer.bind('CPU_OBSERVER');original=host_observer.small
  def missing(path):
   if str(path)==spec['paths']['events']:raise OSError('missing memory.events')
   return original(path)
  with patch.object(host_observer,'small',side_effect=missing),self.assertRaisesRegex(OSError,'memory.events'):host_observer.snapshot(spec,'bad')
