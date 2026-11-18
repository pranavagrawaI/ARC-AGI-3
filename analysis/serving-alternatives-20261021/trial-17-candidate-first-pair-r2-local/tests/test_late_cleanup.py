import json,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import candidate_runner
class Late(unittest.TestCase):
    def test_late_cleanup_cannot_start_resource_worker_or_qualify(self):
            runner=object.__new__(candidate_runner.Runner);runner.server=type('Server',(),dict(pid=123))();runner.binding={};runner.work_end=time.monotonic()+5
            runner.config=dict(final_resource_seconds=.02);runner.output=Path(tempfile.mkdtemp());runner.scope={};events=[]
            runner.emit=lambda event,**fields:events.append(dict(event=event,**fields))
            def delayed(*args,**kwargs):time.sleep(.03)
            runner.cleanup=delayed
            try:
                with patch('candidate_runner.shutdown.owned',return_value={}),patch('candidate_runner.resource_observe.observe',side_effect=AssertionError('late dispatch')):
                    with self.assertRaisesRegex(ValueError,'cleanup late'):runner.finish_candidate()
                receipt=json.loads((runner.output/'candidate-release.json').read_text());self.assertFalse(receipt['pass_']);self.assertEqual(receipt['samples'],[])
            finally:
                import shutil;shutil.rmtree(runner.output)
