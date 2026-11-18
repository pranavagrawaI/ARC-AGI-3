"""Output-finalization witnesses only, never mocked complete scientific qualification."""
import json,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import primary_bundle
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'collection_tools'))
import publication
class Publication(unittest.TestCase):
 def test_late_primary_receipt_cannot_return_success(self):
  with tempfile.TemporaryDirectory() as td:
   output=Path(td)/'run/output';output.mkdir(parents=True)
   (output/'result.json').write_text(json.dumps(dict(scope='PUBLICATION_FIXTURE_ONLY',pass_=False,ratio=None)))
   actual=Path.write_text
   def delayed(path,text,*args,**kwargs):
    if path.name=='primary-bundle-receipt.json':
     deadline=json.loads(text)['deadline'];time.sleep(max(0,deadline-time.monotonic())+.02)
    return actual(path,text,*args,**kwargs)
   with patch.object(Path,'write_text',delayed),self.assertRaisesRegex(TimeoutError,'receipt persistence'):
    primary_bundle.write(output,dict(run_id='CPU'),time.monotonic()+.1)
 def test_origins_or_final_write_failure_clears_ratio_and_acceptance(self):
  for filename in ('ORIGINS.json','SUPPLEMENTAL-AUDIT.json','AUDIT-COMPLETION.pending.json'):
   with tempfile.TemporaryDirectory() as td:
    destination=Path(td);report=dict(status='INVALID_INCONCLUSIVE',ratio=None,issues=[])
    def fail(path,value):
     if Path(path).name==filename:raise OSError('CPU_PUBLICATION_FAILURE')
     publication.durable_write(path,value)
    result=publication.publish(destination,report,dict(status='OUTPUT_FIXTURE_ONLY',ratio=1.3),{},write=fail)
    self.assertEqual(result['status'],'INVALID_INCONCLUSIVE');self.assertIsNone(result['ratio']);self.assertFalse(result['supplemental_complete'])
    self.assertNotEqual(publication.read_completed(destination)['status'],'PAIRED_CAPACITY_QUALIFIED')
