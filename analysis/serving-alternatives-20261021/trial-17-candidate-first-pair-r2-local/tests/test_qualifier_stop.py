"""Actual qualifier CLI on retained raw HTTP with one invalid accounting field."""
import json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]
RAW=HERE.parent/'trial16-three-pass-audit-result-20261024/evidence/serving-candidate-s10-16/output/candidate'
class Qualifier(unittest.TestCase):
 def test_invalid_complete_image_accounting_returns_failure_before_handoff(self):
  events=[json.loads(line) for line in (RAW/'events.jsonl').read_bytes().splitlines()];response=next(row for row in events if row['kind']=='response')
  rid=response['rid']
  with tempfile.TemporaryDirectory() as td:
   output=Path(td)/'output';folder=output/'candidate';folder.mkdir(parents=True)
   for entry in RAW.iterdir():
    if entry.name==rid:continue
    (folder/entry.name).symlink_to(entry.resolve(),target_is_directory=entry.is_dir())
   (folder/rid).mkdir()
   for entry in (RAW/rid).iterdir():
    if entry.name=='body.bin':continue
    (folder/rid/entry.name).symlink_to(entry.resolve())
   body=json.loads((RAW/rid/'body.bin').read_bytes());image=body['usage']['prompt_tokens_details']['multimodal_tokens']['image'];original=(RAW/rid/'body.bin').read_bytes()
   from catalog import canonical
   body['usage']['prompt_tokens_details']['multimodal_tokens']['image']=image+1
   changed=canonical(body);self.assertEqual(len(changed),len(original),'preserve actual HTTP Content-Length to isolate accounting gate')
   (folder/rid/'body.bin').write_bytes(changed)
   completed=subprocess.run([sys.executable,str(HERE/'candidate_reduce.py'),str(output)],capture_output=True,text=True,timeout=30)
   value=json.loads((output/'candidate-arm.json').read_bytes())
   self.assertFalse(value['valid']);self.assertIn('image-token total',' '.join(value['issues']))
   self.assertEqual(completed.returncode,1,'invalid accounting must abort parent command and second load')
