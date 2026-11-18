import hashlib,json,tempfile,unittest
from pathlib import Path
import audit_bindings as A
class AuditBinding(unittest.TestCase):
 def test_both_snapshot_dictionaries_and_actual_import_are_enforced(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'evidence').mkdir();local=root/'local.py';reference=root/'actual_corpus.py';local.write_text('local');reference.write_text('corpus')
   external=root/'external.py';external.write_text('only snapshot references binds this file')
   snapshot=root/'snapshot.json';snapshot.write_text(json.dumps({'files':{'local.py':A.sha(local)},'references':{'actual_corpus.py':A.sha(reference),'external.py':A.sha(external)}}))
   inventory=root/'evidence/actual-audit-imports.json';inventory.write_text(json.dumps({'imports':{'corpus':{'path':str(reference),'sha256':A.sha(reference)}}}))
   binding={'files':{'actual_corpus.py':A.sha(reference),'evidence/actual-audit-imports.json':A.sha(inventory)},'snapshots':{'snapshot.json':A.sha(snapshot)}}
   (root/'audit-bindings.json').write_text(json.dumps(binding));self.assertEqual(A.verify(root)['status'],'ALL_AUDIT_FILES_REFERENCES_ACTUAL_IMPORTS_EXACT')
   for file in (local,reference,external):
    original=file.read_bytes();file.write_bytes(original+b'drift')
    with self.subTest(file=file.name),self.assertRaisesRegex(ValueError,'drift|changed'):A.verify(root)
    file.write_bytes(original)
   # Even if a file is hash-bound, a wrong actual-import pin cannot pass.
   v=json.loads(inventory.read_text());v['imports']['corpus']['sha256']='wrong';inventory.write_text(json.dumps(v));binding['files']['evidence/actual-audit-imports.json']=A.sha(inventory);(root/'audit-bindings.json').write_text(json.dumps(binding))
   with self.assertRaisesRegex(ValueError,'actual imported'):A.verify(root)
