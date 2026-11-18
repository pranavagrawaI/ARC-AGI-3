"""Actual NEW wrapped notebook execution in isolated foreign-cwd CPU children."""
import datetime as dt,hashlib,json,os,shutil,subprocess,sys,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import packet,transport,bootstrap_patch

class NewNotebookContext(unittest.TestCase):
 def test_new_outer_foreign_context_and_preextraction_authority(self):
  spec=json.loads((HERE/'packet-spec.json').read_text());rows=[]
  with tempfile.TemporaryDirectory() as td:
   directory=Path(td);unassigned=packet.prepare(directory/'unassigned')
   fixture=dict(launch_authorized=True,serialized_assignment='LOCAL_CPU_CONTEXT_FIXTURE_NOT_ASSIGNMENT',local_only=False,
    confirmed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),total_seconds=3600,platform_hard_timeout_s=3900,
    kernel_ref=spec['metadata']['id'],sources_sha256=unassigned['sources_sha256'],config_sha256=unassigned['config_sha256'])
   packet.prepare(directory/'cpu-branch',fixture);packet.prepare(directory/'local-only',dict(fixture,local_only=True))
   cases=[('refuse_unassigned','unassigned',False,None,'unassigned'),('local_only','local-only',False,None,'local'),
    ('new_outer_clean','cpu-branch',False,None,'cpu'),('new_outer_preloaded_entry','cpu-branch',False,'release_runner','origin'),
    ('new_outer_preloaded_dependency','cpu-branch',False,'adapters','origin'),('new_outer_foreign_shadow','cpu-branch',True,None,'cpu')]
   for name,variant,shadow,poison,expect in cases:
    case=directory/name;case.mkdir()
    outer=(directory/variant/spec['metadata']['code_file']).read_bytes();raw=transport.decode(outer);source=transport.cell_source(json.loads(raw));self.assertIn(bootstrap_patch.GUARD,source)
    (case/'bootstrap.py').write_text(transport.cell_source(json.loads(outer)))
    (case/'outer.ipynb').write_bytes(outer)
    (case/'case.json').write_text(json.dumps(dict(name=name,shadow=shadow,poison=poison,confirmed_utc=fixture['confirmed_utc'],remote_root=spec['config']['cwd']),indent=2)+'\n')
    value=subprocess.run([sys.executable,'-I','-B',str(HERE/'tests/bootstrap_child.py'),str(case)],cwd=case,capture_output=True,text=True,timeout=20)
    (case/'stdout').write_text(value.stdout);(case/'stderr').write_text(value.stderr)
    proofdir=os.environ.get('PACKET_PROOF_DIR')
    if proofdir:
     target=Path(proofdir)/'new-notebook-context'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(case,target)
    self.assertEqual(value.returncode,0,value.stderr)
    row=json.loads((case/'RESULT.json').read_text());rows.append(row);self.assertEqual(row['forbidden_attempts'],[])
    if expect in ('unassigned','local'):
     self.assertEqual(row['error']['type'],'RuntimeError');self.assertIn('PREPARATION ONLY' if expect=='unassigned' else 'LOCAL ONLY fixture',row['error']['message'])
     self.assertEqual(row['extracted_files'],[]);self.assertFalse((case/'extracted').exists())
    elif expect=='origin':
     self.assertEqual(row['error']['type'],'RuntimeError');self.assertIn('package module origin mismatch:',row['error']['message']);self.assertFalse(row['wrong_module_executed']);self.assertEqual(row['boundaries'],[])
    else:
     self.assertEqual(len(row['boundaries']),1,row);self.assertFalse(row['wrong_module_executed'])
     self.assertEqual(row['result']['qualification'],'NOT_QUALIFIED');self.assertIn('CPU_ONLY_STOP_BEFORE_HARDWARE_QUERY_MODEL',row['result']['error'])
     self.assertEqual(row['boundaries'][0]['source_sha256'],unassigned['files']['release_runner.py'])
     self.assertEqual(row['result']['surviving_phase_groups'],[]);self.assertEqual(row['result']['surviving_native_group'],[]);self.assertEqual(row['result']['retained_queries'],[])
    if expect not in ('unassigned','local'):
     import base64
     values=transport.literals(source)
     for filename,record in values['PAYLOAD'].items():self.assertEqual((case/'extracted'/filename).read_bytes(),base64.b64decode(record['base64']))
   self.assertEqual(len(rows),6)
