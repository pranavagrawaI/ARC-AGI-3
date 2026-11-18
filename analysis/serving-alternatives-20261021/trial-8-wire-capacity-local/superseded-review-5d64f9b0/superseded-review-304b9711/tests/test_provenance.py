import copy
import json
from pathlib import Path
import tempfile
import unittest
import provenance

class Provenance(unittest.TestCase):
    def test_bound_receipts_and_drift_or_cleanup_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'output';root.mkdir()
            def put(path,value):path.parent.mkdir(exist_ok=True,parents=True);path.write_text(json.dumps(value))
            spec=provenance.read(provenance.HERE/'packet-spec.json');config=spec['config'];put(root.parent/'config.json',config)
            contract=provenance.read(provenance.HERE/'capacity-contract.json')
            capture=dict(session_id=config['run_id'],boot_id='CPU_ONLY',gpu_uuid='GPU-CPU',server_pgid='100',server_start='10')
            a={'capture':dict(capture)};b={'capture':dict(capture,server_pgid='200',server_start='20')}
            trace=[{'event':'start','config_sha256':contract['config_sha256'],'run_id':config['run_id'],'boot_id':'CPU_ONLY'},
                   {'event':'hardware_pass','uuid':'GPU-CPU'},
                   {'event':'spawn','phase':'control.server','pgid':100}, {'event':'arm_pass','arm':'control'},
                   {'event':'cleanup','pgid':100,'remaining':[]}, {'event':'spawn','phase':'candidate.prepare','pgid':199},
                   {'event':'spawn','phase':'candidate.server','pgid':200}, {'event':'arm_pass','arm':'candidate'},
                   {'event':'cleanup','pgid':200,'remaining':[]}]
            for i,r in enumerate(trace):r['seq']=i
            (root/'supervisor.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in trace))
            for label in ('control','candidate'):put(root/(label+'-release.json'),{'pass':True,'port8001_free':True,'observed':{'uuid':'GPU-CPU'}})
            plan=provenance.read(provenance.HERE/'control-plan.json');p=root/'control-package'
            put(p/'control-source-identity.json',{'wheel_sha256':plan['wheel']['sha256'],'lock_sha256':plan['wheel']['requirements_lock_sha256'],'installed_source_sha256':plan['installed_source_sha256']})
            put(p/'control-model-identity.json',{'target_version':plan['target_version'],'draft_version':plan['draft_version'],'inventory':{'target':{'assets_sha256':plan['assets_sha256']}}})
            put(p/'control-prepared.json',{'argv':provenance.read(provenance.HERE/'control-argv.reference.json')})
            plan=provenance.read(provenance.HERE/'candidate-plan.json');p=root/'candidate-package'
            put(p/'installed-source-identity.json',plan['candidate_frontend_source_sha'])
            put(p/'model-source-identity.json',{'version':plan['model']['model_version'],'asset_sha256':{k:v['sha256'] for k,v in plan['model']['assets'].items()}})
            put(p/'candidate-launch.json',{'image_max_pixels':409600,'image_limit':58,'sequences':10,'native_overlay_enabled':False})
            self.assertTrue(provenance.check(root,a,b)['bound'])
            original=provenance.read(p/'installed-source-identity.json');bad=dict(original);bad[next(iter(bad))]='changed';put(p/'installed-source-identity.json',bad)
            with self.assertRaisesRegex(ValueError,'candidate source drift'):provenance.check(root,a,b)
            put(p/'installed-source-identity.json',original);trace[4]['remaining']=[100]
            (root/'supervisor.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in trace))
            with self.assertRaisesRegex(ValueError,'not cleaned'):provenance.check(root,a,b)
