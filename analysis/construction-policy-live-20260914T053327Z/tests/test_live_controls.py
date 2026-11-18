import importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest import mock
HERE=Path(__file__).resolve().parents[1]; PACKAGE=HERE.parents[1]/'deployment/kaggle'/HERE.name
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
PRE=module(HERE/'prelaunch_check.py','live_prelaunch'); LAUNCH=module(PACKAGE/'launch_once.py','live_launch')
class Controls(unittest.TestCase):
    def review(self,status='RESOLVED',manifest='m',card='c'):
        return {'status':status,'identity':HERE.name,'manifest_sha256':manifest,'card_sha256':card,'notebook_sha256':PRE.R4['notebook']}
    def test_review_exact_only(self):
        self.assertTrue(PRE.review_valid(self.review(),'m','c'))
        for change in ({'status':'HOLD'},{'manifest_sha256':'wrong'},{'card_sha256':'wrong'},{'identity':'wrong'}):
            r=self.review();r.update(change);self.assertFalse(PRE.review_valid(r,'m','c'))
    def test_empty_authorization_is_rejected(self):
        self.assertFalse(PRE.authorization_valid({},'m','c'))
        valid={'used':False,'identity':HERE.name,'manifest_sha256':'m','card_sha256':'c','maximum_pushes':1,'retries':0}
        self.assertTrue(PRE.authorization_valid(valid,'m','c'))
    def setup_launch(self,tmp):
        root=Path(tmp); package=root/'deployment/kaggle'/HERE.name; analysis=root/'analysis'/HERE.name; package.mkdir(parents=True);analysis.mkdir(parents=True);(analysis/'evidence').mkdir()
        meta=json.loads((HERE/'SERVER-REFERENCE.json').read_text());(package/'kernel-metadata.json').write_text(json.dumps(meta));(analysis/'SERVER-REFERENCE.json').write_text(json.dumps(meta));(analysis/'AUTHORIZATION.json').write_text('{}');(analysis/'REVIEW-RESOLVED.json').write_text('{}')
        return root,package,analysis,meta
    def receipts(self,meta,push_stdout):
        return [
          {'returncode':0,'stdout':'[{"resource":"GPU","remaining":"2.0h","used":"1.0h"}]','stderr':'','arguments':[]},
          {'returncode':0,'stdout':'[]','stderr':'','arguments':[]},
          {'returncode':0,'stdout':push_stdout,'stderr':'','arguments':[],'timed_out':False}]
    def run_main(self,push_stdout):
        td=tempfile.TemporaryDirectory();root,package,analysis,meta=self.setup_launch(td.name); calls=[]; rows=self.receipts(meta,push_stdout)
        def invoke(args,timeout=120): calls.append(args);return rows.pop(0)
        gate={'status':'APPROVED_NOT_LAUNCHED','manifest_sha256':'m'}
        patches=[mock.patch.object(LAUNCH,'HERE',package),mock.patch.object(LAUNCH,'REPO',root),mock.patch.object(LAUNCH,'ANALYSIS',analysis),mock.patch.object(LAUNCH,'invoke',side_effect=invoke),mock.patch.object(LAUNCH,'local_gate',return_value=gate)]
        for p in patches:p.start()
        try: LAUNCH.main(); error=None
        except Exception as e:error=e
        finally:
            for p in reversed(patches):p.stop()
        return td,analysis,calls,error
    def test_simulated_returned_slug(self):
        url='https://www.kaggle.com/code/pranavagrawai/ls20-construction-policy-live-20260914t053327z'
        td,a,calls,error=self.run_main('Kernel version 1 successfully pushed.  Please check progress at '+url+'\n')
        try:
            self.assertIsNone(error);self.assertEqual(sum(x[:2]==['kernels','push'] for x in calls),1);self.assertEqual(json.loads((a/'evidence/LAUNCH-RESULT.json').read_text())['url'],url)
        finally: td.cleanup()
    def test_uncertain_admission_consumes_and_no_retry(self):
        td,a,calls,error=self.run_main('unexpected success text')
        try:
            self.assertIn('UNCERTAIN_ADMISSION_NO_RETRY',str(error));self.assertTrue((a/'LAUNCH-CONSUMED.json').exists());self.assertEqual(sum(x[:2]==['kernels','push'] for x in calls),1)
            with mock.patch.object(LAUNCH,'ANALYSIS',a):
                with self.assertRaises(FileExistsError): LAUNCH.consume(['x'],{'manifest_sha256':'m'})
        finally: td.cleanup()
    def test_target_exists_blocks_before_consume(self):
        with tempfile.TemporaryDirectory() as td:
            root,package,analysis,meta=self.setup_launch(td); rows=self.receipts(meta,'unused');rows[1]['stdout']=json.dumps([{'ref':meta['id']}]);calls=[]
            with mock.patch.object(LAUNCH,'HERE',package),mock.patch.object(LAUNCH,'REPO',root),mock.patch.object(LAUNCH,'ANALYSIS',analysis),mock.patch.object(LAUNCH,'invoke',side_effect=lambda args,timeout=120:(calls.append(args) or rows.pop(0))),mock.patch.object(LAUNCH,'local_gate',return_value={'status':'APPROVED_NOT_LAUNCHED','manifest_sha256':'m'}):
                with self.assertRaisesRegex(RuntimeError,'FRESH_TARGET_NOT_ABSENT'): LAUNCH.main()
            self.assertFalse((analysis/'LAUNCH-CONSUMED.json').exists());self.assertFalse(any(x[:2]==['kernels','push'] for x in calls))
    def test_current_package_prepared_not_approved(self):
        result=PRE.check(write=False);self.assertEqual(result['status'],'PREPARED_NOT_APPROVED');self.assertEqual(result['final_review'],'MISSING')
if __name__=='__main__': unittest.main()
