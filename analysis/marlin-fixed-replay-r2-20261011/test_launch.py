import datetime as dt
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import execute
import launch
import collect


class LauncherTests(unittest.TestCase):
    def test_failed_live_quota_never_arms_collector_or_pushes(self):
        quota=types.SimpleNamespace(time_used=dt.timedelta(seconds=34386),time_reserved=dt.timedelta(),total_time_allowed=dt.timedelta(seconds=21600))
        backend=types.SimpleNamespace(quota=lambda:types.SimpleNamespace(gpu_quota=quota,quota_refresh_time=dt.datetime(2026,10,17)))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);calls=[]
            with patch.object(launch,'verify_local',return_value={'sealed':True}):
                with self.assertRaisesRegex(ValueError,'Live reported allowance'):
                    execute.execute(backend,root,collector=lambda _:calls.append('collector'))
            self.assertEqual(calls,[])
            self.assertFalse((root/'evidence/launch-attempt.json').exists())
            self.assertEqual(json.loads((root/'evidence/quota-observed.json').read_text())['allowed_seconds'],21600)

    def test_ambiguous_push_consumes_exclusive_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'evidence').mkdir();identity={'sealed':True}
            (root/'evidence/GO.json').write_text(json.dumps({'run_id':root.name,'hard_timeout_s':3600,'instruction':'fixture authority'}))
            calls=[]
            def push(*args):calls.append(args);raise TimeoutError()
            backend=types.SimpleNamespace(push=push)
            with patch.object(launch,'verify_local',return_value=identity):
                with self.assertRaises(TimeoutError):launch.push_once(backend,{'identity':identity},root)
                with self.assertRaises(FileExistsError):launch.push_once(backend,{'identity':identity},root)
            self.assertEqual(len(calls),1)
            self.assertEqual(json.loads((root/'evidence/launch-attempt.json').read_text())['state'],'acceptance unresolved; no retry')

    def test_collection_audit_failure_is_terminal_after_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);calls=[]
            def download(item,out):
                path=out/'result.json';path.write_text('{}');calls.append('download');return path
            backend=types.SimpleNamespace(output_page=lambda *a:([{}],None,None),download=download)
            def broken(_):raise ModuleNotFoundError('numpy')
            with patch.object(collect,'remote_guard',return_value={}):
                result=collect.collect_outputs(backend,{'kernelId':1,'versionNumber':1},{'status':'COMPLETE'},root,auditor=broken)
            self.assertFalse(result['qualified'])
            self.assertEqual(calls,['download'])
            self.assertTrue((root/'readout/collection.json').is_file())


if __name__=='__main__':unittest.main()
