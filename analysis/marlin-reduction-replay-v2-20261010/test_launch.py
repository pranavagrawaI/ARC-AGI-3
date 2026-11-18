import datetime as dt
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import collect
import execute
import launch
from kaggle_backend import KaggleBackend


def acceptance():
    return {'error':None,'versionNumber':1,'kernelId':137800001,'url':'https://www.kaggle.com/code/'+launch.SLUG,
            **{name:[] for name in launch.INVALID}}


class LaunchTests(unittest.TestCase):
    def test_typed_quota_counts_days_and_reserved_time(self):
        good=types.SimpleNamespace(gpu_quota=types.SimpleNamespace(time_used=dt.timedelta(days=1),time_reserved=dt.timedelta(hours=4)))
        receipt=launch.quota_receipt(good)
        self.assertEqual(receipt['used_seconds'],86400);self.assertEqual(receipt['reserved_seconds'],14400)
        good.gpu_quota.time_reserved=dt.timedelta(hours=6)
        with self.assertRaisesRegex(ValueError,'Quota'):launch.quota_receipt(good)

    def test_untyped_missing_or_negative_quota_blocks_launch(self):
        for used,reserved in [(None,dt.timedelta()),('2h',dt.timedelta()),(dt.timedelta(),None),(dt.timedelta(seconds=-1),dt.timedelta())]:
            with self.assertRaises(ValueError):launch.quota_receipt(types.SimpleNamespace(gpu_quota=types.SimpleNamespace(time_used=used,time_reserved=reserved)))

    def test_admission_refuses_competing_or_unknown_kernels(self):
        backend=types.SimpleNamespace(quota=lambda:types.SimpleNamespace(gpu_quota=types.SimpleNamespace(time_used=dt.timedelta(),time_reserved=dt.timedelta())),
            active_sessions=lambda owner:[{'ref':'other','status':'RUNNING'}],require_new_slug=lambda slug:self.fail('must not proceed'))
        with patch.object(launch,'verify_local',return_value={'fixed':'identity'}):
            with self.assertRaisesRegex(ValueError,'Competing'):launch.preflight(backend)

    def test_admission_network_failure_is_not_a_push_attempt(self):
        class Backend:
            def quota(self):raise ConnectionError('mock network unavailable')
            def push(self,*args):raise AssertionError('must not push')
        with tempfile.TemporaryDirectory() as tmp,patch.object(launch,'verify_local',return_value={'fixed':'identity'}):
            with self.assertRaises(ConnectionError):execute.execute(Backend(),Path(tmp),collector=lambda *a:self.fail('must not arm'))
            self.assertFalse((Path(tmp)/'evidence/launch-attempt.json').exists())
            self.assertIn('ConnectionError',(Path(tmp)/'evidence/preflight-history.jsonl').read_text())

    def test_attempt_is_claimed_before_single_push_and_blocks_a_second_push(self):
        calls=[]
        with tempfile.TemporaryDirectory() as tmp,patch.object(launch,'verify_local',return_value={'fixed':'identity'}):
            root=Path(tmp);(root/'evidence').mkdir()
            class Backend:
                def push(self,folder,cap):
                    calls.append(cap)
                    self.assert_claim=json.loads((root/'evidence/launch-attempt.json').read_text())['state']=='attempting'
                    return acceptance()
            backend=Backend();admission={'identity':{'fixed':'identity'}}
            self.assertEqual(launch.push_once(backend,admission,root)['versionNumber'],1)
            self.assertTrue(backend.assert_claim);self.assertEqual(calls,[3600])
            with self.assertRaises(FileExistsError):launch.push_once(backend,admission,root)
            self.assertEqual(calls,[3600])

    def test_ambiguous_save_is_preserved_and_cannot_be_retried(self):
        class Backend:
            calls=0
            def push(self,*args):self.calls+=1;raise ConnectionError('response lost after send')
        with tempfile.TemporaryDirectory() as tmp,patch.object(launch,'verify_local',return_value={'fixed':'identity'}):
            root=Path(tmp);(root/'evidence').mkdir();backend=Backend()
            with self.assertRaises(ConnectionError):launch.push_once(backend,{'identity':{'fixed':'identity'}},root)
            state=json.loads((root/'evidence/launch-attempt.json').read_text())['state'];self.assertIn('no retry',state)
            with self.assertRaises(FileExistsError):launch.push_once(backend,{'identity':{'fixed':'identity'}},root)
            self.assertEqual(backend.calls,1)

    def test_partial_or_wrong_acceptance_is_not_success(self):
        self.assertTrue(launch.accepted_response(acceptance()))
        for key,value in [('versionNumber',2),('versionNumber','1'),('kernelId',0),('invalidModelSources',['bad']),('error','rejected'),('url','other')]:
            row=acceptance();row[key]=value;self.assertFalse(launch.accepted_response(row))

    def test_live_admission_is_repeated_after_collector_starts(self):
        events=[];proc=types.SimpleNamespace(pid=1,poll=lambda:None)
        def admission(*a):events.append('admit');return {'identity':{'fixed':'identity'}}
        def collector(*a):events.append('collector');return proc
        def push(*a):events.append('push');return acceptance()
        with tempfile.TemporaryDirectory() as tmp,patch.object(execute,'preflight',side_effect=admission),patch.object(execute,'push_once',side_effect=push):
            result=execute.execute(object(),Path(tmp),collector=collector)
        self.assertEqual(events,['admit','collector','admit','push']);self.assertEqual(result['state'],'accepted')

    def test_dead_collector_blocks_push(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(execute,'preflight',return_value={}),patch.object(execute,'push_once') as push:
            with self.assertRaisesRegex(ValueError,'Collector'):execute.execute(object(),Path(tmp),collector=lambda r:types.SimpleNamespace(poll=lambda:1))
            push.assert_not_called()

    def test_output_pages_are_explicit_and_version_guarded(self):
        calls=[]
        class Backend:
            def output_page(self,slug,token):
                calls.append(token)
                return [types.SimpleNamespace(file_name='first' if token is None else 'second')],('next' if token is None else None),'raw log'
            def download(self,item,root):path=root/item.file_name;path.write_text('evidence');return path
        with tempfile.TemporaryDirectory() as tmp,patch.object(collect,'remote_guard',return_value={}) as guard:
            root=Path(tmp);result=collect.collect_outputs(Backend(),acceptance(),{'status':'COMPLETE'},root,auditor=lambda path:{'qualified':False,'reason':'fixture'})
            self.assertEqual(calls,[None,'next']);self.assertEqual(guard.call_count,2)
            self.assertTrue((root/'readout/collection.json').exists());self.assertFalse(result['qualified'])

    def test_repeated_output_token_never_becomes_completion(self):
        backend=types.SimpleNamespace(output_page=lambda *a:([],'repeat',None))
        with tempfile.TemporaryDirectory() as tmp,patch.object(collect,'remote_guard',return_value={}):
            root=Path(tmp)
            with self.assertRaisesRegex(ValueError,'pagination'):collect.collect_outputs(backend,acceptance(),{'status':'COMPLETE'},root)
            self.assertFalse((root/'readout/collection.json').exists())

    def test_remote_version_and_source_mismatch_are_rejected(self):
        backend=KaggleBackend.__new__(KaggleBackend)
        expected={'cells':[{'cell_type':'code','source':['print(1)\n']}]}
        meta=types.SimpleNamespace(id=acceptance()['kernelId'],current_version_number=2)
        backend.get_kernel=lambda slug:types.SimpleNamespace(metadata=meta,blob=types.SimpleNamespace(source=json.dumps(expected)))
        with self.assertRaisesRegex(ValueError,'identity'):backend.verify_remote(launch.SLUG,acceptance(),expected,{})
        meta.current_version_number=1
        backend.get_kernel=lambda slug:types.SimpleNamespace(metadata=meta,blob=types.SimpleNamespace(source=json.dumps({'cells':[{'cell_type':'code','source':'changed'}]})))
        with self.assertRaisesRegex(ValueError,'cells'):backend.verify_remote(launch.SLUG,acceptance(),expected,{})


if __name__=='__main__':unittest.main()
