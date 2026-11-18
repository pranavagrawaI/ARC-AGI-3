"""Collection ordering and actual watchdog termination, without GPU work."""
from __future__ import annotations
import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/'experiment'))
from collect_outputs import collect
from watchdog import process


class CollectorWatchdog(unittest.TestCase):
    def test_core_precedes_bulk_and_matches_index_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            calls=[]
            names=['tycho-dfranzen/hybrid-games/game/compute.json',
                   'tycho-dfranzen/hybrid-games/game/decision-checkpoints/blob']
            contents={name:b'{"completion_tokens":13}' for name in names}
            index={'files':[{'path':name,'critical':i==0,
                             'sha256':hashlib.sha256(contents[name]).hexdigest()} for i,name in enumerate(names)]}
            contents.update({'tycho-dfranzen/benchmark.json':b'{}',
                             'tycho-dfranzen/candidate-config.json':b'{}',
                             'qualification-start.json':b'{}',
                             'tycho-dfranzen/evidence-index.json':json.dumps(index).encode()})
            class Remote:
                def fetch(self,name): calls.append(name); return contents[name]
            receipt=collect(Remote(),directory,bulk=True)
            self.assertTrue(receipt['critical_complete'])
            self.assertLess(calls.index(names[0]),calls.index(names[1]))
            self.assertEqual(receipt['files'][-1]['phase'],'bulk')

    def test_missing_or_corrupt_core_is_pending_and_prevents_bulk(self):
        for corrupt in (False,True):
            with self.subTest(corrupt=corrupt),tempfile.TemporaryDirectory() as directory:
                calls=[]
                class Remote:
                    def fetch(self,name):
                        calls.append(name)
                        if name.endswith('evidence-index.json'):
                            return json.dumps({'files':[{'path':'tycho-dfranzen/compute.json',
                                 'critical':True,'sha256':'0'*64},
                                 {'path':'tycho-dfranzen/blob','critical':False}]}).encode()
                        if name.endswith('compute.json'):
                            if corrupt:return b'bad'
                            raise OSError('missing')
                        return b'{}'
                receipt=collect(Remote(),directory,bulk=True)
                self.assertFalse(receipt['critical_complete'])
                self.assertNotIn('tycho-dfranzen/blob',calls)
                self.assertTrue(receipt['errors'])

    def test_watchdog_reaps_parent_and_descendant_before_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            body="import subprocess,sys,time; from pathlib import Path; " \
                 "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); " \
                 f"Path({str(root/'child.pid')!r}).write_text(str(p.pid)); time.sleep(60)"
            parent=subprocess.Popen([sys.executable,'-c',body],start_new_session=True)
            start=time.monotonic()
            guard=subprocess.Popen([sys.executable,str(HERE/'experiment/watchdog.py'),
                '--parent',str(parent.pid),'--root',str(root),'--started',str(start),
                '--soft','1','--hard','3','--force-margin','.2'],start_new_session=True)
            try:
                guard.wait(timeout=5); parent.wait(timeout=2)
                child=int((root/'child.pid').read_text())
                for _ in range(100):
                    item=process(child)
                    if item is None or item['state']=='Z':break
                    time.sleep(.01)
                self.assertTrue(process(child) is None or process(child)['state']=='Z')
                self.assertLess(time.monotonic()-start,4)
                events=json.loads((root/'watchdog-receipt.json').read_text())['events']
                self.assertEqual([x['event'] for x in events],['soft_stop_requested','forced_stop'])
            finally:
                if parent.poll() is None: parent.kill(); parent.wait()
                if guard.poll() is None: guard.kill(); guard.wait()


if __name__=='__main__':unittest.main(verbosity=2)
