import copy
import json
import unittest
import control_prepare
import release_check

class Guards(unittest.TestCase):
    def test_live_listener_cannot_pass_release(self):
        import socket
        with socket.socket() as listener:
            listener.bind(('127.0.0.1',0));listener.listen()
            with self.assertRaises(OSError):release_check.free_port(listener.getsockname()[1])
    def test_release_eligibility_not_total_memory_only(self):
        args=dict(gpu='NVIDIA RTX PRO 6000 Blackwell Server Edition, 0, GPU-CPU\n',apps='',meminfo='MemAvailable: 160000000 kB\n',maximum=str(175*1024**3),current=str(100*1024**3),stat='inactive_file 90000000000\n',disk=150*1024**3)
        self.assertGreater(release_check.validate(**args)['reclaimable_host_bytes'],140*1024**3)
        for changed in ({'apps':'42, server, 40000'},{'gpu':'Tesla T4, 0, GPU-CPU\n'},{'current':str(170*1024**3),'stat':'inactive_file 0\n'},{'disk':99*1024**3},{'gpu':'NVIDIA RTX PRO 6000 Blackwell Server Edition, 60000, GPU-CPU\n'}):
            with self.assertRaises(ValueError):release_check.validate(**dict(args,**changed))
    def test_shipping_numeric_argv_drift_rejected(self):
        a=json.loads((control_prepare.HERE/'control-argv.reference.json').read_text())
        from pathlib import Path
        target=Path(a[a.index('--model-path')+1]);draft=Path(a[a.index('--speculative-draft-model-path')+1])
        control_prepare.verify_argv(a,target,draft,Path('/not-used'))
        a[a.index('--max-running-requests')+1]='9'
        with self.assertRaisesRegex(ValueError,'argv drift'):control_prepare.verify_argv(a,target,draft,Path('/not-used'))
