"""Containment and unchanged-parent gates without model/API use."""
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

import screen
from patch_runtime import install, WHEEL_SHA, SOURCE_SHA, MEMBER
import zipfile

HERE=Path(__file__).resolve().parent


class ScreenTests(unittest.TestCase):
    def test_owned_server_group_is_reaped(self):
        proc=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)'],start_new_session=True)
        screen.OWNED['proc']=proc
        screen.stop_server()
        self.assertIsNotNone(proc.poll())
        self.assertIsNone(screen.OWNED['proc'])

    def test_top_level_await_and_sync_snippets_execute(self):
        ns={'asyncio':asyncio}
        asyncio.run(screen.execute('value=2\n',ns))
        asyncio.run(screen.execute('await asyncio.sleep(0)\nvalue+=3',ns))
        self.assertEqual(ns['value'],5)

    def test_patch_install_control_candidate_restore_and_drift(self):
        wheel=Path(os.environ['REFERENCE_WHEEL'])
        with tempfile.TemporaryDirectory() as tmp:
            site=Path(tmp);target=site/MEMBER;target.parent.mkdir(parents=True)
            (site/'sglang/srt/layers/moe').mkdir(parents=True)
            with zipfile.ZipFile(wheel) as archive:original=archive.read(MEMBER)
            target.write_bytes(original)
            receipt=site/'receipt.json'
            install(wheel,site,'control',receipt);self.assertEqual(target.read_bytes(),original)
            install(wheel,site,'candidate',receipt);self.assertNotEqual(target.read_bytes(),original)
            install(wheel,site,'control',receipt);self.assertEqual(target.read_bytes(),original)
            target.write_bytes(original+b'\n')
            with self.assertRaises(ValueError):install(wheel,site,'candidate',receipt)


if __name__=='__main__':unittest.main(verbosity=2)
