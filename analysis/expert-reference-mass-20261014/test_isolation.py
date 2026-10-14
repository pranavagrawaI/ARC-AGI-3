"""Exercise the actual child runner with a miniature stateful harness."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parent


class IsolationTests(unittest.TestCase):
    def test_each_arm_gets_a_fresh_gate_and_separate_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'sources.json'
            # Install a fake harness module inside each child's execution path.
            s14="import sys,types\na=types.ModuleType('inference');b=types.ModuleType('inference.agent');c=types.ModuleType('inference.agent.tool_agent');c._PRIORITY_GATE=None;b.tool_agent=c;sys.modules.update({'inference':a,'inference.agent':b,'inference.agent.tool_agent':c})\n"
            s20="from inference.agent import tool_agent\nassert tool_agent._PRIORITY_GATE is None\ntool_agent._PRIORITY_GATE={'deadline':1}\nawait asyncio.sleep(0)\n"
            source.write_text(json.dumps({'14':s14,'16':'value=1\n','20':s20}))
            receipts=[]
            for arm in ['control','candidate']:
                out=root/arm;out.mkdir();config=root/(arm+'.json')
                config.write_text(json.dumps({'phase_sources':str(source),'bundle_dir':str(root),'working_dir':str(out),'play_s':900}))
                subprocess.run([sys.executable,'-B',str(HERE/'isolated_phase.py'),str(config)],check=True,timeout=10)
                receipts.append(json.loads((out/'harness-isolation.json').read_text()))
            self.assertTrue(all(r['gate_is_fresh'] for r in receipts))
            self.assertNotEqual(receipts[0]['pid'],receipts[1]['pid'])


if __name__=='__main__':unittest.main(verbosity=2)
