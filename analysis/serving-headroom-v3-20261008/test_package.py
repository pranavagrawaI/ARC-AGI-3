"""Exercise the actual generated installed-source guard with real source files."""
import ast
import json
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import build

HERE = Path(__file__).resolve().parent


class InstalledGuard(unittest.TestCase):
    def test_generated_guard_accepts_package_rejects_git_fixture_and_drift(self):
        build.build()
        nb=json.loads((HERE/'kernel'/(build.SLUG+'.ipynb')).read_text())
        cell=''.join(nb['cells'][5]['source'])
        calls=[n for n in ast.parse(cell).body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
               and isinstance(n.value.func,ast.Name) and n.value.func.id=='run']
        check=ast.literal_eval(calls[1].value.args[0].elts[2])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'sglang';shutil.copytree(HERE/'native-source',root)
            fake=types.ModuleType('sglang');fake.__file__=str(root/'__init__.py')
            with patch.dict(sys.modules,sglang=fake):
                exec(compile(check,'<actual-installed-guard>','exec'),{})
                allocator=root/'srt/mem_cache/kv_cache_configurator.py'
                actual=allocator.read_bytes()
                allocator.write_bytes((HERE.parent/'serving-headroom-v2-20261008/native-source/srt/mem_cache/kv_cache_configurator.py').read_bytes())
                with self.assertRaisesRegex(ValueError,'installed source mismatch'):
                    exec(compile(check,'<actual-installed-guard>','exec'),{})
                allocator.write_bytes(actual+b'\n# drift\n')
                with self.assertRaisesRegex(ValueError,'installed source mismatch'):
                    exec(compile(check,'<actual-installed-guard>','exec'),{})


if __name__=='__main__':unittest.main()
