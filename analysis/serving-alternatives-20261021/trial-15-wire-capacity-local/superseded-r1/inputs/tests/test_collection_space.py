import importlib.util,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
P=Path(__file__).resolve().parents[1]/'collection_tools/space.py';spec=importlib.util.spec_from_file_location('collection_space',P);space=importlib.util.module_from_spec(spec);spec.loader.exec_module(space)
class Space(unittest.TestCase):
 def test_simultaneous_archive_extraction_derived_and_reserve(self):
  with tempfile.TemporaryDirectory() as td:
   need=space.ARCHIVE+space.EXTRACTED+space.DERIVED+space.RESERVE
   with patch.object(space.shutil,'disk_usage',return_value=SimpleNamespace(free=need)):self.assertTrue(space.check(td)['eligible'])
   for free in (8*1024**3,space.ARCHIVE+space.EXTRACTED,need-1):
    with patch.object(space.shutil,'disk_usage',return_value=SimpleNamespace(free=free)):self.assertFalse(space.check(td)['eligible'])
   with patch.object(space.shutil,'disk_usage',return_value=SimpleNamespace(free=space.DERIVED+space.RESERVE)):self.assertTrue(space.check(td,'derived')['eligible'])
