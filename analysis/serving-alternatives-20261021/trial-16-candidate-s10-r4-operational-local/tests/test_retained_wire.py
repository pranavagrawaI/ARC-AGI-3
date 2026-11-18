"""Exact incumbent producer/accounting regressions against current packet imports."""
import importlib,sys,unittest
import api_arm
sys.modules['reduce_api']=api_arm
from pathlib import Path
OLD=Path(__file__).resolve().parents[2]/'trial-10-exact-wire-local'
def load_tests(loader,tests,pattern):
 sys.path.append(str(OLD/'tests'))
 suite=unittest.TestSuite()
 for name in ('test_accounting','test_source_semantics','test_body_binding','test_deadline_race','test_persistence','test_protocol_cutoff'):
  suite.addTests(loader.loadTestsFromModule(importlib.import_module(name)))
 return suite
