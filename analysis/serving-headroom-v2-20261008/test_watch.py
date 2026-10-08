import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent/'mtp-depth-20261006')]
import watch
from test_audit import fixture

class WatchTests(unittest.TestCase):
    def test_all_output_pages_collected_and_audited(self):
        calls=[]
        class API:
            def kernels_output(self,slug,path,**kw):
                calls.append(kw['page_token'])
                if kw['page_token'] is None:
                    fixture(Path(path))
                    return ['first'],'next'
                return ['second'],None
        with tempfile.TemporaryDirectory() as tmp,patch.object(watch,'READOUT',Path(tmp)):
            (Path(tmp)/'raw').mkdir()
            self.assertTrue(watch.collect(API(),{'status':'COMPLETE'})['qualified'])
            self.assertEqual(calls,[None,'next'])
            self.assertTrue((Path(tmp)/'audit.json').is_file())

    def test_repeating_output_token_cannot_be_completion(self):
        class API:
            def kernels_output(self,*args,**kw):return [],'same'
        with tempfile.TemporaryDirectory() as tmp,patch.object(watch,'READOUT',Path(tmp)):
            with self.assertRaisesRegex(RuntimeError,'repeated output'):watch.collect(API(),{'status':'COMPLETE'})
            self.assertFalse((Path(tmp)/'collection.json').exists())

if __name__=='__main__':unittest.main()
