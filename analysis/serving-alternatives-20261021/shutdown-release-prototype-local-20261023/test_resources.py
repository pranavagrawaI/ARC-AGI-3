import copy
import importlib.util
import json
from pathlib import Path
import unittest

import resources

HERE=Path(__file__).resolve().parent
TRIAL=HERE.parent/'trial-10-exact-wire-local'


class ResourceTests(unittest.TestCase):
    def setUp(self):
        self.raw=json.loads((TRIAL/'remote-readback/priority-output/serving-wire-capacity10/output/control-release.json').read_text())['raw']
        self.uuid=self.raw['gpu'].split(',')[2].strip()

    def test_original_gpu_host_disk_boundaries_remain_exact(self):
        raw=copy.deepcopy(self.raw)
        raw['gpu']=f'{resources.GPU_NAME},2048,{self.uuid}\n'
        self.assertTrue(resources.gpu(raw['gpu'],self.uuid)['eligible'])
        self.assertFalse(resources.gpu(raw['gpu'].replace(',2048,',',2049,'),self.uuid)['eligible'])
        fields={k:int(v) for k,v in (s.split() for s in raw['stat'].splitlines())}
        raw['current']=str(int(raw['maximum'])+fields['inactive_file']-140*1024**3)
        self.assertTrue(resources.cgroup(raw)['eligible'])
        raw['current']=str(int(raw['current'])+1)
        self.assertFalse(resources.cgroup(raw)['eligible'])
        for free,expected in [(100*1024**3,True),(100*1024**3-1,False)]:
            raw['disk']=free
            probe=resources.providers(raw,self.uuid,lambda:dict(eligible=True),lambda:dict(eligible=True))
            self.assertEqual(probe['disk']()['eligible'],expected)

    def test_partitioned_predicates_agree_with_frozen_validator(self):
        spec=importlib.util.spec_from_file_location('frozen_release',TRIAL/'release_check.py')
        old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        for gpu in [0,2048,2049,95906]:
            for current in [0,int(self.raw['current'])]:
                for disk in [100*1024**3-1,100*1024**3]:
                    raw=copy.deepcopy(self.raw)
                    raw.update(gpu=f'{resources.GPU_NAME},{gpu},{self.uuid}\n',current=str(current),disk=disk)
                    probes=resources.providers(raw,self.uuid,lambda:dict(eligible=True),lambda:dict(eligible=True))
                    partitioned=all(probes[name]()['eligible'] for name in ('gpu','apps','cgroup','disk'))
                    try:old.validate(**raw);expected=True
                    except ValueError:expected=False
                    self.assertEqual(partitioned,expected)

    def test_bad_gpu_identity_missing_fields_and_nonempty_apps_fail(self):
        with self.assertRaises(ValueError):resources.gpu(self.raw['gpu'],'GPU-wrong')
        with self.assertRaises(ValueError):resources.gpu('truncated',self.uuid)
        with self.assertRaises(KeyError):resources.cgroup({})
        self.raw['apps']='1234, unrelated, 100\n'
        probes=resources.providers(self.raw,self.uuid,lambda:dict(eligible=False),lambda:dict(eligible=False))
        self.assertFalse(probes['apps']()['eligible'])
        self.assertFalse(probes['port']()['eligible'])


if __name__=='__main__':unittest.main()
