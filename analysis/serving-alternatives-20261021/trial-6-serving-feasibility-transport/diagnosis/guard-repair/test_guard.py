import json,os,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
P=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(P/'sources'))
FIXTURE=P/'remote-readback/output/serving-candidate-feas6/output'
TABLE=json.loads((FIXTURE/'actual-nvidia-smi.json').read_text())['raw']
QUERY=(FIXTURE/'hardware.stdout').read_text()
TRACE=[json.loads(line) for line in (FIXTURE/'supervisor.jsonl').read_text().splitlines()]
MEM=next(x['value'] for x in TRACE if x.get('name')=='meminfo')
CGROUP=next(x['value'] for x in TRACE if x.get('name')=='cgroup_memory_max')
CORRECTED=Path(__file__).with_name('prepare_corrected.py')

def load(path):
    ns={'__file__':str(P/'sources/prepare.py'),'__name__':'cpu_guard_fixture'}
    exec(compile(path.read_text(),str(path),'exec'),ns)
    return SimpleNamespace(**ns)

class GuardTests(unittest.TestCase):
    def run_guard(self,source,query=QUERY,table=TABLE,mem=MEM,cgroup=CGROUP,disk=81*1024**3):
        module=load(source);read=Path.read_text
        def read_fixture(path,*a,**k):
            if str(path)=='/proc/meminfo':return mem
            if str(path)=='/sys/fs/cgroup/memory.max':return cgroup
            return read(path,*a,**k)
        def command(argv,**kwargs):
            if argv==['nvidia-smi']:return table
            if argv==['nvidia-smi','--query-gpu=name,memory.total,driver_version,uuid','--format=csv,noheader,nounits']:return query
            raise AssertionError('unanticipated external command '+repr(argv))
        with tempfile.TemporaryDirectory() as directory,patch.dict(os.environ,SCREEN_OUTPUT=directory),patch.object(module.subprocess,'check_output',side_effect=command),patch.object(Path,'read_text',read_fixture),patch.object(module.shutil,'disk_usage',return_value=SimpleNamespace(free=disk)):
            module.host_guard()
            return {p.name:json.loads(read(p)) for p in Path(directory).glob('*.json')}

    def test_frozen_failure_before_captured_truncation(self):
        with self.assertRaisesRegex(ValueError,'GPU identity mismatch'):self.run_guard(P/'sources/prepare.py')

    def test_corrected_passes_actual_identity_and_captured_host_with_explicit_disk_fixture(self):
        result=self.run_guard(CORRECTED)
        self.assertEqual(result['actual-gpu-query.json']['name'],'NVIDIA RTX PRO 6000 Blackwell Server Edition')
        self.assertEqual(result['actual-gpu-query.json']['memory_mib'],97887)
        self.assertEqual(result['actual-nvidia-smi.json']['raw'],TABLE)
        self.assertEqual(result['host-headroom.json']['effective_memory_bytes'],187904819200)
        self.assertEqual(result['host-headroom.json']['tmp_free_bytes'],81*1024**3)

    def test_wrong_missing_extra_or_insufficient_gpu_refuses(self):
        negatives=['',QUERY+QUERY,QUERY.replace('Server Edition','Workstation Edition'),
                   QUERY.replace('97887','89999'),QUERY.replace('97887','N/A'),
                   QUERY.replace(', GPU-',', BAD-'),QUERY.replace('580.178.04',''),QUERY.split(',')[0]]
        for query in negatives:
            with self.subTest(query=query),self.assertRaises(ValueError):self.run_guard(CORRECTED,query=query)

    def test_cuda_host_memory_disk_guards_still_refuse(self):
        for kwargs in ({'table':TABLE.replace('CUDA Version: 13.0','CUDA Version: 12.9')},
                       {'mem':'MemTotal:       100000000 kB\n'},
                       {'cgroup':str(169*1024**3)}, {'disk':79*1024**3}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):self.run_guard(CORRECTED,**kwargs)

    def test_exact_thresholds_preserved(self):
        result=self.run_guard(CORRECTED,query=QUERY.replace('97887','90000'),cgroup=str(170*1024**3),disk=80*1024**3)
        self.assertEqual(result['actual-gpu-query.json']['memory_mib'],90000)

if __name__=='__main__':unittest.main(verbosity=2)
