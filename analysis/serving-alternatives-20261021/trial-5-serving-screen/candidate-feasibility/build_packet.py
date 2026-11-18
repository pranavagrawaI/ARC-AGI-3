"""Build a review-only candidate packet. No SDK, allocation or upload."""
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
RUNNER=HERE.parent/'runner'
sys.path.insert(0,str(RUNNER))
import build as runner_build


def main(destination):
    root='/kaggle/working/serving-candidate-feas5'
    def phase(operation,seconds):return dict(argv=['python3',root+'/prepare.py',operation],seconds=seconds)
    config=dict(mode='GPU_SCREEN',purpose='CANDIDATE_FEASIBILITY',arm_order=['candidate'],
                run_id='serving-candidate-feas5-20261023',cwd=root,total_seconds=5100,cleanup_seconds=300,
                hardware=dict(argv=['nvidia-smi','--query-gpu=name,memory.total,driver_version,uuid',
                                    '--format=csv,noheader,nounits'],seconds=30,minimum_memory_mib=90000),
                candidate=dict(prepare=phase('prepare',900),identity=phase('identity',120),server=phase('serve',2700),
                    ready_url='http://127.0.0.1:8001/health',ready_seconds=2700,request_seconds=10,drain_seconds=5,
                    health_interval_seconds=2,
                    functional=dict(argv=['python3',root+'/client.py','run',root+'/cases.json','http://127.0.0.1:8001'],seconds=900),
                    qualify=dict(argv=['python3',root+'/client.py','qualify',root+'/cases.json'],seconds=30)))
    meta=dict(id='pranavagrawai/serving-candidate-feas5-20261023',title='serving-candidate-feas5-20261023',
              code_file='serving-candidate-feas5-20261023.ipynb',language='python',kernel_type='notebook',is_private=True,
              enable_gpu=True,enable_tpu=False,enable_internet=False,machine_shape='NvidiaRtxPro6000',
              docker_image=runner_build.IMAGE,competition_sources=['arc-prize-2026-arc-agi-3'],kernel_sources=[],
              dataset_sources=['lordhansolo/vllm-main-e975732-arc3'],
              model_sources=['lordhansolo/qwen3-8-flash-next-mixed-nvfp4-fp8/PyTorch/hf-mixed-mtp-nvfp4/1'])
    names=['prepare.py','client.py','candidate_library.py','plan.json','cases.json','draft_vocab_32k.json','chat_template.normalized.jinja']
    files={name:str(HERE/name) for name in names}
    files.update({str(p.relative_to(HERE)):str(p) for p in (HERE/'cases').glob('*.json')})
    spec=dict(config=config,metadata=meta,files=files)
    (HERE/'packet-spec.json').write_text(json.dumps(spec,indent=2)+'\n')
    manifest=runner_build.build(spec,destination)
    manifest['platform_hard_timeout_s']=5400
    (HERE/'review-packet.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(destination=str(destination),sources_sha256=manifest['sources_sha256'],
                         config_sha256=manifest['config_sha256'],platform_timeout_s=5400,supervisor_timeout_s=5100,
                         launch_authorized=False)))


if __name__=='__main__':main(Path(sys.argv[1]))
