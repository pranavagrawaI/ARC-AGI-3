"""Freeze only this isolated quick-probe source snapshot."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = ['nooa_worker.py','headless.py','state_delivery.py','local_solver.py','stub_agent.py',
         'assess_candidate.py',
         'chassis/decision_packet.py','chassis/designer_player.md','chassis/step_check.py',
         'chassis/evidence_renderer.py','chassis/discovery-demo.md',
         'nooa_public_probe.py','chassis/arc_llm.py','chassis/solver_agent.py',
         'chassis/prediction_feedback.py','chassis/host_contract.py','chassis/stall_contract.py','chassis/integrity_guard.py',
         'chassis/skills/interactive-game-solver/SKILL.md']
SOURCES = ['exact_request_outer_runtime.py','build_notebook.py','freeze_candidate.py',
           'adapter.py','shared_local_solver.py','run_official_agent.py',
           'openrouter_probe.py','probe_lifecycle.py']
SOURCES.append('procedural_transfer_probe.py')
SOURCES.append('procedural_transfer_reliability_probe.py')
SOURCES.append('typed_self_test_live_probe.py')
SOURCES.append('heterogeneous_self_test_probe.py')


def manifest():
    hashes = lambda names: {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in names}
    contents = {'files': hashes(FILES), 'sources': hashes(SOURCES)}
    return {'schema':'q6-nooa-candidate-v1', 'identity':'designer-player-discovery-20260908-v2',
            'base':'designer-player-reasoning-20260907', 'contract_sha256':hashlib.sha256(json.dumps(contents,sort_keys=True).encode()).hexdigest(), **contents}


def verify():
    actual=manifest()
    if json.loads((HERE/'CHASSIS.json').read_text()) != actual:
        raise RuntimeError('Quick-probe source drift; rebuild and requalify before launch')
    return actual


if __name__=='__main__':
    import sys
    if '--check' not in sys.argv:
        (HERE/'CHASSIS.json').write_text(json.dumps(manifest(),indent=2)+'\n')
    print(verify()['contract_sha256'])
