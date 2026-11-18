"""CPU-only retained-source/log audit; no loaders, process signals or APIs."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import signal
import time

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
T14 = BASE / 'trial-14-phase-handles-local'
T15 = BASE / 'trial-15-wire-capacity-local'
R14 = T14 / 'remote-readback/archive-v1-kernel140449504/members/serving-post-control-load14/output'
R15 = T15 / 'remote-readback/partial-final-members/members/serving-wire-capacity15/output'
OVERLAY = BASE.parent / 'm2-winners-20261001/audit/lord-runtime/overlay'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def normalize(value):
    if isinstance(value, str):
        return value.replace('serving-post-control-load14', 'serving-wire-capacity15')
    if isinstance(value, list):
        return [normalize(item) for item in value]
    if isinstance(value, dict):
        return {key: normalize(item) for key, item in value.items()}
    return value


def stages(root):
    stdout = (root / 'candidate.server.stdout').read_text()
    stderr = (root / 'candidate.server.stderr').read_text()
    progress = []
    for match in re.finditer(r'(\d+)/97 \[([0-9:]+)[<,]', stderr):
        seconds = 0
        for number in match[2].split(':'):
            seconds = seconds * 60 + int(number)
        progress.append(dict(shards=int(match[1]), loader_elapsed_seconds=seconds))
    ram = re.search(r'Checkpoint size: ([\d.]+) GiB\. Available RAM: ([\d.]+) GiB', stdout)
    relevant = []
    for number, line in enumerate(stdout.splitlines(), 1):
        if any(term in line for term in ('Initialized PLE embedding', 'Filesystem type for checkpoints',
                'Skipping auto-prefetch', 'Loading weights took', 'Model loading took', 'Starting vLLM server')):
            relevant.append(dict(line=number, text=line))
    return dict(progress=progress, checkpoint_GiB=float(ram[1]), logged_available_RAM_GiB=float(ram[2]),
        source_stage_lines=relevant, observed_full97=any(row['shards'] == 97 for row in progress),
        logs_weight_completion='Loading weights took' in stdout,
        logs_server_start='Starting vLLM server' in stdout,
        health_stdout_count=len(list(root.glob('candidate.health-*.stdout'))),
        health_connection_refused_count=sum('ConnectionRefusedError' in p.read_text() for p in root.glob('candidate.health-*.stdout')))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    signal.alarm(600)
    started = time.monotonic()
    inputs = []
    for root in (R14, R15):
        inputs += [root / ('candidate.server.' + ext) for ext in ('stdout', 'stderr')]
        inputs += [root / 'candidate-package' / name for name in ('installed-source-identity.json',
            'installed-package-identity.json', 'model-source-identity.json', 'runtime-input-hashes.json',
            'candidate-launch.json', 'prepared.json', 'host-headroom.json')]
    inputs += [T15 / name for name in ('phase_tools.py', 'capacity_runner.py', 'release_core.py', 'candidate_prepare.py', 'packet-spec.json')]
    inputs += [T14 / 'candidate_prepare.py', T15 / 'remote-readback/PRIMARY-RESULT.json',
        T15 / 'remote-readback/TERMINAL-HANDOFF.json', R14 / 'supervisor.jsonl', R15 / 'candidate-admission.json']
    source_names = ('vllm/model_executor/model_loader/weight_utils.py',
        'vllm/model_executor/model_loader/default_loader.py', 'vllm/models/qwen4_exp/nvidia/ngram_embedding.py')
    inputs += [OVERLAY / name for name in source_names]
    before = {str(path): sha(path) for path in inputs}
    ids14 = read(R14 / 'candidate-package/installed-source-identity.json')
    ids15 = read(R15 / 'candidate-package/installed-source-identity.json')
    assert ids14 == ids15
    for name in source_names:
        assert sha(OVERLAY / name) == ids15[name], name
    equal = {}
    for name in ('installed-package-identity.json', 'model-source-identity.json',
                 'runtime-input-hashes.json', 'candidate-launch.json', 'prepared.json'):
        equal[name] = normalize(read(R14 / 'candidate-package' / name)) == read(R15 / 'candidate-package' / name)
        assert equal[name], name
    assert (T14 / 'candidate_prepare.py').read_text().replace('serving-post-control-load14-runtime', 'serving-wire-capacity15-runtime') == (T15 / 'candidate_prepare.py').read_text()
    old, new = stages(R14), stages(R15)
    assert old['observed_full97'] and not new['observed_full97']
    assert max(row['shards'] for row in new['progress']) == 19
    assert 'pinned=True' in (R14 / 'candidate.server.stdout').read_text() and 'pinned=True' in (R15 / 'candidate.server.stdout').read_text()
    last = new['progress'][-1]
    first19 = next(row for row in old['progress'] if row['shards'] == 19)
    for packet in (T14, T15):
        freeze = read(packet / 'freeze.json')
        for name, digest in freeze['files'].items():
            assert sha(packet / name) == digest, name
        for name, digest in freeze['references'].items():
            assert sha(name) == digest, name
    assert before == {str(path): sha(path) for path in inputs}
    report = dict(status='LOCAL_READOUT_ONLY_CAUSE_NOT_IDENTIFIED', seconds=time.monotonic()-started,
        inputs=before, installed_source_pins_identical=len(ids15), loader_sources_match_actual_pins={name: ids15[name] for name in source_names},
        package_model_source_config_equal_after_identity_paths_only=equal,
        trial14=old, trial15=new, trial14_first19=first19, trial15_last_progress=last,
        source_findings=[
            'Both logs show BF16 CPU pinned PLE and the same auto-prefetch-skipped branch; this branch difference cannot explain the contrast.',
            'Pinned _get_available_ram_bytes uses min(host available, max(0,cgroup limit-current)); no inactive_file credit. This is not original140 accounting or proof of reclaimability.',
            'Pinned iterator natural-sorts shards then safe_open/get_tensor/yield; progress spans consumer work too, not isolated NFS transfer time.',
            'Pinned default_loader completion log is emitted only after model.load_weights(get_all_weights(...)) returns.',
            'ready() clips to startup end; a final sleep may reach the end before start(next health child), whose generic phase-deadline check raises the retained error.',
            'Candidate primary error is retained through emergency cleanup; this source path supports possible startup-budget exhaustion, not exact remote call-stack attribution.'
        ],
        missing_discriminants=['Trial15 full supervisor spawn/startup_end/probe clocks and exit/cleanup trace.',
            'Trial15 per-stage cgroup/stat/oom series, per-process RSS/HWM, faults/reclaim and storage/I/O latency.',
            'Per-shard tensor loading/CPU copy/quantization timing; progress bars do not identify these costs.',
            'Independent repeated or intervened order evidence; two sessions bundle workload history, cache/NFS service and host load.'],
        alternative_only='Candidate-first could distinguish some order/history effects but changes protocol/cache history; no test, packet or launch authorized, and one contrast would not establish memory/NFS causality.',
        no_GPU=True, no_external_network=True, no_runtime_edits=True, no_timeout_extension=True,
        no_threshold_fit=True, no_loader_cache_or_order_change=True, no_successor_packet=True,
        original_trial15_invalid=True, capacity_ratio=None)
    (args.output / 'AUDIT.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('status', 'seconds', 'installed_source_pins_identical', 'trial14_first19', 'trial15_last_progress')}))
    signal.alarm(0)


if __name__ == '__main__':
    main()
