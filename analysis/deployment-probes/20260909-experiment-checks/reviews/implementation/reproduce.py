"""Read-only-to-candidate regression probes; all evidence lives in temporary dirs."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from tools.experiment_checks import batching, coverage, flash


def write(path, value):
    path.write_text(json.dumps(value))


def rows(path, values):
    path.write_text(''.join(json.dumps(v) + '\n' for v in values))


def check(label, run):
    try:
        print(label, json.dumps(run()))
    except Exception as exc:
        print(label, type(exc).__name__, str(exc))


with TemporaryDirectory() as temporary:
    raw = Path(temporary)
    card = ROOT / 'deployment/kaggle/flash-next-feasibility-20260909/EXPERIMENT-CARD.md'
    (raw / 'EXPERIMENT-CARD.md').write_text(card.read_text())
    (raw / 'vllm-openai-server.log').write_text('CUDA out of memory\n')
    write(raw / 'vllm-server-identity.json', {'ready_epoch': 10})
    rows(raw / 'resources.jsonl', [{'gpu': 'GPU, 1 MiB, 2 MiB, 0 %', 'epoch': None}])
    check('OOM followed by malformed resource epoch', lambda: flash(raw))

with TemporaryDirectory() as temporary:
    ipc = Path(temporary)
    rows(ipc / 'model_feedback.jsonl', [{'batch_allowed': False}])
    rows(ipc / 'model_predictions.jsonl', [])
    rows(ipc / 'actions.jsonl', [{'turn': 1, 'step': 1, 'actions': 'UP'}])
    rows(ipc / 'batch_stops.jsonl', [])
    check('String action accepted and counted as batch', lambda: batching(ipc))
    rows(ipc / 'actions.jsonl', [{'turn': 1, 'step': 1, 'actions': None}])
    check('Null action crashes inventory', lambda: batching(ipc))

with TemporaryDirectory() as temporary:
    raw = Path(temporary)
    rows(raw / 'results.jsonl', [{'name': 'actor-a', 'usage': None}])
    rows(raw / 'calls.jsonl', [{'request_purpose': 'actor', 'prompt_tokens': 100}])
    check('Null usage crashes coverage', lambda: coverage(raw, [raw / 'calls.jsonl'], 1000, 10))
    rows(raw / 'results.jsonl', [{'name': 'actor-a', 'usage': {'prompt_tokens': 200}},
                                {'name': 'actor-b', 'usage': {'prompt_tokens': '10000'}}])
    check('Malformed fixture silently dropped', lambda: coverage(raw, [raw / 'calls.jsonl'], 1000, 10))
