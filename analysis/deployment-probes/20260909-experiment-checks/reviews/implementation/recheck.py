import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from tools.experiment_checks import flash

with TemporaryDirectory() as temporary:
    raw = Path(temporary)
    (raw / 'EXPERIMENT-CARD.md').write_text((ROOT / 'deployment/kaggle/flash-next-feasibility-20260909/EXPERIMENT-CARD.md').read_text())
    (raw / 'vllm-openai-server.log').write_text('CUDA out of memory\n')
    (raw / 'resources.jsonl').write_text(json.dumps({'epoch': 10, 'gpu': 'GPU, , 2 MiB, 0 %'}) + '\n')
    try:
        result = flash(raw)
        assert result['status'] == 'invalid'
        assert any('unusable GPU sample' in item for item in result['unresolved'])
        print('OOM preserved with unusable-GPU diagnostic')
    except Exception as exc:
        print(type(exc).__name__, str(exc))
        raise
