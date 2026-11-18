"""Exercise installed Kaggle request construction with transport mocked."""

from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from kaggle.api.kaggle_api_extended import KaggleApi

HERE = Path(__file__).resolve().parent
captured = []


@contextmanager
def client(_self):
    yield SimpleNamespace(
        kernels=SimpleNamespace(
            kernels_api_client=SimpleNamespace(
                save_kernel=lambda request: captured.append(request)
            )
        )
    )


with patch.object(KaggleApi, "build_kaggle_client", client):
    KaggleApi().kernels_push(str(HERE), timeout=7800, acc="NvidiaRtxPro6000")
assert len(captured) == 1
request = captured[0]
assert request.slug == "pranavagrawai/mtp-sampling-repair-20260906"
assert request.machine_shape == "NvidiaRtxPro6000"
assert request.session_timeout_seconds == 7800
assert request.is_private and request.enable_gpu and not request.enable_internet
assert list(request.competition_data_sources) == ["arc-prize-2026-arc-agi-3"]
assert json.loads(request.text) == json.loads(
    (HERE / "sampling_repair.ipynb").read_text()
)
print(
    json.dumps(
        {
            "status": "PASS",
            "transport_calls": 0,
            "slug": request.slug,
            "timeout_s": 7800,
            "competition": list(request.competition_data_sources),
        },
        indent=2,
    )
)
