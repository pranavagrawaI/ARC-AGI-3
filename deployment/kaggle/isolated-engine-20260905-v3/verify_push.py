"""Check competition, accelerator and bounds in the installed CLI request."""

from contextlib import contextmanager
import json
from types import SimpleNamespace
from unittest.mock import patch
from kaggle.api.kaggle_api_extended import KaggleApi
from build import HERE, SLUG, COMPETITION, build

for name, data in build().items():
    assert (HERE / name).read_bytes() == data
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
    KaggleApi().kernels_push(str(HERE), timeout=3600, acc="NvidiaRtxPro6000")
assert len(captured) == 1
request = captured[0]
assert request.slug == SLUG
assert request.machine_shape == "NvidiaRtxPro6000"
assert request.session_timeout_seconds == 3600
assert request.is_private and request.enable_gpu and not request.enable_internet
assert list(request.competition_data_sources) == [COMPETITION]
print(
    json.dumps(
        {
            "status": "PASS",
            "transport_calls": 0,
            "slug": SLUG,
            "competition": COMPETITION,
            "timeout_s": 3600,
            "accelerator": request.machine_shape,
        },
        indent=2,
    )
)
