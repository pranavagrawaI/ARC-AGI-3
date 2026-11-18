"""Validate the installed upload request without contacting Kaggle."""

from contextlib import contextmanager
import json
from types import SimpleNamespace
from unittest.mock import patch
from kaggle.api.kaggle_api_extended import KaggleApi
from build import HERE, SLUG, build

for name, data in build().items():
    assert (HERE / name).read_bytes() == data, name
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
r = captured[0]
assert r.slug == SLUG and r.machine_shape == "NvidiaRtxPro6000"
assert r.session_timeout_seconds == 3600 and r.is_private and not r.enable_internet
assert list(r.competition_data_sources) == ["arc-prize-2026-arc-agi-3"]
print(
    json.dumps(
        {
            "status": "PASS",
            "transport_calls": 0,
            "slug": SLUG,
            "timeout": 3600,
            "competition_binding": True,
        },
        indent=2,
    )
)
