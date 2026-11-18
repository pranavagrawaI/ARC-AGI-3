"""Verify the installed push request includes the competition; no transport."""

from contextlib import contextmanager
import json
from types import SimpleNamespace
from unittest.mock import patch
from kaggle.api.kaggle_api_extended import KaggleApi
from build import HERE, PREVIOUS, SLUG, COMPETITION, build


def main():
    for name, data in build().items():
        assert (HERE / name).read_bytes() == data
    assert (HERE / "engine.ipynb").read_bytes() == (
        PREVIOUS / "engine.ipynb"
    ).read_bytes()
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
    result = {
        "status": "PASS",
        "transport_calls": 0,
        "slug": request.slug,
        "machine_shape": request.machine_shape,
        "timeout_s": request.session_timeout_seconds,
        "competition_sources": list(request.competition_data_sources),
        "notebook_unchanged": True,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
