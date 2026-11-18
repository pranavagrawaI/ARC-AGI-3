"""Check both candidates through the installed Kaggle CLI; transport is mocked."""

from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from kaggle.api.kaggle_api_extended import KaggleApi

HERE = Path(__file__).resolve().parent


def main():
    results = []
    for folder, timeout in [
        (HERE, 7200),
        (HERE.parent / "q6-mtp-integration-20260906", 4800),
    ]:
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
            KaggleApi().kernels_push(
                str(folder), timeout=timeout, acc="NvidiaRtxPro6000"
            )
        assert len(captured) == 1
        request = captured[0]
        metadata = json.loads((folder / "kernel-metadata.json").read_text())
        assert request.slug == metadata["id"]
        assert request.session_timeout_seconds == timeout
        assert request.machine_shape == "NvidiaRtxPro6000"
        assert request.is_private and request.enable_gpu and not request.enable_internet
        assert list(request.competition_data_sources) == ["arc-prize-2026-arc-agi-3"]
        assert json.loads(request.text) == json.loads(
            (folder / metadata["code_file"]).read_text()
        )
        results.append(
            {
                "slug": request.slug,
                "timeout_s": timeout,
                "competition": list(request.competition_data_sources),
                "transport_calls": 0,
            }
        )
    print(json.dumps({"status": "PASS", "requests": results}, indent=2))


if __name__ == "__main__":
    main()
