"""Exercise installed Kaggle request construction with transport replaced; no push."""

from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from importlib.metadata import version
from unittest.mock import patch

from kaggle.api.kaggle_api_extended import KaggleApi

from prepare import HERE, prepare


def main():
    binding = prepare(check=True)
    assert version("kaggle") == binding["kaggle_cli_version"]
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
        KaggleApi().kernels_push(str(HERE), timeout=14400, acc="NvidiaRtxPro6000")
    assert len(captured) == 1
    request = captured[0]
    assert request.slug == binding["kernel"]
    assert request.session_timeout_seconds == 14400
    assert request.machine_shape == "NvidiaRtxPro6000"
    assert request.is_private and request.enable_gpu and not request.enable_internet
    notebook = json.loads(request.text)
    code = "\n".join("".join(c["source"]) for c in notebook["cells"])
    assert "ARC_STALL_REVIEW'] = '0'" in code
    assert "TYCHO_KAGGLE_MAX_ACTIONS'] = '200'" in code
    assert "TYCHO_PUBLIC_GAME_IDS'] = 'ls20'" in code
    print(
        json.dumps(
            {
                "status": "PASS",
                "transport_calls": 0,
                "cli_version": version("kaggle"),
                "session_timeout_seconds": request.session_timeout_seconds,
                "machine_shape": request.machine_shape,
                "private": request.is_private,
                "api_source_sha256": hashlib.sha256(
                    Path(inspect.getfile(KaggleApi)).read_bytes()
                ).hexdigest(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
