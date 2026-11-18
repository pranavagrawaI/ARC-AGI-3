"""Exercise installed Kaggle request construction for the speed probe; no push."""

from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace
from importlib.metadata import version
from unittest.mock import patch

from kaggle.api.kaggle_api_extended import KaggleApi

HERE = Path(__file__).resolve().parent


def main():
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
        KaggleApi().kernels_push(str(HERE), timeout=10800, acc="NvidiaRtxPro6000")
    assert len(captured) == 1
    request = captured[0]
    assert request.slug == "pranavagrawai/q6-summary-boundary-20260906", request.slug
    assert request.session_timeout_seconds == 10800
    assert request.machine_shape == "NvidiaRtxPro6000"
    assert request.is_private and request.enable_gpu and not request.enable_internet
    notebook = json.loads(request.text)
    code = "\n".join("".join(c["source"]) for c in notebook["cells"])
    checks = {
        "control arm": "ARC_STALL_REVIEW'] = '0'" in code,
        "100 actions": "TYCHO_KAGGLE_MAX_ACTIONS'] = '100'" in code,
        "ls20 only": "TYCHO_PUBLIC_GAME_IDS'] = 'ls20'" in code,
        "worker 900s": "TYCHO_NOOA_WORKER_TIMEOUT_SECONDS'] = '900'" in code,
        "launcher 8400s": "'--launcher-timeout-s', '8400'" in code,
        "request 180s": "'--request-timeout-s', '180'" in code,
        "no submit": "kaggle competitions submit" not in code,
    }
    assert all(checks.values()), checks
    print(json.dumps({
        "status": "PASS",
        "transport_calls": 0,
        "cli_version": version("kaggle"),
        "slug": request.slug,
        "session_timeout_seconds": request.session_timeout_seconds,
        "machine_shape": request.machine_shape,
        "private": request.is_private,
        "checks": checks,
    }, indent=2))


if __name__ == "__main__":
    main()
