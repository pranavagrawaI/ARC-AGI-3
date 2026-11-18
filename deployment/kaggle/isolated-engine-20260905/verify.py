"""Offline package/metric tests and installed Kaggle request construction."""

from contextlib import contextmanager
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from kaggle.api.kaggle_api_extended import KaggleApi

from build import HERE, SLUG, build
import runner


def main():
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
    request = captured[0]
    assert request.slug == SLUG
    assert request.machine_shape == "NvidiaRtxPro6000"
    assert request.session_timeout_seconds == 3600
    assert request.is_private and request.enable_gpu and not request.enable_internet
    notebook = json.loads(request.text)
    assert "".join(notebook["cells"][3]["source"]).count("/tmp/engine-runner.py") == 1
    assert "launcher" not in "".join(notebook["cells"][3]["source"])
    assert "solver_manifest_path =" not in "".join(notebook["cells"][2]["source"])
    # Two SSE messages contain 1 then 9 cumulative tokens: never count chunks as tokens.
    chunks = [
        {"meta_info": {"completion_tokens": 1}},
        {
            "output_ids": list(range(9)),
            "text": "test",
            "meta_info": {
                "completion_tokens": 9,
                "finish_reason": {"type": "length"},
                "decode_throughput": 40,
            },
        },
    ]
    data = (
        b"".join(b"data: " + json.dumps(x).encode() + b"\n\n" for x in chunks)
        + b"data: [DONE]\n"
    )
    with tempfile.TemporaryDirectory(prefix="engine-metric-test-") as folder:
        with patch.object(
            runner.urllib.request, "urlopen", return_value=io.BytesIO(data)
        ):
            result = runner.generate(
                Path(folder),
                "fixed-1",
                {"sampling_params": {"max_new_tokens": 9, "ignore_eos": True}},
            )
        assert result["completion_tokens"] == 9
        assert result["steady_decode_tok_s"] == 40
        assert result["tokenizer_decode_span_s"] == 0.2
    print(
        json.dumps(
            {
                "status": "PASS",
                "transport_calls": 0,
                "slug": request.slug,
                "timeout_s": request.session_timeout_seconds,
                "machine_shape": request.machine_shape,
                "checks": [
                    "deterministic package",
                    "engine-only launch",
                    "preserved runtime bootstrap",
                    "token-count metric",
                    "installed push construction",
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
