#!/usr/bin/env python3
"""Project Kaggle CLI 2.2.4 metadata into a save request without network I/O."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Self

from kaggle.api.kaggle_api_extended import KaggleApi


class CapturingSaveClient:
    def __init__(self) -> None:
        self.request: Any | None = None

    def save_kernel(self, request: Any) -> SimpleNamespace:
        self.request = request
        return SimpleNamespace()


class FakeKaggleClient:
    def __init__(self, save_client: CapturingSaveClient) -> None:
        self.kernels = SimpleNamespace(kernels_api_client=save_client)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api = KaggleApi()
    save_client = CapturingSaveClient()
    fake_client = FakeKaggleClient(save_client)
    api.build_kaggle_client = lambda: fake_client  # type: ignore[method-assign]
    api.kernels_push(str(args.candidate), timeout="300", acc=None)
    request = save_client.request
    if request is None:
        raise RuntimeError("Kaggle CLI did not construct a save request")
    value = {
        "schema": "kaggle-2.2.4-predecessor-mount-request-projection-v1",
        "network_calls": 0,
        "slug": request.slug,
        "id": request.id,
        "is_private": request.is_private,
        "enable_gpu": request.enable_gpu,
        "enable_tpu": request.enable_tpu,
        "enable_internet": request.enable_internet,
        "session_timeout_seconds": request.session_timeout_seconds,
        "dataset_data_sources": request.dataset_data_sources,
        "competition_data_sources": request.competition_data_sources,
        "kernel_data_sources": request.kernel_data_sources,
        "model_data_sources": request.model_data_sources,
        "machine_shape": request.machine_shape,
        "normalized_request_text_sha256": hashlib.sha256(request.text.encode()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
