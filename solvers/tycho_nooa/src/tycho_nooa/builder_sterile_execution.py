"""Versioned sterile process boundary for Builder proposal computation."""

from __future__ import annotations

import asyncio
import json
import os
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any, Literal, TypeVar

from pydantic import JsonValue, field_validator

from tycho_nooa.sterile_execution import ExecutionManifest, SterileExecutor

ReturnT = TypeVar("ReturnT")
_BUILDER_PROTOCOL_BYTES = 8 * 1024 * 1024


class BuilderExecutionManifest(ExecutionManifest):
    """Data-only Builder realm with exactly one legal host capability."""

    role: Literal["builder"] = "builder"
    capabilities: tuple[Literal["builder.evaluate"], ...] = ()

    @field_validator("input_values")
    @classmethod
    def builder_namespace_is_reserved(
        cls, values: dict[str, JsonValue]
    ) -> dict[str, JsonValue]:
        if "builder" in values:
            raise ValueError("input_values name is unavailable: 'builder'")
        return values


class BuilderSterileExecutor(SterileExecutor[ReturnT]):
    """Run Builder cells in a clean process without changing the frozen Actor worker."""

    async def __aenter__(self) -> BuilderSterileExecutor[ReturnT]:
        if self._process is not None:
            raise RuntimeError("sterile executor is already running")
        if self._provenance is not None:
            self._provenance.start(self._manifest.model_dump(mode="json"))
        self._scratch = tempfile.TemporaryDirectory(prefix="tycho-builder-codeact-")
        worker = Path(__file__).with_name("_builder_sterile_worker.py")
        environment = {
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": os.defpath,
            "PYTHONHASHSEED": "0",
            "TMPDIR": self._scratch.name,
        }
        try:
            self._process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-I",
                "-S",
                str(worker),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._scratch.name,
                env=environment,
                close_fds=True,
            )
            await self._send(
                {
                    "protocol_version": self._manifest.protocol_version,
                    "type": "initialize",
                    "manifest": self._manifest.model_dump(mode="json"),
                }
            )
            response = await self._receive()
            if response.get("type") != "ready":
                raise RuntimeError(
                    f"Builder worker initialization failed: {response}"
                )
        except Exception:
            if self._provenance is not None:
                self._provenance.fail("worker_start_failed", locals_lost=True)
            await self.aclose()
            raise
        return self

    async def _send(self, payload: dict[str, Any]) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise RuntimeError("Builder sterile executor is not running")
        encoded = json.dumps(
            payload, allow_nan=False, separators=(",", ":")
        ).encode("utf-8")
        if len(encoded) > _BUILDER_PROTOCOL_BYTES:
            raise ValueError("Builder sterile request exceeds its byte limit")
        process.stdin.write(struct.pack(">I", len(encoded)) + encoded)
        await process.stdin.drain()

    async def _receive(self) -> dict[str, Any]:
        process = self._process
        if process is None or process.stdout is None:
            raise RuntimeError("Builder sterile worker stdout is unavailable")
        header = await process.stdout.readexactly(4)
        size = struct.unpack(">I", header)[0]
        if size > _BUILDER_PROTOCOL_BYTES:
            raise RuntimeError("Builder sterile response exceeds its byte limit")
        parsed = json.loads(await process.stdout.readexactly(size))
        if not isinstance(parsed, dict):
            raise TypeError("Builder sterile response must be an object")
        return parsed
