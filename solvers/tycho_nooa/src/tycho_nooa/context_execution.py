"""Versioned sterile executor for Context/Evidence Plane v1."""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import JsonValue, field_validator

from tycho_nooa.sterile_execution import ExecutionManifest, SterileExecutor


class ContextExecutionManifest(ExecutionManifest):
    """Add only the read/view capabilities required by the context plane."""

    capabilities: tuple[
        Literal["evidence.query", "evidence.read", "evidence.view"], ...
    ] = ()

    @field_validator("input_values")
    @classmethod
    def context_names_do_not_shadow_capabilities(
        cls, values: dict[str, JsonValue]
    ) -> dict[str, JsonValue]:
        if "evidence_cache" in values:
            raise ValueError("input_values name is unavailable: 'evidence_cache'")
        return values


class ContextComputationalSterileExecutor[ReturnT](SterileExecutor[ReturnT]):
    """Use a new worker without changing any frozen computational package."""

    async def __aenter__(self) -> ContextComputationalSterileExecutor[ReturnT]:
        if self._process is not None:
            raise RuntimeError("sterile executor is already running")
        if self._provenance is not None:
            self._provenance.start(self._manifest.model_dump(mode="json"))
        self._scratch = tempfile.TemporaryDirectory(prefix="tycho-codeact-")
        worker = Path(__file__).with_name("_context_computational_worker.py")
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
                    "protocol_version": 1,
                    "type": "initialize",
                    "manifest": self._manifest.model_dump(mode="json"),
                }
            )
            response = await self._receive()
            if response.get("type") != "ready":
                raise RuntimeError(f"sterile worker initialization failed: {response}")
        except Exception:
            if self._provenance is not None:
                self._provenance.fail("worker_start_failed", locals_lost=True)
            await self.aclose()
            raise
        return self
