"""Sterile executor for the congruent post-pilot evidence receipt."""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

from tycho_nooa import context_execution_v2 as _v2
from tycho_nooa.sterile_execution import SterileExecutor

ContextExecutionManifest = _v2.ContextExecutionManifest


class ContextComputationalSterileExecutor[ReturnT](SterileExecutor[ReturnT]):
    """Start the versioned worker without altering completed experiments."""

    async def __aenter__(self) -> ContextComputationalSterileExecutor[ReturnT]:
        if self._process is not None:
            raise RuntimeError("sterile executor is already running")
        if self._provenance is not None:
            self._provenance.start(self._manifest.model_dump(mode="json"))
        self._scratch = tempfile.TemporaryDirectory(prefix="tycho-codeact-")
        worker = Path(__file__).with_name("_context_computational_worker_v3.py")
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
