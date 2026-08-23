"""The small task interface shared by model-facing reasoning controllers."""

from __future__ import annotations

from typing import Any, Protocol


class TaskDescriptor(Protocol):
    @property
    def task_kind(self) -> str: ...


class ModelTask(Protocol):
    """Supply treatment-visible data and the typed-answer discriminator."""

    @property
    def task(self) -> TaskDescriptor: ...

    def treatment_payload(self) -> dict[str, Any]: ...
