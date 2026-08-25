"""Authority value for the five-packet Matched Actor V2 pilot."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from tycho_nooa.compatibility_authorization_v2 import CompatibilityAuthorization


class MatchedActorPilotAuthorizationV3(CompatibilityAuthorization):
    """Ten sessions at six calls each; no authority transfers from older pilots."""

    schema_version: Literal[1] = 1
    maximum_calls: Literal[60] = Field(default=60)
