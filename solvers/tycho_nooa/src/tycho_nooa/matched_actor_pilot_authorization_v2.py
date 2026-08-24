"""Pilot-specific authority value over the proven durable ledger protocol."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from tycho_nooa.compatibility_authorization_v2 import CompatibilityAuthorization


class MatchedActorPilotAuthorizationV2(CompatibilityAuthorization):
    """The pilot alone earns a 72-call global ceiling; old schemas stay frozen."""

    schema_version: Literal[1] = 1
    maximum_calls: Literal[72] = Field(default=72)
