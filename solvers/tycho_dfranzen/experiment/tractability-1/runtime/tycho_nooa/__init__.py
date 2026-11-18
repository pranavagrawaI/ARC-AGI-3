# Modified for the Tycho/dfranzen candidate; see canonical-candidate.patch.
"""Tycho's native workspace-tool flow implemented with NOOA."""

def __getattr__(name):
    if name == 'TychoNooaAgent':
        from tycho_nooa.agent import TychoNooaAgent
        return TychoNooaAgent
    raise AttributeError(name)
from tycho_nooa.models import ActionDecision, GameTurn

__all__ = ["ActionDecision", "GameTurn", "TychoNooaAgent"]
