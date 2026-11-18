"""Treatment overlay; the control is the imported current Actor class itself."""
from typing import Annotated

import bootstrap  # noqa: F401
from solver_agent import MdArcSolverAgent, hidden
from nooa.storage.markers import nosnapshot
from nooa.context_blocks import Context

from theory import Store, digest, IntegrityError
from bootstrap import HERE


class OperationalActor(MdArcSolverAgent):
    __doc__ = MdArcSolverAgent.__doc__

    _operational: Annotated[Store, hidden, nosnapshot]

    def __init__(self, *args, operational_store, **kwargs):
        super().__init__(*args, **kwargs)
        self._operational = operational_store
        self.context['operational_cycle'] = Context(
            'TREATMENT: before submitting, call self.workflow_skill() and complete '
            'the required operational cycle. Operational Theory is the sole authoritative '
            'model-authored knowledge asset; notes are scratch, executable helpers are '
            'hash-bound derivatives. Preserve uncertainty. Host acceptance is mechanical '
            'and does not certify game semantics. Existing tool capabilities remain available.',
            prefix=True,
        )

    def workflow_skill(self) -> str:
        """Load the reusable operational cycle and exact API protocol."""
        return (HERE / 'workflow-skill.md').read_text()

    def operational_hash(self, value: dict) -> str:
        """Canonical SHA256 for previous claim payloads and theory bindings."""
        return digest(value)

    def operational_cycle(self, operation: str, payload=None) -> dict:
        """Execute/read the host-enforced operational cycle; see workflow_skill()."""
        return self._operational.call(operation, payload)

    def register_operational_artifact(self, filename: str) -> dict:
        """Bind a helper/procedure file's current bytes for canonical references."""
        if not isinstance(filename, str) or '/' in filename or not filename.endswith('.py'):
            return self._operational.feedback('filename must be a helper basename ending .py')
        path = self.helpers_dir / filename
        if path.is_symlink() or not path.is_file():
            return self._operational.feedback('helper missing or symlink')
        return self._operational.register(filename, path.read_text())

    def submit_actions(self, actions: list[str], rationale: str, predicted_path: list[dict] | None = None,
                       *, search_id: str | None = None, closes: dict | None = None) -> str:
        """Submit only the exact prepared canonical action; ordinary errors can repair."""
        store = self._operational
        if store.phase != 'prepared' or store.pending['actions'] != actions or store.pending['theory_sha256'] != digest(store.current):
            return str(store.feedback('submission requires exact prepared canonical decision'))
        return super().submit_actions(actions, rationale, predicted_path, search_id=search_id, closes=closes)
