# Result: solver-prompt-preservation-local-v1

Status: invalid after the red observation; no qualification claim.

Observation:
- First targeted execution failed in 6.28s at the intended independent assertion with `KeyError: 'request_messages'`, establishing that the copied v10 implementation did not preserve the request.
- The logging implementation was then added.
- Second targeted execution failed in 3.37s before the Actor path started: the immutable candidate guard reported `WorkerError: candidate drift: chassis/solver_agent.py` because `CHASSIS.json` still described the copied v10 source.
- External model calls: 0. ARC environment calls: 0. Spend: $0.

Interpretation:
The red half is valid evidence of the missing contract. The attempted green run cannot interpret prompt preservation because candidate admission failed first. Per the card cap, this run stops rather than silently taking a third pytest execution.

Next action:
Freeze the v11 candidate and use a new immutable card for one targeted qualification execution.
