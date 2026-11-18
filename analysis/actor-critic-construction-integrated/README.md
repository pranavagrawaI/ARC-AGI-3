# actor-critic-construction-integrated

Local join of v11 Actor+critic with construction. Not a Kaggle package and not a competence claim.

## Acceptance (this identity)

1. Prefix is worker `ipc/transitions.jsonl` (`host_contract.py` executed transitions). Not `states.jsonl` zip.
2. Live `constructor.run_construction` with injected `evaluate_visible_abi`. Not `integration_boundary.observed()` / LS20 bar.
3. Freeze is host-owned `FROZEN.json` after sealed visible ABI replay. Label: visible replay only; prospective not run; not a supported model.
4. Play path is v11 `submit_actions` via `headless.py` wrapping incumbent headless with `ConstructionAwareAgent`. GameplaySolver is not the player.
5. Critic invocation in incumbent `solver_agent.py` is unedited. Critic packets contain no constructor diagnosis.
6. Candidate execution is unchanged `SealedBoundary` / `CandidateRPC`.

## Command

Scripted/loopback transport, not live model evidence. Requires mapped-root user namespace:

```
unshare --user --map-auto --map-root-user deployment/kaggle/.venv/bin/python -m pytest analysis/actor-critic-construction-integrated/tests -q
```

## Limitations

- One construction episode per game after a nonempty executed prefix. Empty prefix is not capped and may retry.
- Donor ConstructionLimits wall default 45s is not a hard Actor-60 bound: remaining<=20 stops the donor loop, but sealed RPC/HTTP can still overrun. A separate ARC_CONSTRUCTOR_HARD_SECONDS (default 50) join-timeout is measured; it is not ConstructionLimits.wall_seconds.
- Loopback HTTP and FixtureTransport are not live models.
- Visible ABI replay is not sealed-suffix prospective support and not gameplay utility.
- Failed construction (`NEEDS_EVIDENCE`, sealed failure) leaves no freeze; Actor still plays.
- Incumbent files are not modified; this identity is a new overlay.
