"""0033 - Causal trigger memory replay.

Run from the repo root:
    uv run python teach/lessons/0033-causal-trigger-memory-replay.py
    uv run python teach/lessons/0033-causal-trigger-memory-replay.py --live --levels 1,2,3,4,5,6,7 --window 6

0032 proved the spatial micro-skill when given the future contacted trigger:
track it backward, route on the CELL lattice, and compare the final approach to
the trace. That is still a replay oracle: a live agent cannot start at the future
contact and walk identity backward.

This lesson removes that oracle. It gives the agent a tiny causal memory:

    After a supported caused contact is observed, remember the contacted trigger
    under its learned cause clause. On later frames, track that remembered object
    forward by nonzero pixel overlap. When a later trace event wants the same
    cause clause, route to the remembered trigger from earlier-window frames.

This asks the first agent-facing question:

    Once the agent has observed a trigger cause, can it remember and route to that
    causal trigger again without reading the future contacted object?

Still replay-only: the trace supplies evaluation windows and learned-supported
effects, but target recovery for a decision uses only prior observed contacts.
"""

from __future__ import annotations

import argparse
import importlib.util as _ilu
import json
import sys as _sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


ARTIFACT_DIR = Path(__file__).resolve().parent / ".artifacts/0033-causal-trigger-memory-replay"


def _load_0032():
    repo = Path(__file__).resolve().parents[2]
    path = repo / "teach/lessons/0032-earlier-window-routing-replay.py"
    spec = _ilu.spec_from_file_location("ctm33_ewr32", path)
    mod = _ilu.module_from_spec(spec)
    _sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


ewr32 = _load_0032()
ltr31 = ewr32.ltr31
cdr29 = ewr32.cdr29
pp27 = ewr32.pp27
rcf18 = ewr32.rcf18


@dataclass(frozen=True)
class MemoryDecision:
    contact_frame: int
    start_frame: int
    window: int
    desired_effect: tuple[str, str, str]
    cause_clause: tuple[str, ...]
    memory_available: bool
    route_found: bool
    route_actions: tuple[int, ...]
    trace_window_actions: tuple[int, ...]
    first_action_match: bool
    last_action_match: bool
    common_prefix_len: int
    common_suffix_len: int
    exact_match: bool

    @property
    def route_len(self) -> int:
        return len(self.route_actions)

    @property
    def common_prefix_frac(self) -> float:
        return self.common_prefix_len / self.route_len if self.route_len else 0.0

    @property
    def common_suffix_frac(self) -> float:
        return self.common_suffix_len / self.route_len if self.route_len else 0.0


def _memory_match(view, parts: list, matcher: str, oc16):
    if matcher == "overlap":
        return ewr32._match_overlap(view, parts)
    if matcher == "continuity":
        return ewr32._match_continuity(view, parts, oc16.CELL)
    if matcher == "loose":
        return oc16._match_part_view(view, parts)
    raise ValueError(f"unknown memory matcher: {matcher!r}")


def _update_memory(
    memory: dict[tuple[str, ...], object],
    parts: list,
    matcher: str,
    oc16,
) -> dict[tuple[str, ...], object]:
    updated = {}
    for cause_clause, view in memory.items():
        nxt = _memory_match(view, parts, matcher, oc16)
        if nxt is not None:
            updated[cause_clause] = nxt
    return updated


def _snapshot_memory(memory: dict[tuple[str, ...], object]) -> dict[tuple[str, ...], object]:
    return dict(memory)


def _decision_from_memory(
    records,
    contact_frame: int,
    start_frame: int,
    signal: tuple[str, str, str],
    cause_clause: tuple[str, ...],
    target,
    actor_views: list,
    oc16,
) -> MemoryDecision | None:
    trace_window = ewr32._trace_window(records, start_frame, contact_frame)
    if trace_window is None:
        return None
    firing_action = int(records[contact_frame + 1].action_num)
    final_action = ltr31._parse_action_clause(cause_clause) or firing_action
    route = None
    if target is not None:
        route = ewr32.route_to_trigger_grid(
            records[start_frame].frame,
            actor_views,
            target,
            oc16.ACTION_DELTAS,
            records[start_frame].available_actions,
            final_action=final_action,
            cell=oc16.CELL,
        )
    route_actions = route.actions if route is not None else ()
    first, cpl, exact = ewr32.compare_route_to_window(route_actions, trace_window)
    last, csl = ewr32.compare_route_suffix(route_actions, trace_window)
    return MemoryDecision(
        contact_frame=contact_frame,
        start_frame=start_frame,
        window=contact_frame - start_frame,
        desired_effect=signal,
        cause_clause=cause_clause,
        memory_available=target is not None,
        route_found=route is not None,
        route_actions=route_actions,
        trace_window_actions=trace_window,
        first_action_match=first,
        last_action_match=last,
        common_prefix_len=cpl,
        common_suffix_len=csl,
        exact_match=exact,
    )


def evaluate_memory_replay(
    trace: Path | None,
    levels: set[int],
    window: int,
    memory_matcher: str = "overlap",
) -> list[MemoryDecision]:
    events, meta, _counts = rcf18._live_step_events(trace, levels)
    supported = ltr31._supported_effects(events)
    effects = tuple(pp27.LearnedTriggerEffect.from_remote_hypothesis(entry.hypothesis) for entry in supported)
    wanted_by_trace = ewr32._wanted_by_trace(events, meta, effects)

    oc16 = rcf18.rca17._load_oc16()
    records = oc16.load_trace(trace or oc16.DEFAULT_TRACE)
    _game_id, _convention, segments = oc16.hygiene_step0(records)
    decisions: list[MemoryDecision] = []

    for seg in oc16.segments_for_levels(segments, levels):
        belief = oc16.AgentBelief()
        memory: dict[tuple[str, ...], object] = {}
        memory_by_frame: dict[int, dict[tuple[str, ...], object]] = {}
        actors_by_frame: dict[int, list] = {}

        for idx in range(seg.start, seg.end):
            grid = records[idx].frame
            _bg, parts = oc16.il.views_from_grid(grid, frame=idx)
            memory = _update_memory(memory, parts, memory_matcher, oc16)
            actor_views = oc16._actor_views(parts, belief.controlled_sigs)
            memory_by_frame[idx] = _snapshot_memory(memory)
            actors_by_frame[idx] = actor_views

            if idx in wanted_by_trace:
                firing_action = records[idx + 1].action_num
                for _signal, cause_clause in wanted_by_trace[idx]:
                    trigger_shape = ltr31._parse_trigger_shape(cause_clause)
                    contacted = ewr32._contacted_trigger(parts, actor_views, trigger_shape, firing_action, oc16)
                    if contacted is not None:
                        # Learn only after snapshotting this frame: same-frame decisions
                        # cannot use the event they are about to cause.
                        memory[cause_clause] = contacted

            if idx + 1 <= seg.end and records[idx + 1].action_num is not None:
                _bg_t, parts_t = oc16.il.views_from_grid(records[idx].frame, frame=idx)
                _bg_t1, parts_t1 = oc16.il.views_from_grid(records[idx + 1].frame, frame=idx + 1)
                belief.learn_remote_dynamics(parts_t, parts_t1)
                belief.observe_transition(records[idx].frame, records[idx + 1].action_num, records[idx + 1].frame)
                belief.frame_idx += 1
                belief.perceive(records[idx + 1].frame)

        for contact_frame, wanted in wanted_by_trace.items():
            if not (seg.start <= contact_frame < seg.end):
                continue
            for signal, cause_clause in wanted:
                for w in range(0, window + 1):
                    start_frame = contact_frame - w
                    if start_frame < seg.start:
                        break
                    target = memory_by_frame.get(start_frame, {}).get(cause_clause)
                    decision = _decision_from_memory(
                        records,
                        contact_frame,
                        start_frame,
                        signal,
                        cause_clause,
                        target,
                        actors_by_frame.get(start_frame, []),
                        oc16,
                    )
                    if decision is not None:
                        decisions.append(decision)
    return decisions


def _write_artifact(decisions: list[MemoryDecision], levels: set[int]) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    level_slug = "-".join(str(x) for x in sorted(levels))
    path = ARTIFACT_DIR / f"ls20-levels-{level_slug}-memory-decisions.jsonl"
    with path.open("w") as fh:
        for d in decisions:
            fh.write(json.dumps({
                "cause_clause": d.cause_clause,
                "common_prefix_len": d.common_prefix_len,
                "common_suffix_len": d.common_suffix_len,
                "contact_frame": d.contact_frame,
                "desired_effect": d.desired_effect,
                "exact_match": d.exact_match,
                "first_action_match": d.first_action_match,
                "last_action_match": d.last_action_match,
                "memory_available": d.memory_available,
                "route_actions": d.route_actions,
                "route_found": d.route_found,
                "start_frame": d.start_frame,
                "trace_window_actions": d.trace_window_actions,
                "window": d.window,
            }, sort_keys=True) + "\n")
    return path


def _table(decisions: list[MemoryDecision]) -> dict[int, dict[str, float]]:
    by_window: dict[int, list[MemoryDecision]] = defaultdict(list)
    for d in decisions:
        by_window[d.window].append(d)
    table = {}
    for w, rows in sorted(by_window.items()):
        n = len(rows)
        routed = [d for d in rows if d.route_found]
        remembered = [d for d in rows if d.memory_available]
        table[w] = {
            "n": n,
            "memory_rate": sum(d.memory_available for d in rows) / n,
            "route_rate": sum(d.route_found for d in rows) / n,
            "route_given_memory": (
                sum(d.route_found for d in remembered) / len(remembered) if remembered else 0.0
            ),
            "first_rate": sum(d.first_action_match for d in rows) / n,
            "last_rate": sum(d.last_action_match for d in rows) / n,
            "exact_rate": sum(d.exact_match for d in rows) / n,
            "prefix": sum(d.common_prefix_frac for d in routed) / len(routed) if routed else 0.0,
            "suffix": sum(d.common_suffix_frac for d in routed) / len(routed) if routed else 0.0,
        }
    return table


def _check(name: str, ok: bool, detail: str) -> bool:
    print(f"  [{'✓' if ok else '✗'}] {name}: {detail}")
    return ok


def _run_checks() -> bool:
    a = ltr31._FakeView({(0, 0), (0, 1)}, centroid=(0, 0), area=2)
    b = ltr31._FakeView({(0, 1), (0, 2)}, centroid=(0, 1), area=2)
    c = ltr31._FakeView({(5, 5)}, centroid=(5, 5), area=1)
    memory = {("cause",): a}
    oc16 = type("FakeOc16", (), {"CELL": 5, "_match_part_view": staticmethod(lambda view, parts: parts[0] if parts else None)})()
    updated = _update_memory(memory, [c, b], "overlap", oc16)
    checks = [
        _check("1 memory tracks by overlap", updated.get(("cause",)) is b, f"updated={updated}"),
        _check("2 memory drops lost triggers", _update_memory(memory, [c], "overlap", oc16) == {}, "lost becomes absent"),
    ]
    return all(checks)


def run_live(trace: Path | None, levels: set[int], window: int, memory_matcher: str) -> list[MemoryDecision]:
    decisions = evaluate_memory_replay(trace, levels, window, memory_matcher)
    artifact = _write_artifact(decisions, levels)
    print(f"\nLive ls20 causal-trigger memory replay (memory_matcher={memory_matcher}):")
    print(f"  decisions={len(decisions)} window=0..{window}")
    if decisions:
        print(f"    {'w':>2} {'n':>4} {'memory':>7} {'route':>6} {'r|mem':>6} {'first':>6} {'last':>6} {'exact':>6} {'prefixΔ':>8} {'suffixΔ':>8}")
        for w, stats in _table(decisions).items():
            print(
                f"    {w:>2} {int(stats['n']):>4} {stats['memory_rate']:>7.3f} {stats['route_rate']:>6.3f} "
                f"{stats['route_given_memory']:>6.3f} {stats['first_rate']:>6.3f} {stats['last_rate']:>6.3f} {stats['exact_rate']:>6.3f} "
                f"{stats['prefix']:>8.3f} {stats['suffix']:>8.3f}"
            )
    print(f"  memory_artifact={artifact}")
    return decisions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="evaluate causal trigger memory on the recorded trace")
    parser.add_argument("--trace", type=Path, default=None, help="recorded ls20 jsonl trace")
    parser.add_argument("--levels", default="1,2,3,4,5,6,7", help="comma-separated human levels")
    parser.add_argument("--window", type=int, default=6, help="max frames to step back before each trigger event")
    parser.add_argument(
        "--memory-matcher",
        choices=("overlap", "continuity", "loose"),
        default="overlap",
        help="how remembered trigger objects are tracked forward between frames",
    )
    args = parser.parse_args()

    print("Offline checks (causal trigger memory replay):\n")
    if not _run_checks():
        raise SystemExit("\nFix failed checks before live memory replay.")
    print("\nCausal trigger memory can retain and drop tracked trigger objects.")

    if args.live:
        levels = {int(x) for x in args.levels.split(",") if x.strip()}
        run_live(args.trace, levels, args.window, args.memory_matcher)


if __name__ == "__main__":
    main()
