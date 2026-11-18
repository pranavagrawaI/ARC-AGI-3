"""D-prime priority scheduler + Phase C cross-level rule ledger and per-level action budgeter."""
from __future__ import annotations

import math
import os
import threading
import time
from typing import Any

CHOSEN = {"level_pow": 0.2, "eff_const": 100.0, "pace_pow": 0.25, "pace_ref": 30_000.0,
          "token_scale": 124_000.0, "b1": 2.5, "b2": 3.5, "b3": 4.0}
SCALE = 1000.0


class FormPace:
    """Mean generated tokens per cleared level across the current game."""

    def __init__(self):
        self.completed = 0
        self.tokens = 0.0

    def record_completion(self, tokens_used: float, reference_tokens: float = 0.0) -> None:
        self.completed += 1
        self.tokens += max(0.0, float(tokens_used))

    def cost_multiplier(self) -> float:
        return self.tokens / self.completed if self.completed else 0.0


def _levels(total) -> int | None:
    try:
        n = int(total)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def form_value(level: int, actions: float, tokens: float, pace: float, total_levels,
               params: dict = CHOSEN) -> float:
    level = max(1, int(level))
    n = _levels(total_levels) or 10
    n_clamped = min(10, max(6, n))
    norm = 55 / (n_clamped * (n_clamped + 1) / 2)
    a = max(0.0, float(actions))
    t = max(0.0, float(tokens))
    value = math.exp(params["level_pow"] * (level - 1)) * norm
    value *= 2 ** (-(a / params["eff_const"]))
    if pace > 0:
        value *= min(4.0, max(0.25, (params["pace_ref"] / max(1.0, pace)) ** params["pace_pow"]))
    value *= 2 ** (-((t / params["token_scale"]) ** 2))
    remaining = max(0, n - level)
    bonus = (0.0, params["b1"], params["b2"], params["b3"])[min(remaining, 3)]
    return value * (1.0 + bonus / 8.0)


def form_priority(state, **_ignored) -> int:
    value = form_value(state.level, state.actions, state.tokens,
                       float(getattr(state, "cost_multiplier", 0.0) or 0.0),
                       getattr(state, "total_levels", None))
    return max(1, int(value * SCALE))


D_PRIME = {"level_pow": 0.5, "eff_const": 300.0, "eff_pow": 2.5, "pace_pow": 0.4,
           "pace_ref": 30_000.0, "c_action": 0.1, "action_scale": 115.0, "c_token": 0.9,
           "token_scale": 225_000.0, "lin_floor": 0.1, "rel_pow": 0.5,
           "b1": 10.0, "b2": 14.0, "b3": 16.0, "fade": 0.4}

D_PRIME_FORM = {"fade": 0.4, "level_form": 3, "level_pow": 0.5, "eff_form": 0, "eff_const": 300.0,
                "eff_pow": 2.5, "shape": 3, "c_action": 0.1, "c_token": 0.9, "token_scale": 225000.0,
                "combine": 1, "b1": 10.0, "b2": 14.0, "b3": 16.0, "pace_pow": 0.4, "form_fresh": 0,
                "lin_floor": 0.1, "eff_floor": 0.2, "rel_pow": 0.5}


def d_parts(level: int, actions: float, tokens: float, pace: float, total_levels,
            params: dict = D_PRIME) -> tuple:
    level = max(1, int(level))
    n = _levels(total_levels) or 10
    n_clamped = min(10, max(6, n))
    norm = 55 / (n_clamped * (n_clamped + 1) / 2)
    a = max(0.0, float(actions))
    t = max(0.0, float(tokens))
    A = (1.0 + params["level_pow"] * (level - 1)) * norm
    A *= (params["eff_const"] / (params["eff_const"] + a)) ** params["eff_pow"]
    if pace > 0:
        A *= min(4.0, max(0.25, (params["pace_ref"] / max(1.0, pace)) ** params["pace_pow"]))
    scale = params["token_scale"]
    rel = pace if pace > 0 else params["pace_ref"]
    scale *= min(2.0, max(0.5, rel / params["pace_ref"])) ** params["rel_pow"]
    floor = params["lin_floor"]
    C = (params["c_action"] * max(floor, 1.0 - a / params["action_scale"])
         + params["c_token"] * max(floor, 1.0 - t / scale))
    remaining = max(0, n - level)
    B = (0.0, params["b1"], params["b2"], params["b3"])[min(remaining, 3)]
    return A * C, B


def d_value(level, actions, tokens, pace, total_levels, phi: float = 1.0) -> float:
    u, v = d_parts(level, actions, tokens, pace, total_levels)
    return u + v * phi


def d_priority(state, *, tail_fraction=None, **_ignored) -> int:
    phi = 1.0 if tail_fraction is None else min(1.0, max(0.0, float(tail_fraction)))
    value = d_value(state.level, state.actions, state.tokens,
                    float(getattr(state, "cost_multiplier", 0.0) or 0.0),
                    getattr(state, "total_levels", None), phi)
    return max(1, int(value * SCALE))


# ---- Phase C: Cross-Level Mechanics Ledger & Per-Level Action Budgeter ----

_CANONICAL_CONTROLS = ("UP", "DOWN", "LEFT", "RIGHT", "SPACE", "MOUSE", "UNDO", "RESET")


def _normalize_control_name(raw_action: Any) -> str:
    text = str(raw_action or "").strip().upper()
    if not text:
        return ""
    head = text.split("(", 1)[0].split()[0].strip()
    alias_map = {
        "ACTION1": "UP",
        "ACTION2": "DOWN",
        "ACTION3": "LEFT",
        "ACTION4": "RIGHT",
        "ACTION5": "SPACE",
        "ACTION6": "MOUSE",
        "ACTION7": "UNDO",
    }
    return alias_map.get(head, head)


def extract_cross_level_ledger(
    history_entries: list[Any] | None,
    current_level: int | None,
    valid_actions: list[str] | None = None,
) -> dict[str, Any]:
    """Extract empirical per-level stats and verified control effects from history_entries."""
    target_level = max(1, int(current_level or 1))
    entries = list(history_entries or [])
    level_stats: dict[int, dict[str, Any]] = {}
    control_stats: dict[str, dict[str, int]] = {}
    fatal_actions: list[str] = []

    def _ensure_level(lv: int) -> dict[str, Any]:
        if lv not in level_stats:
            level_stats[lv] = {
                "level": lv,
                "total_actions": 0,
                "attempt_actions": 0,
                "deaths": 0,
                "completed_by": "",
            }
        return level_stats[lv]

    active_level = 1
    _ensure_level(active_level)

    for entry in entries:
        action_raw = str(getattr(entry, "action", "") or "").strip()
        result = getattr(entry, "result", None)
        if not isinstance(result, dict):
            result = {}
        frame = getattr(entry, "frame", None)
        frame_level = getattr(frame, "level", None) if frame is not None else None
        res_level = result.get("level")
        entry_level = None
        for cand in (res_level, frame_level):
            try:
                if cand is not None and int(cand) >= 1:
                    entry_level = int(cand)
                    break
            except (TypeError, ValueError):
                pass

        upper_action = action_raw.upper()
        if upper_action in ("", "START"):
            if entry_level is not None:
                active_level = entry_level
                _ensure_level(active_level)
            continue

        if bool(result.get("automatic")) or upper_action == "RESET":
            if entry_level is not None:
                active_level = entry_level
            st = _ensure_level(active_level)
            st["attempt_actions"] = 0
            continue

        if not result.get("executed", True):
            continue

        lv_before = active_level
        st = _ensure_level(lv_before)
        st["total_actions"] += 1
        st["attempt_actions"] += 1

        ctrl = _normalize_control_name(action_raw)
        if ctrl and ctrl not in ("RESET", "START"):
            cstat = control_stats.setdefault(ctrl, {"effect": 0, "noop": 0, "total": 0})
            cstat["total"] += 1
            gp_changed = result.get("gameplay_changed")
            if gp_changed is True or result.get("level_completed"):
                cstat["effect"] += 1
            elif gp_changed is False or result.get("no_op") is True:
                cstat["noop"] += 1

        if result.get("game_over"):
            st["deaths"] += 1
            st["attempt_actions"] = 0
            tag = f"L{lv_before}:{action_raw}"
            if tag not in fatal_actions:
                fatal_actions.append(tag)

        if result.get("level_completed"):
            st["completed_by"] = action_raw
            if entry_level is not None and entry_level > lv_before:
                active_level = entry_level
            else:
                active_level = lv_before + 1
            _ensure_level(active_level)
        elif entry_level is not None and entry_level != active_level:
            active_level = entry_level
            _ensure_level(active_level)

    cur_stat = _ensure_level(target_level)
    completed_levels = [
        level_stats[k]
        for k in sorted(level_stats)
        if k < target_level and (level_stats[k]["total_actions"] > 0 or level_stats[k]["completed_by"])
    ]

    active_controls: list[str] = []
    inert_controls: list[str] = []
    for ctrl in (*_CANONICAL_CONTROLS, *sorted(control_stats)):
        if ctrl not in control_stats:
            continue
        cs = control_stats[ctrl]
        label = f"{ctrl}({cs['effect']}/{cs['total']})"
        if cs["effect"] > 0 and label not in active_controls:
            active_controls.append(label)
        elif cs["effect"] == 0 and cs["noop"] >= 2 and label not in inert_controls:
            inert_controls.append(label)

    normalized_valid = []
    for va in (valid_actions or []):
        nc = _normalize_control_name(va)
        if nc and nc not in ("UNDO", "RESET", "START") and nc not in normalized_valid:
            normalized_valid.append(nc)
    untested_controls = [c for c in normalized_valid if c not in control_stats]

    return {
        "current_level": target_level,
        "completed_levels": completed_levels,
        "current_level_total_actions": cur_stat["total_actions"],
        "current_level_attempt_actions": cur_stat["attempt_actions"],
        "current_level_deaths": cur_stat["deaths"],
        "control_stats": control_stats,
        "active_controls": active_controls,
        "inert_controls": inert_controls,
        "untested_controls": untested_controls,
        "fatal_actions": fatal_actions[-5:],
    }


def compute_level_budget_metrics(
    level: int,
    total_levels: int | None,
    total_level_actions: int,
    attempt_actions: int,
    deaths_on_level: int,
    non_exec_streak: int = 0,
    single_step_streak: int = 0,
    tokens_on_level: int = 0,
    valid_actions: list[str] | None = None,
) -> dict[str, Any]:
    """Compute per-level RHAE action budget status and anti-stall / search directives."""
    lv = max(1, int(level or 1))
    n = max(lv, _levels(total_levels) or 10)
    total_weight = n * (n + 1) // 2
    remaining_levels = max(0, n - lv)
    remaining_weight = sum(range(lv + 1, n + 1))
    remaining_weight_pct = int(round(100.0 * remaining_weight / max(1, total_weight)))

    ref_budget = min(65, 18 + 7 * (lv - 1))
    a_used = max(0, int(total_level_actions))
    att_used = max(0, int(attempt_actions))
    if a_used <= ref_budget:
        est_eff_pct = 100
    else:
        est_eff_pct = max(1, min(100, int(round(100.0 * ((ref_budget / float(a_used)) ** 2)))))

    norm_valid = {_normalize_control_name(v) for v in (valid_actions or [])}
    has_undo = "UNDO" in norm_valid

    directive_kind: str | None = None
    directive_text = ""

    if non_exec_streak >= 2 or (a_used == 0 and tokens_on_level >= 8000):
        directive_kind = "anti_stall"
        directive_text = (
            f"ANTI-STALL DIRECTIVE: You have spent {max(non_exec_streak, 1)} consecutive "
            f"analyzer turn(s) ({tokens_on_level} tokens) on Level {lv} without executing "
            "an action. Do NOT re-inspect `current_frame` from scratch. Call `action(actions)` "
            "in THIS turn with either a Python-computed sequence or a 1-3 step diagnostic probe."
        )
    elif a_used >= 2 * ref_budget:
        directive_kind = "recovery"
        undo_clause = " use `UNDO` to roll back a deadlocked push, or" if has_undo else ""
        directive_text = (
            f"BUDGET RECOVERY DIRECTIVE: You have spent {a_used} actions on Level {lv} "
            f"(reference budget ~{ref_budget}; remaining levels L{lv + 1}..L{n} hold "
            f"{remaining_weight_pct}% of the total game score). Stop manual trial-and-error:{undo_clause} "
            "write a Python state-space search (BFS/DFS) over `current_frame.segmentation` to "
            "verify the win condition and batch the solution."
        )
    elif single_step_streak >= 3 or (att_used >= ref_budget and single_step_streak >= 2):
        directive_kind = "batch_search"
        directive_text = (
            f"BATCH & SEARCH DIRECTIVE: {single_step_streak} consecutive single-action turns "
            f"on Level {lv} ({a_used} actions used vs ~{ref_budget} reference budget, est. RHAE "
            f"credit {est_eff_pct}%). Once action semantics are known, compute the full path in "
            "Python (BFS/shortest-path) and submit the ordered batch in one `action(actions)` call."
        )

    return {
        "level": lv,
        "total_levels": n,
        "level_weight": lv,
        "total_weight": total_weight,
        "remaining_levels": remaining_levels,
        "remaining_weight_pct": remaining_weight_pct,
        "ref_budget": ref_budget,
        "total_level_actions": a_used,
        "attempt_actions": att_used,
        "deaths_on_level": max(0, int(deaths_on_level)),
        "est_efficiency_pct": est_eff_pct,
        "directive_kind": directive_kind,
        "directive_text": directive_text,
    }


def render_phase_c_guidance(
    ledger: dict[str, Any],
    budget: dict[str, Any],
    *,
    is_level_transition: bool = False,
    is_game_over: bool = False,
) -> list[str]:
    """Render concise, prefix-cache-safe lines for the current turn opener."""
    lines: list[str] = []
    lv = budget["level"]
    n = budget["total_levels"]
    lines.append(
        f"Level budget status: L{lv}/{n} (weight {budget['level_weight']}/{budget['total_weight']}, "
        f"remaining levels hold {budget['remaining_weight_pct']}% of score) | "
        f"actions on L{lv}: {budget['total_level_actions']} (attempt: {budget['attempt_actions']}, "
        f"100%-credit ref ~{budget['ref_budget']}, est. efficiency {budget['est_efficiency_pct']}%) | "
        f"deaths on L{lv}: {budget['deaths_on_level']}."
    )

    completed = ledger.get("completed_levels") or []
    active_ctrls = ledger.get("active_controls") or []
    inert_ctrls = ledger.get("inert_controls") or []
    untested_ctrls = ledger.get("untested_controls") or []

    if completed or lv >= 2 or is_game_over:
        parts: list[str] = []
        if completed:
            comp_str = ", ".join(
                f"L{c['level']}:{c['total_actions']}a"
                + (f"(via {c['completed_by']})" if c.get("completed_by") else "")
                + (f"[{c['deaths']}d]" if c.get("deaths") else "")
                for c in completed[-5:]
            )
            parts.append(f"cleared=[{comp_str}]")
        if active_ctrls:
            parts.append(f"effective_controls=[{', '.join(active_ctrls[:6])}]")
        if inert_ctrls:
            parts.append(f"inert_so_far=[{', '.join(inert_ctrls[:4])}]")
        if untested_ctrls and lv == 1:
            parts.append(f"untested=[{', '.join(untested_ctrls[:4])}]")
        if parts:
            lines.append("Cross-level empirical ledger: " + "; ".join(parts) + ".")

    if is_level_transition and lv >= 2:
        lines.append(
            f"Cross-level transfer note: Reuse verified control rules and retained Python helpers "
            f"from L1..L{lv - 1}; focus initial probes only on newly appeared object types."
        )

    if budget.get("directive_text"):
        lines.append(budget["directive_text"])

    return lines


FRESH = {"replaced": 0}
_TL = threading.local()


def install_d(tool_agent, solver) -> str:
    """Install D-prime priority scheduler and Phase C cross-level & budget hooks."""
    for name in ("priority_value", "ProgressPace", "_PriorityGate", "PrioritySnapshot",
                 "_PRIORITY_UNTRIMMED_BASE", "_PRIORITY_BAND", "ToolAgent"):
        if not hasattr(tool_agent, name):
            raise RuntimeError(f"tool_agent missing {name}")
    session_cls = getattr(solver, "_HarnessGameSession", None)
    if session_cls is None or not callable(getattr(session_cls, "play", None)):
        raise RuntimeError("solver missing _HarnessGameSession.play")
    gate_cls = tool_agent._PriorityGate
    for name in ("acquire", "_enqueue_and_wait", "_snapshot_priority"):
        if not callable(getattr(gate_cls, name, None)):
            raise RuntimeError(f"_PriorityGate missing {name}")

    tool_agent.priority_value = d_priority
    tool_agent.ProgressPace = FormPace
    os.environ["ARC3_PRIORITY_PACE"] = "1"
    os.environ["ARC3_PRIORITY_REFRESH_QUEUE"] = "1"
    os.environ["ARC3_PRIORITY_TAIL_FADE"] = "1"
    os.environ["ARC3_PRIORITY_TAIL_FADE_FRACTION"] = str(D_PRIME["fade"])

    if not getattr(session_cls.play, "_ours_d", False):
        orig_play = session_cls.play

        def play(self, *args, **kwargs):
            game = getattr(self, "game", None)
            run = getattr(game, "game_run", None)
            _TL.n = (
                getattr(run, "number_of_levels", None)
                or getattr(game, "number_of_levels", None)
            )
            try:
                return orig_play(self, *args, **kwargs)
            finally:
                _TL.n = None

        play._ours_d = True
        session_cls.play = play

    if not getattr(gate_cls.acquire, "_ours_d", False):
        orig_acquire = gate_cls.acquire
        fresh_floor = tool_agent._PRIORITY_UNTRIMMED_BASE - tool_agent._PRIORITY_BAND
        snapshot_cls = tool_agent.PrioritySnapshot

        def acquire(self, priority):
            if priority < fresh_floor:
                return orig_acquire(self, priority)
            snap = snapshot_cls(1, 0, 0.0, 0.0, getattr(_TL, "n", None))
            value = self._snapshot_priority(snap, time.monotonic())
            with self._cond:
                self._enqueue_and_wait(value, snap)
            FRESH["replaced"] += 1

        acquire._ours_d = True
        gate_cls.acquire = acquire

    agent_cls = tool_agent.ToolAgent
    if not getattr(agent_cls._build_user_prompt, "_ours_phase_c", False):
        orig_build_user_prompt = agent_cls._build_user_prompt
        orig_build_user_message = agent_cls._build_user_message

        def _patched_build_user_prompt(self, *args: Any, **kwargs: Any) -> str:
            base_prompt = orig_build_user_prompt(self, *args, **kwargs)
            try:
                valid_actions = kwargs.get("valid_actions") or []
                current_frame = kwargs.get("current_frame")
                history_entries = kwargs.get("history_entries") or []
                previous_step_summary = kwargs.get("previous_step_summary")

                cur_level = getattr(current_frame, "level", None) if current_frame is not None else None
                if cur_level is None and isinstance(previous_step_summary, dict):
                    cur_level = previous_step_summary.get("level")
                cur_level = max(1, int(cur_level or 1))

                last_lv = getattr(self, "_phase_c_last_level", None)
                if last_lv != cur_level:
                    self._phase_c_last_level = cur_level
                    self._phase_c_non_exec_streak = 0
                    self._phase_c_single_step_streak = 0

                if getattr(self, "_resume_after_yield", False):
                    self._phase_c_non_exec_streak = getattr(self, "_phase_c_non_exec_streak", 0) + 1
                else:
                    self._phase_c_non_exec_streak = 0
                    exec_cnt = 0
                    if isinstance(previous_step_summary, dict) and not previous_step_summary.get("stale"):
                        try:
                            exec_cnt = int(previous_step_summary.get("executed_count") or 0)
                        except (TypeError, ValueError):
                            exec_cnt = 0
                    if (
                        exec_cnt == 1
                        and isinstance(previous_step_summary, dict)
                        and not previous_step_summary.get("level_transition")
                        and not previous_step_summary.get("game_over")
                    ):
                        self._phase_c_single_step_streak = getattr(self, "_phase_c_single_step_streak", 0) + 1
                    elif exec_cnt > 1:
                        self._phase_c_single_step_streak = 0

                total_levels = self._priority_total_levels()
                tokens_on_lv = max(
                    0,
                    int(getattr(self, "_session_generated_tokens", 0))
                    - int(getattr(self, "_tokens_at_level_start", 0)),
                )
                ledger = extract_cross_level_ledger(history_entries, cur_level, valid_actions)
                budget = compute_level_budget_metrics(
                    level=cur_level,
                    total_levels=total_levels,
                    total_level_actions=ledger["current_level_total_actions"],
                    attempt_actions=ledger["current_level_attempt_actions"],
                    deaths_on_level=ledger["current_level_deaths"],
                    non_exec_streak=getattr(self, "_phase_c_non_exec_streak", 0),
                    single_step_streak=getattr(self, "_phase_c_single_step_streak", 0),
                    tokens_on_level=tokens_on_lv,
                    valid_actions=valid_actions,
                )
                self._phase_c_last_budget = budget
                is_lv_trans = bool(isinstance(previous_step_summary, dict) and previous_step_summary.get("level_transition"))
                is_game_over = bool(isinstance(previous_step_summary, dict) and previous_step_summary.get("game_over"))
                extra_lines = render_phase_c_guidance(
                    ledger,
                    budget,
                    is_level_transition=is_lv_trans,
                    is_game_over=is_game_over,
                )
                if extra_lines:
                    return base_prompt + "\n" + "\n".join(extra_lines)
            except Exception:
                pass
            return base_prompt

        def _patched_build_user_message(self, user_prompt: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
            try:
                budget = getattr(self, "_phase_c_last_budget", None)
                if (
                    isinstance(budget, dict)
                    and budget.get("directive_kind") == "anti_stall"
                    and budget.get("directive_text")
                    and budget["directive_text"] not in user_prompt
                ):
                    user_prompt = user_prompt + "\n\n" + budget["directive_text"]
            except Exception:
                pass
            return orig_build_user_message(self, user_prompt, *args, **kwargs)

        _patched_build_user_prompt._ours_phase_c = True
        agent_cls._build_user_prompt = _patched_build_user_prompt
        agent_cls._build_user_message = _patched_build_user_message

    line = ("#OURS_FORM ok version=d_prime+phase_c priority_value replaced; fresh-first removed; "
            "cross_level_ledger=on; level_action_budgeter=on; "
            "tail fade " + str(D_PRIME["fade"]) + "; "
            + " ".join(f"{k}={v:g}" for k, v in D_PRIME.items() if k != "fade"))
    print(line, flush=True)
    return line


def install(tool_agent) -> str:
    for name in ("priority_value", "ProgressPace", "_PriorityGate"):
        if not hasattr(tool_agent, name):
            raise RuntimeError(f"tool_agent missing {name}")
    tool_agent.priority_value = form_priority
    tool_agent.ProgressPace = FormPace
    os.environ["ARC3_PRIORITY_PACE"] = "1"
    line = ("#OURS_FORM ok priority_value replaced; pace=mean tokens per cleared level; "
            + " ".join(f"{k}={v:g}" for k, v in CHOSEN.items()))
    print(line, flush=True)
    return line
