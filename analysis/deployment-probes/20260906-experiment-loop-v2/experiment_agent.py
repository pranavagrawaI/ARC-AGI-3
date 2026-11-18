"""Experimental workflow over the frozen q6 actor; not the queued candidate."""

from typing import Annotated, Literal

from solver_agent import MdArcSolverAgent, hidden, _ReturnResultSignal
from stall_contract import append, read_rows


class ExperimentAgent(MdArcSolverAgent):
    _experiment: Annotated[dict | None, hidden] = None

    def experiment(
        self,
        hypothesis: str,
        alternative: str,
        action: str,
        region: list[int],
        expect_change: bool,
        rationale: str,
    ) -> str:
        """Commit ONE discriminating experiment. Ends this turn on acceptance.

        region=[row_min,row_max,col_min,col_max], inclusive; at most quarter-grid.
        Predict whether ANY pixel in that region will change after this action.
        The alternative must predict the opposite. Explain why this difference
        distinguishes the hypotheses. Use experiment_result then close_experiment
        before another experiment. A rejected prediction is informative, not failure.
        """
        if self._experiment is not None:
            return "REJECTED: inspect experiment_result and close_experiment first."
        state = self.latest_state()
        grid = state["grid_rows"]
        if (
            not isinstance(region, list)
            or len(region) != 4
            or any(type(v) is not int for v in region)
        ):
            return "REJECTED: region needs four integer bounds."
        r0, r1, c0, c1 = region
        if not (0 <= r0 <= r1 < len(grid) and 0 <= c0 <= c1 < len(grid[0])):
            return "REJECTED: region outside grid."
        if (r1 - r0 + 1) * (c1 - c0 + 1) > len(grid) * len(grid[0]) // 4:
            return "REJECTED: choose a specific region, at most quarter-grid."
        if (
            not all(
                isinstance(s, str) and s.strip()
                for s in (hypothesis, alternative, rationale)
            )
            or hypothesis.strip() == alternative.strip()
            or type(expect_change) is not bool
        ):
            return "REJECTED: distinct hypotheses, a boolean prediction and rationale required."
        record = dict(
            step=state["step"],
            turn=state["turn"],
            action=action,
            region=region,
            expect_change=expect_change,
            hypothesis=hypothesis,
            alternative=alternative,
            rationale=rationale,
        )
        self._experiment = record
        # Log before the action notification can wake the host.
        append(self.ipc_dir / "experiments.jsonl", {"event": "PROPOSED", **record})
        try:
            result = super().submit_actions([action], "predict: " + rationale)
        except _ReturnResultSignal:
            append(self.ipc_dir / "experiments.jsonl", {"event": "COMMITTED", **record})
            raise
        else:
            self._experiment = None
            append(
                self.ipc_dir / "experiments.jsonl",
                {"event": "REJECTED", **record, "reason": result},
            )
            return result

    def experiment_result(self) -> dict:
        """Read host-observed pixel differences for the committed experiment."""
        pending = self._experiment
        if pending is None:
            return {"status": "no pending experiment"}
        row = next(
            (
                r
                for r in read_rows(self.ipc_dir / "transitions.jsonl")
                if r["step"] == pending["step"] + 1
            ),
            None,
        )
        if row is None:
            return {"status": "awaiting successor observation"}
        if row["action"] != pending["action"]:
            raise RuntimeError("experiment action/transition mismatch")
        a, b = row["before"]["grid"], row["after"]["grid"]
        r0, r1, c0, c1 = pending["region"]
        changes = [
            [r, c]
            for r in range(len(a))
            for c in range(len(a[r]))
            if a[r][c] != b[r][c]
        ]
        inside = [p for p in changes if r0 <= p[0] <= r1 and c0 <= p[1] <= c1]
        remaining = {tuple(point) for point in changes}
        components = []
        while remaining:
            seed = min(remaining)
            remaining.remove(seed)
            component = [seed]
            frontier = [seed]
            while frontier:
                r, c = frontier.pop()
                for point in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if point in remaining:
                        remaining.remove(point)
                        frontier.append(point)
                        component.append(point)
            components.append(
                {
                    "count": len(component),
                    "bbox": [
                        min(r for r, c in component),
                        max(r for r, c in component),
                        min(c for r, c in component),
                        max(c for r, c in component),
                    ],
                }
            )
        return {
            "status": "observed",
            "step": row["step"],
            "region": pending["region"],
            "hypothesis": pending["hypothesis"],
            "alternative": pending["alternative"],
            "expected_change": pending["expect_change"],
            "actual_change": bool(inside),
            "prediction_matched": bool(inside) == pending["expect_change"],
            "changed_inside": inside,
            "changed_outside_count": len(changes) - len(inside),
            "total_changed_pixels": len(changes),
            "changed_components_complete": components,
            "required_conclusion": "supported"
            if bool(inside) == pending["expect_change"]
            else "rejected",
            "interpretation_limit": "This checks only ANY change in the specified region. It does not establish object identity, direction, inversion or cause.",
        }

    def close_experiment(
        self,
        conclusion: Literal["supported", "rejected"],
        next_hypothesis: str,
        next_test: str,
    ) -> str:
        """Use conclusion="supported" or conclusion="rejected", exactly; no prose.
        Copy experiment_result()["required_conclusion"]. Close the observed loop, matching
        experiment_result. On rejection change the hypothesis. Explain a concrete
        next test; agreement supports a prediction but does not prove its cause.
        """
        result = self.experiment_result()
        if result["status"] != "observed":
            return "REJECTED: no successor observation yet."
        required = "supported" if result["prediction_matched"] else "rejected"
        if conclusion != required:
            return f"REJECTED: observed result requires conclusion={required!r}."
        if not next_hypothesis.strip() or not next_test.strip():
            return "REJECTED: specify next hypothesis and a concrete next test."
        if (
            required == "rejected"
            and next_hypothesis.strip() == self._experiment["hypothesis"].strip()
        ):
            return "REJECTED: revise the contradicted hypothesis."
        append(
            self.ipc_dir / "experiments.jsonl",
            {
                "event": "CLOSED",
                **self._experiment,
                "observed": result,
                "conclusion": conclusion,
                "next_hypothesis": next_hypothesis,
                "next_test": next_test,
            },
        )
        self._experiment = None
        return "Experiment closed. Test the next hypothesis with experiment()."

    def submit_actions(
        self,
        actions: list[str],
        rationale: str,
        predicted_path: list[dict] | None = None,
    ) -> str:
        """Use experiment() to make one predicted action, and close its observed
        result before the next. Raw submission is disabled in this experimental arm.
        """
        return "REJECTED: use experiment(hypothesis, alternative, action, region, expect_change, rationale)."

    def trajectory(
        self,
        per_step: bool = True,
        last_n: int | None = None,
        include_grids: bool = True,
        include_events: bool = False,
    ) -> list[dict]:
        """Read executed host transitions. Each entry includes step, turn, action,
        levels_completed and diff_pixels. With include_grids, before_grid_rows and
        grid_rows are the exact pre/post frames, so no REPL variable must survive.
        Animation intermediates are unavailable on this host and never fabricated.
        """
        if not per_step:
            return super().trajectory(False, last_n, include_grids, include_events)
        entries = []
        for row in read_rows(self.ipc_dir / "transitions.jsonl"):
            a, b = row["before"]["grid"], row["after"]["grid"]
            entry = {
                "step": row["step"],
                "turn": row["turn"],
                "action": row["action"],
                "levels_completed": row["after"]["levels_completed"],
                "diff_pixels": sum(
                    x != y for ra, rb in zip(a, b) for x, y in zip(ra, rb)
                ),
                "event_frame_count": None,
            }
            if include_grids:
                entry["before_grid_rows"] = [
                    "".join(format(v, "x") for v in r) for r in a
                ]
                entry["grid_rows"] = ["".join(format(v, "x") for v in r) for r in b]
            if include_events:
                entry["event_frames"] = []
                entry["animation_available"] = False
            entries.append(entry)
        return entries if last_n is None else entries[-last_n:] if last_n > 0 else []
