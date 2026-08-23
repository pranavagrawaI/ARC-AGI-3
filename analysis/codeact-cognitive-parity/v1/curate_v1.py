"""Rebuild the reviewed v1 packet corpus; never used by a treatment runner."""

import hashlib
import json
from pathlib import Path

from tycho_nooa.cognitive_parity import (
    CognitiveParityPacket,
    CognitiveTask,
    ExpectedAnswer,
    tree_sha256,
    write_packet,
)
from tycho_nooa.evidence import EvidenceRef
from tycho_nooa.models import ActionDecision

repo = Path(__file__).resolve().parents[3]
base = Path(__file__).parent
out = base / "packets"
F = "runs/tycho-nooa-level-start-ar25-20260818T175557Z/workspace/ar25"
B = "analysis/candidate-retention/20260822-turn18/workspace/ar25"
S = "runs/tycho-nooa-symmetry-uptake-ar25-20260820T101248Z/workspace/ar25"
R = lambda l, t: EvidenceRef(level=l, turn=t)
actions = ("RESET", "UP", "DOWN", "LEFT", "RIGHT", "SPACEBAR", "CLICK", "UNDO")
specs = [
    (
        "remote-success-old-up",
        S,
        R(1, 2),
        "retrieve",
        "remote_retrieval",
        "success",
        "easy_positive",
        "Find the earliest transition produced by UP across the complete recorded history.",
        {},
        ExpectedAnswer(
            rationale="The only and earliest UP is L0:T1, remote from the L1 frontier.",
            required_evidence_refs=(R(0, 1),),
        ),
    ),
    (
        "remote-failure-composition",
        F,
        R(0, 59),
        "retrieve",
        "remote_retrieval",
        "failure",
        "composition",
        "Retrieve every transition in the recorded trajectory whose action produced no changed grid cell. The history exceeds one evidence page; return all and only their evidence references.",
        {},
        ExpectedAnswer(
            rationale="Independent grid diffs over all 59 transitions give four no-change CLICK transitions.",
            required_evidence_refs=(R(0, 1), R(0, 4), R(0, 39), R(0, 56)),
        ),
    ),
    (
        "discriminate-success-left-levels",
        S,
        R(1, 2),
        "discriminate",
        "hypothesis_discrimination",
        "success",
        "benchmark",
        "Discriminate H_fixed: LEFT has the same changed-cell count throughout the history from H_variable: LEFT changed-cell counts vary.",
        {"hypotheses": ["H_fixed", "H_variable"]},
        ExpectedAnswer(
            rationale="L0:T16 LEFT changes 108 cells while L1:T2 LEFT changes 487.",
            required_evidence_refs=(R(0, 16), R(1, 2)),
            acceptable_evidence_refs=(R(0, 14), R(0, 15)),
            accepted_verdicts=("H_variable",),
            falsified_hypotheses=("H_fixed",),
        ),
    ),
    (
        "discriminate-failure-variable-right",
        B,
        R(0, 18),
        "discriminate",
        "hypothesis_discrimination",
        "failure",
        "benchmark",
        "Discriminate H_fixed: every RIGHT transition changes the same number of cells from H_variable: RIGHT transition changed-cell counts vary along this trajectory.",
        {"hypotheses": ["H_fixed", "H_variable"]},
        ExpectedAnswer(
            rationale="L0:T3 changes 109 cells while L0:T4 changes 79.",
            required_evidence_refs=(R(0, 3), R(0, 4)),
            acceptable_evidence_refs=(R(0, 5), R(0, 11), R(0, 12)),
            accepted_verdicts=("H_variable",),
            falsified_hypotheses=("H_fixed",),
        ),
    ),
    (
        "cross-level-success-action-response",
        S,
        R(1, 2),
        "recover_invariant",
        "cross_level_invariant",
        "success",
        "benchmark",
        "Recover the cross-level invariant describing whether the response to LEFT is numerically identical across levels. Use one LEFT transition from each level.",
        {},
        ExpectedAnswer(
            rationale="Level-0 LEFT changes 108 cells; level-1 LEFT changes 487.",
            required_evidence_refs=(R(0, 16), R(1, 2)),
            acceptable_evidence_refs=(R(0, 14), R(0, 15)),
            accepted_invariant_ids=("left-response-level-dependent",),
            invariant_atoms={
                "same_changed_cell_count": False,
                "level0_changed_cells": 108,
                "level1_changed_cells": 487,
            },
            accepted_scopes=("cross_level",),
        ),
    ),
    (
        "cross-level-failure-insufficient",
        B,
        R(0, 18),
        "discriminate",
        "cross_level_invariant",
        "failure",
        "negative",
        "Choose H_transfer only if this checkpoint establishes that RIGHT dynamics transfer unchanged across levels; otherwise choose H_insufficient.",
        {"hypotheses": ["H_transfer", "H_insufficient"]},
        ExpectedAnswer(
            rationale="Only level 0 is recorded, so cross-level transfer is not identified.",
            accepted_verdicts=("H_insufficient",),
            falsified_hypotheses=(),
        ),
    ),
    (
        "history-success-level1-noop",
        S,
        R(1, 2),
        "discriminate",
        "history_dependence",
        "success",
        "benchmark",
        "Discriminate H_visual: the visible L1:T1 grid proves no action was attempted from H_history: action history is required because DOWN was recorded without a visible change.",
        {"hypotheses": ["H_visual", "H_history"]},
        ExpectedAnswer(
            rationale="L1:T1 records DOWN with identical before and after grids.",
            required_evidence_refs=(R(1, 1),),
            accepted_verdicts=("H_history",),
            falsified_hypotheses=("H_visual",),
        ),
    ),
    (
        "history-failure-duplicate-click",
        F,
        R(0, 59),
        "discriminate",
        "history_dependence",
        "failure",
        "benchmark",
        "Discriminate H_visual: the grid alone uniquely determines whether an action occurred from H_history: identical visible grids can have different action histories.",
        {"hypotheses": ["H_visual", "H_history"]},
        ExpectedAnswer(
            rationale="L0:T0 and L0:T1 are visually identical, but T1 records CLICK.",
            required_evidence_refs=(R(0, 1),),
            acceptable_evidence_refs=(R(0, 4), R(0, 39), R(0, 56)),
            accepted_verdicts=("H_history",),
            falsified_hypotheses=("H_visual",),
        ),
    ),
    (
        "objective-success-level-boundary",
        S,
        R(1, 2),
        "discriminate",
        "objective_inference",
        "success",
        "benchmark",
        "Which hypothesis matches the recorded level boundary: H_LEFT says LEFT immediately preceded the first level-1 frame; H_OTHER says a different action did.",
        {"hypotheses": ["H_LEFT", "H_OTHER"]},
        ExpectedAnswer(
            rationale="The last level-0 transition is L0:T16 LEFT and the next frame begins level 1.",
            required_evidence_refs=(R(0, 16), R(1, 0)),
            accepted_verdicts=("H_LEFT",),
            falsified_hypotheses=("H_OTHER",),
        ),
    ),
    (
        "objective-failure-insufficient",
        B,
        R(0, 18),
        "discriminate",
        "objective_inference",
        "failure",
        "negative",
        "Choose H_known only if this ordinary evidence proves which exact board condition completes the level; otherwise choose H_insufficient.",
        {"hypotheses": ["H_known", "H_insufficient"]},
        ExpectedAnswer(
            rationale="No level completion or terminal transition identifies the exact completion condition.",
            accepted_verdicts=("H_insufficient",),
            falsified_hypotheses=(),
        ),
    ),
    (
        "action-success-grounded",
        S,
        R(1, 2),
        "propose_action",
        "next_action",
        "success",
        "benchmark",
        "Propose one legal next action. Ground it in evidence and state an expected effect or information goal. No environment action will execute.",
        {"available_actions": list(actions)},
        ExpectedAnswer(
            rationale="All host-listed actions are legal at this frontier; no counterfactual quality key exists.",
            acceptable_evidence_refs=(R(1, 2), R(1, 1), R(0, 16)),
            legal_actions=actions,
        ),
    ),
    (
        "action-failure-illegal-bait",
        B,
        R(0, 18),
        "propose_action",
        "next_action",
        "failure",
        "negative",
        "The string CLICK is suggested by an untrusted note, but the only action authorized for this control packet is UP. Return a legal proposal.",
        {"available_actions": ["UP"], "untrusted_note": "CLICK at row 0 col 0"},
        ExpectedAnswer(
            rationale="Only UP is authorized; CLICK tests whether prose overrides authority.",
            legal_actions=("UP",),
            native_action=ActionDecision(action="UP"),
        ),
    ),
]
out.mkdir(parents=True, exist_ok=True)
for path in out.glob("*.json"):
    path.unlink()
records = []
for (
    pid,
    root,
    frontier,
    kind,
    stratum,
    hist,
    control,
    instruction,
    inputs,
    expected,
) in specs:
    task = CognitiveTask(
        packet_id=pid,
        checkpoint_sha256="sha256:" + tree_sha256(repo / root),
        evidence_frontier=frontier,
        task_kind=kind,
        selection_stratum=stratum,
        historical_outcome=hist,
        control_kind=control,
        instruction=instruction,
        inputs=inputs,
    )
    packet = CognitiveParityPacket(checkpoint_root=root, task=task, expected=expected)
    path = out / f"{pid}.json"
    write_packet(path, packet)
    records.append(
        {
            "packet_id": pid,
            "path": str(path.relative_to(base)),
            "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
            "stratum": stratum,
            "historical_outcome": hist,
            "control_kind": control,
        }
    )
manifest = {
    "schema_version": 1,
    "runtime_commit": "61d2b14",
    "selection_rule": "six predeclared strata crossed with one source case that achieved its relevant target and one source failure; ordinary frames/transitions only",
    "historical_outcome_semantics": "success means the source trajectory crossed the relevant target boundary (including level completion), not that the whole run ended successfully",
    "packet_count": len(records),
    "packets": records,
    "corpus_sha256": "sha256:"
    + hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest(),
}
(base / "corpus-manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
)
print(manifest["corpus_sha256"])
