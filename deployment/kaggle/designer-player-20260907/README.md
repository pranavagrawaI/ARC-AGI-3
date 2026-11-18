# Designer-player solver candidate

**Hold GPU launch. Local checks pass, but OpenRouter produced no committed
actions; the live behavioral gate has not passed.** See QUALIFICATION.md.

One combined notebook: repaired loop, then the same loop with designer-player
reasoning guidance. Neither arm changes checkpoint, serving kernels, MTP, or
reasoning allowance. New source identity; retained q6 directory untouched.

Use `prepare_candidate.py` and `verify_candidate.py`.
`assess_candidate.py` runs inside the notebook after each arm. The
inherited builder is only a frozen substrate; do not launch its standalone output.

```
python3 deployment/kaggle/designer-player-20260907/prepare_candidate.py --check
/home/pranav/.local/share/uv/tools/kaggle/bin/python deployment/kaggle/designer-player-20260907/verify_candidate.py
```

The notebook is competition-attached for RTX PRO6000, offline, and private.
`LAUNCH.json` records the proposed command, not an executed upload or launch.
Read the local qualification report before uploading. A built notebook does not
mean OpenRouter or Kaggle behavioral qualification passed.

Mechanisms:

- `self.decision_packet()` returns compact facts, last three grid-change boxes,
  frozen feedback, helper hashes, and the last labelled prospective question.
  Full observation/history remain accessible.
- `self.set_experiment(question, hypotheses, action, outcomes)` appends a short
  prospective record. It is not an action tool, goal oracle or rigid action gate.
- Worker stops a queued plan on mismatch or unresolved frozen-encoder evaluation.
  Separate checker has bounded memory/CPU/output/wall time and parent-death
  termination. It is not a new OS security sandbox.
- `chassis/designer_player.md` adds objective framing and revision triggers;
  designer intent is explicitly provisional. No mandatory per-turn essay.

Sources for policy: existing interactive-game-solver skill, board53/44/119,
Simon candidate plan; no game solution or environment source enters instructions.
Neither zero levels nor better wording establishes the other's cause.
