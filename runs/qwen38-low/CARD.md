# qwen38-low — can brevity buy back the decode advantage?

Frozen 2026-09-22 18:10Z, before push.

## Why this arm exists
Arms 1–2 established: Qwen3.8 decodes ~35 % faster per stream than 3.6 (25–26 vs 18.7 tok/s)
but emits ~70 % more tokens per call (862–941 vs 514), netting ~21 % slower turns and fewer
levels. `reasoning_effort=medium` explained only 19 % of the excess, so the verbosity is the
checkpoint's disposition. `low` is the template's explicit brevity instruction — "Keep your
thinking brief and focused, moving directly to the conclusion without unnecessary elaboration" —
and is the only remaining knob that targets the measured defect directly.

## Question
At `reasoning_effort=low`, does Qwen3.8 produce a turn faster than Qwen3.6's 46.3 s median
without losing levels?

## Change from arm 2 (one knob)
`DUCK_CHAT_TEMPLATE_KWARGS='{"preserve_thinking": true, "reasoning_effort": "low"}'`.
Everything else identical: 5 games x 2 passes, 600 s, 10-way, no critic.

## Decision rule (frozen)
3.8 @ low becomes a submission candidate **only if both**:
1. median s/turn **< 46.3** (strictly faster than 3.6 on the same recipe), and
2. levels **>= 4** over the 10 episodes (3.6's count; both prior 3.8 arms scored 3).

Otherwise Qwen3.6 is retained and the 3.8 line is closed — no further checkpoint arms.
n=10; a one-level difference is noise, which is why the rule requires the latency win too.

## Prediction (before the run)
Brevity cuts output/call below 600: 55 %. Median s/turn < 46.3: 45 %. Levels >= 4: 30 %.
Both conditions: ~20 %. The honest base rate is that two prior knobs moved latency by <2 %.
