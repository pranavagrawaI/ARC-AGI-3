# Epistemic questioner runtime v8

Successor to the deployed v7 tempo critic. The Actor and critic still use the
same reasoning-off Qwen3.8-27B-FP8 model with separate contexts, and only the
Actor can call tools or spend environment actions.

The critic now asks one source-bound, decision-relevant epistemic question. It
may cite only an explicit ActorEvidence event; host instructions and tool output
remain visible as context but cannot be attributed to the Actor. The three-field
response replaces the six-field critique.

Reviews are event-driven: level advance, prediction-check failure, observed
multi-action sequence, three observed actions, or eight Actor calls since the
last valid review. Triggers coalesce. There is no per-game review-count cap, and
malformed output does not consume a pending boundary. The game admission deadline
still bounds total inference.

This candidate has local deterministic and loopback qualification only. It has
not been run on Kaggle or shown gameplay benefit.
