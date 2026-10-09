# Likelihood probe result (2026-10-09)

Kernels `arc3-lprobe-a1` and `arc3-lprobe-b1` ran 04:28–05:53Z and used 9,583 GPU-seconds in total (2.7 h). Both SGLang servers
died of CUDA OOM after 63 and 74 minutes. With `MEMFRAC=0.96` there is no headroom for the 0.8–0.9 GB that prompt
logprobs need. Use about 0.88 if this is ever rerun. Scored: R0 on 241 of 355 items (cd82–su15) and KF0 on 264.
R1, MIN0 and the session repeat never ran. Fidelity: the server's prompt is exactly the local render plus a constant
11 tokens. The template is byte-identical to the local copy.

## Verdict: **drop it.** The probe measures imitation of the policy, not decision quality, and it ranks prompts the wrong way.
Pre-registered checks (PLAN.md), read on the partial data:

| check | result | pass? |
|---|---|---|
| V1 policy (R0, same-kind) | P(taken) 0.455 vs uniform 0.155; top-1 agreement 41%; mean log-lift +0.49 | borderline. It tracks the policy |
| V2 quality (oracle) | AUC 0.76, **but circular**. On the 22 oracle states where the run chose a *non-optimal* action, P(opt) = 0.108 vs uniform 0.119, and the probe preferred the run's mistake in 82% of them | **fail** |
| V2 quality (decisive) | AUC of P(decisive) → solved within 10 min: 0.56–0.60 | fail (< 0.6 same-kind) |
| V3 prompts | R0 − KF0 = **−0.17 nats** (t = −1.2), 49% of items positive, 5/11 games; game-level t = −1.9. Expected positive (hidden 23.87 vs 2.96) | **fail, wrong sign** |

- Swapping the whole system prompt barely moves the probe: item-level Spearman 0.82 between R0 and KF0, and the same top-1 on 60% of items.
  Without reasoning, the model's next action is set by the visible state and history, not the instructions. Prompt quality acts through
  the reasoning, which teacher forcing skips.
- Per game, mean P(target) correlates with levels solved (Spearman 0.54, 11 games). That signal is game ease. It does not separate arms.

## What this rules out
Any sampling-free score of the same model at fixed states, at least without reasoning. Such a score sees what the
policy already does, including its mistakes, so it cannot rank a change that works through deliberation.
The R1 variant (conditioning on the logged reasoning) would only confirm the scorer and was not rerun.
A noise-free metric that is valid would have to include the model's own reasoning rollout, so sampling noise comes back. See `../sprint-power-20261009`.
