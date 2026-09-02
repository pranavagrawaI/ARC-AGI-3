# 20260901T-public3-continued-play-v7

Question: Can the exact V7 Solver run up to 32 reconciled decisions on each of `ls20`, `s5i5`, and `sb26` through the official offline launcher, cross public level boundaries without hidden Scribe requests, and produce timing evidence for the resumable all-25 route?

Posture: bounded feasibility, action-discipline, and performance qualification on public games only.

Contrast: V4 established the real model route but stopped after one decision per game and exposed 27 Actor calls. V6 is invalid for continued play because a level completion opened a model-backed Scribe lane outside the frozen call cap and then violated the Actor-only evidence rule. V7 replaces only that boundary behavior with a deterministic checkpoint.

Prediction: On the requested physical RTX Pro 6000, the notebook returns `passed`; preserves exactly 96 Solver actions unless a game reaches `WIN` before its cap; makes at most 111 model requests; gives sequence 0 up to six 16,000-token Actor requests and every successor exactly one 4,096-token `take_action`/`required` request; records no Scribe request; reconciles every action with the public successor; writes an exact deterministic checkpoint and regular carry-forward note for every recorded level advance or terminal `WIN`; and closes the model server and workers without residual cleanup. Public progress is desirable but not required for transport qualification; zero progress changes the action-quality decision, not the integrity verdict.

Measure: `/kaggle/working/tycho-nooa-public25/result.json`; `/kaggle/working/tycho-nooa-public25/solver-workspaces`; official recordings under `/kaggle/working/server_recording`; worker metadata; tool and LLM traces; decision request/response/workspace checkpoints; deterministic `level_<N>/level_boundary_checkpoint.json` records and their `notes/level_<N>_insights.md`; worker stdout/stderr; model-server access and lifecycle evidence. Interpret public `levels_completed` only as public progress, not an official competition score.

Cap: three public games, seed 0, at most 32 Solver decisions per game; host-owned resets do not consume that Solver cap and remain inside the official loop's `2N+1` safety backstop; `single` mode with `single_decision_discipline`; 32,768-token context; up to 6 Actor calls on sequence 0, exactly 1 Actor call on each realized successor, at most 111 whole-model calls total, 16,000 then 4,096 output tokens, zero retries, C1 model admission; requested Nvidia RTX Pro 6000; 21,600-second launcher cap and 25,200-second kernel cap; no internet, competition rerun, or competition submission.

Stop: Success when all three sessions terminate at their cap or public `WIN` and every required artifact closes. Abort before model load or ARC action if the physical accelerator is not the frozen RTX route. Abort on source, dataset, accelerator, request-envelope, boundary-policy, checkpoint, trace, protocol, lifecycle, or public-recording mismatch; timeout; OOM; malformed/illegal action; nonzero launcher result; residual cleanup; or any action/model count beyond cap.

Invalid if: Any frozen byte below differs; the accelerator identity differs; V7 is not installed from its exact wheel path without cache in an isolated CPython 3.12 environment; the 31 installed Solver source files differ; ready metadata does not declare `deterministic_level_checkpoint`; a public boundary or `WIN` lacks its exact checkpoint and regular in-workspace note; an undeclared checkpoint exists; any model trace is not Actor/decision; any successor request is not the sole `take_action` tool with `required`; the initial/successor/context/retry budgets differ; a session shorter than 32 decisions lacks public `WIN`; evidence is outside the launcher-owned root; the run enters competition mode; or partial evidence is reported as passed.

Result map:

- Passed with valid actions and any progress: use measured request latency and realized action counts to freeze the V7 all-25 V2 checkpoint contract; do not require a one-kernel projection.
- Passed with zero progress: retain transport/action-discipline qualification, but do not call the Solver competition-worthy; diagnose decisions against the three preserved trajectories before scaling gameplay.
- Physical P100/T4 allocation: classify as allocator mismatch with zero Solver evidence; do not cold-load, play, or weaken the route.
- Model transport, protocol, or lifecycle failure: repair only the failed layer and repeat under a new card.
- Boundary, call-cap, or evidence mismatch: reject V7 promotion and preserve the complete failure.

Outputs: immutable kernel output and remote source copied under this directory with the Kaggle notebook version. No `kaggle competitions submit` command is authorized.

## Frozen candidate

| Artifact | SHA-256 |
|---|---|
| `deployment/kaggle/s0-public3-continued-play-v7-candidate/s0_public3_continued_play.ipynb` | `a5f2304ec504a42eabdeae4259c52d1ce0a118ae61c4267fac98fffc645f5576` |
| `deployment/kaggle/s0-public3-continued-play-v7-candidate/build_notebook.py` | `06682d79d73fc72571b3b5bf3f269f804ea919573411d1ee69933d297133401f` |
| `deployment/kaggle/s0-public3-continued-play-v7-candidate/public25_launcher_probe.py` | `d14531dfdae517e70abd853ae34d5f4fe8e71e4a677666805a8b4b0c14688012` |
| `deployment/kaggle/s0-public3-continued-play-v7-candidate/kernel-metadata.json` | `50fa8b20ac7223115e64f5373ad444535964389c7d5b3683eef24d2263e1c778` |
| V7 wheelhouse manifest | `e440ef5bd8c2e898b7fdbb20d2f0c8528cfbd3dda6f70fdb4b96f8112a942851` |
| V7 source manifest | `f59bc64c7454855197c16972ecb5d3a1608f7b5b8b12792fcfea86069bfceaf6` |
| V7 Solver wheel | `175374b1df444fb47c4ce6c37e656b3c7d9ab8418ae697a86fa201d7276f9580` |
| `deployment/kaggle/runtime/local_solver.py` | `bdf1ce780f0635adb8eff690f90a9c1eb3cedd4de3e87a05996579789f324776` |
| `deployment/kaggle/agent/my_agent.py` | `d772f7076c46950daa2cc05c82fdc5d05cb135cd036fb3002df848a2ec961449` |
| `deployment/kaggle/runtime/run_official_agent.py` | `e34b6ec37fee36935846b9f081cea9f57101dc22bd9b86469b9bf654e374a689` |

## Launch prerequisite

Private dataset `pranavagrawai/tycho-nooa-solver-wheelhouse-cp312-v7`, version 1, has been read back with 85/85 required files byte-exact and the sole allowed Kaggle omission `dataset-metadata.json`. The remotely downloaded verifier reported revision 7, 77 wheels, 77 packages, and no failures. Launch may proceed, but an allocator-provided P100 or T4 remains an immediate pre-model invalidation, not permission to change the route.
