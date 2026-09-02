# R8-B2 independent exact-byte review

## Decision

**ACCEPT** the exact six-artifact R8-B2 candidate for preparation. This decision does **not** authorize a Kaggle push, GPU execution, retry, lifecycle repair, S0, or submission.

Reviewer: fresh-context read-only subagent `sa-0-0854cee1` in delegation `deleg_b3a8021b`.

## Candidate identity

| Artifact | SHA-256 |
|---|---|
| `deployment/kaggle/PROTOTYPE-qwen3_8_phase_a_probe.py` | `caf650598a9a15b290efe7cda710947a2d4b3ce092a5999c2019c46b5e14ad2e` |
| `deployment/kaggle/scripts/build_phase_a_probe_notebook.py` | `e96218af6adca1c6bcdfa0f52dd14c64777d8d20fa6dde45848e126168b8c65f` |
| `deployment/kaggle/PROTOTYPE-qwen3_8_fp8-manifest.json` | `2c2c2f2a4344c33d6f54d9f065a4975b91822cd2836d4300ff50f425e0a322a4` |
| `deployment/kaggle/r8b2-notebook/r8b2_probe.ipynb` | `5abc0ca07ed0d481c7316f09848cababba4b43a707a1d108c50d78fb94703734` |
| `deployment/kaggle/r8b2-notebook/kernel-metadata.json` | `2fd13b8dfbdcedf77ee885b6df75b41bcbcf6e0fb84808c74854bdd119ed9d72` |
| `deployment/kaggle/PROTOTYPE-r8b2-qwen38-fp8-sglang-warmup-request-EXPERIMENT-CARD.md` | `0e37b502e1d4c9aacfa14de3eaeea60d1ea547fe91daf9a88b893dd091fb3063` |

All six hashes were recomputed by the reviewer and rechecked unchanged after the review completed.

## Verified properties

1. R8-B2 leaves SGLang's production warmup enabled and waits for the pinned `The server is fired up and ready to roll!` boundary.
2. Before the probe, the log must contain exactly one chat-completion attempt, exactly one successful chat-completion access, one successful `/freeze_gc` access, and one ready marker.
3. After the probe, the log must contain exactly one additional attempt and success: two total attempts, both successful.
4. Total-attempt and successful-access counts fail closed. Warmup invalidity stops before probe dispatch.
5. Post-request count validity is decided before response decoding or parser interpretation.
6. Exact request bytes, served model, tool schema, parsers, backends, and SGLang command are inherited from frozen R8-B.
7. Teardown code is unchanged and lifecycle remains outside the R8-B2 request pass criterion.
8. Fresh notebook generation reproduced byte-identical notebook and metadata; every executable cell compiled.
9. Embedded source and manifest equal the frozen files.
10. The unchanged title and corrected metadata ID resolve to the observed slug `pranavagrawai/prototype-qwen3-8-fp8-r8-b-request-path`; private, no-internet, and RTX PRO 6000 bindings are present.
11. The experiment card binds the exact candidate and grants no launch permission.
12. There is one probe dispatch, no retry loop, no lifecycle repair, and no unrelated treatment change.

Blocking findings: none.

## Prospective hardening and disposition

- The five-second post-response access-log window could cause a false `INVALID` if access-log flushing is unusually delayed. Preserve the accepted bytes. If this edge appears in evidence, create a successor rather than reinterpreting R8-B2.
- Pinned SGLang only warns when `/freeze_gc` fails, whereas R8-B2 invalidates that route. This is intentional fail-closed behavior under the frozen card.
- One timeout message says “successful” while its wait predicate observes total accesses; exact success is checked immediately afterward. Cosmetic only; do not rotate accepted bytes.
- The warmup reader samples once after the ready marker. The pinned implementation orders warmup POST, `/freeze_gc` POST, then the marker sequentially, so this is sufficient for the exact pinned route. Reconsider only if runtime bytes change.

## Maximum licensed claim

The candidate is mechanically ready for a separately authorized R8-B2 qualification run. If that run passes, it will establish only that the R8-A2-qualified Qwen3.8-27B-FP8 and pinned SGLang route on RTX PRO 6000 completed exactly one runtime-owned VLM warmup, then accepted the frozen 2,447-byte visual request and returned one parser-accepted native `take_action({"action":"LEFT"})` tool call, with exactly two chat-completion attempts and successes in total.

It establishes no lifecycle, gameplay, concurrency, throughput, hosted/local parity, S0, or submission claim.

## Next authority boundary

Nothing may run from this decision alone. The next move requires separate explicit user authorization bound to the six hashes above, after a fresh quota and remote-predecessor check, for exactly one private Kaggle push and execution under the existing slug. Evidence retrieval and classification are read-only follow-up. Retry and lifecycle repair require separate successor authority.
