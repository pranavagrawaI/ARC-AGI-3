# Field survey 2026-09-28: what could lift KF

Incumbent: KF (`duck-flash-next-kf2` v1) scored 44 levels on public-25 (local 10.15); submitted as 56645750.

## Outside

**Board (CLI, 2026-09-28 15:40Z):**

| Rank | Team | Score |
|---|---|---:|
| 1 | Yi-Chia Chen | 28.34 |
| 2 | Tufa Labs | 27.29 |
| 3 | Daniel Franzen | 21.01 |
| 4 | Lord Han Solo | 20.80 |
| 5 | Tong Hui Kang | 20.53 |
| | Keith Tyser | 9.17 |

On 2026-09-24 the leader was at 19.40. The recipes of the top 5 are not public; the Milestone 2 close (09-30) is the disclosure point. Tong Hui Kang's public `arc3` repo is a leaderboard tracker plus a June CNN sweep, not his solver.

**Scott Le Grand, `taaf-flashnext-sheetu12b-0922`, indexed score 5.19** (Keith's base 3.38). This is Keith's stack with:
- 16 vLLM sequences and a 16K analyzer context (the "c16" profile);
- an `agentfix` module of runtime patches for defects measured in Kaggle transcripts.

Enabled in his shipped config:
- **F1 images:** keep board images only in the newest 2 image-bearing messages, and charge images a flat 120 tokens. Duck's estimator charges them len(b64)/3, 5–8x too much, so reasoning gets evicted. He measured a median of 8 obsolete boards per request, about 26 % of the budget.
- **F3 memory:** a tolerant parser for the world-model labels ("World model (revised):" was dropped by the exact-prefix match), and a game_over no longer wipes the world model.
- **F9 timing:** a timing probe.

Disabled after his "v3 regression hunt": the dedup, ledger, loop guard, ACTION7, stall, result, no-impact and animation fixes. He found that refusals misled the model.

His version has none of our fixes: yield stays 60, and there's no game-over or level-up prompt. So the two are complementary.
Source: `field/scottlegrand_taaf-flashnext-sheetu12b-0922/`.

**muhibullahansir/duck-qwen3-8-anim-base** adds:
- the ACTION7→UNDO mapping patch;
- one nudge after 3 consecutive inspections without an action;
- an MTP on/off A/B.

**Calibration:** anandsingh8687 reports local 5.27 → public 2.37 (2.2x). KF's local 10.15 then implies roughly 4–4.6 public. Scott's 5.19 public implies local roughly 11–13, the same class as KF. Both are one pass each.

**Papers:**
- Rodionov, arXiv 2607.15439: with Codex agents, a textual world model beats a flexible executable one. A verification variant ranks first but costs much more. Capability dominates.
- PRO-LONG, arXiv 2607.20064: programmatic memory, a full structured interaction log searched by code. +18.0 points over a base coding agent on the full ARC-AGI-3 public set, with frontier models.

**Kaggle forum:** the competition forum is not listable via the API (403). GetTopic works by id, but a bulk id sweep got throttled (404s). Don't sweep again.

## Inside (out-kf2 vs out-k2)

- **Time binds:** 13 of KF's 44 levels landed after minute 80, and 8 after minute 100 (K: 7 and 4). This supersedes the 3.6-era "plateau by minute 78" memory for this stack. More thinking per game-minute should convert to levels.
- **ACTION7 is dead:** 6 of 25 games advertise it (ar25, bp35, lf52, sb26, sk48, su15). `action_names.py` has no entry for it, so `to_engine_action('ACTION7')` returns None, and it was executed 0 times. The model mentions it 57–132 times per game in those transcripts.
- **Yields:** 20 % of turns still end without an action (K: 47 %). Median 183 s per turn.
- **No-op actions:** 8 % (266 of 3484).
- **Images:** `MULTIMODAL_CONTEXT=current_grid`, `UPSCALE=4`. Every user message carries its board, and eviction uses the len/3 estimator. This is the same code Scott measured.

## Candidate arms, ranked (all vs KF, same games and caps)

1. **KF + Scott's F1 + F3** (images, memory parser, no wipe on game-over). Measured on this exact bundle by someone else, and it targets a mechanism we can see here. Cheap: a verbatim runtime patch.
2. **KF + ACTION7→UNDO.** 24 % of public games expose it, and there's direct evidence the model wants it. Evidence is mixed: Scott shipped it off, muhib ships it on. Test it alone.
3. **KF + c16** (16 sequences, 16K context). A throughput lever while time binds. Risk: halving the context interacts with F1. Test it after F1.
4. **Programmatic memory (PRO-LONG-style).** Duck already exposes `history` in Python; the missing piece is evicted notes and world models. A bigger build; do it after 1–3.
5. **Progress-aware time allocation.** Needs gateway control of the game order. Not now.

**Noise:** the sd of a one-pass delta is about 4 levels. Arms 1–3 need two passes each, or pooling, to separate a +10–20 % effect.
**Milestone 2:** when the leaders' code appears (from 09-30), their mechanisms outrank everything above.
