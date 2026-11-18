# flash-next — submission candidate for 2026-09-25 (frozen before push)

## Why
Board best is ours at 1.58 (arm F, dense Qwen3.8-27B-FP8). Keith Tyser's public Duck +
Qwen3.8-Flash-Next NVFP4 / MTP3 notebook reports 3.38 on the same harness family (TAAF duck-harness,
ARC3-Inference). Source: tall-humerus analysis/kaggle-approaches-20260925/REPORT.md.
His `serving_setup.py` pins `LOCAL_ANALYZER_YIELD_SECONDS=60` — the yield that ended 45 % of our 3.8
turns with no action — and has neither prompt fix.

## Arms (25 public games, 1 pass, 28-way, 7920 s/game, Keith's notebook gates)
| arm | kernel | source bundle | change |
|---|---|---|---|
| K  | pranavagrawai/duck-flash-next-k  | keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1 (as-is) | none: reproduction/control |
| KF | pranavagrawai/duck-flash-next-kf | pranavagrawai/duck-flash-next-kf-src | K + arm F: tool_agent.py and inference/critic from runs/qwen38-fixes/f bundle (critic off), notebook sets YIELD 300, GAMEOVER_RETRY=1, LEVELUP_REVIEW=1 after setup, before the benchmark pickle imports tool_agent |

serving_setup.py is untouched in KF (it self-hash-checks against SOURCE_IDENTITY.json).
Local check: KF tree imports in isolation, flags resolve to 300/True/True, benchmark unpickles (duck-harness, 7920 s, 28).

## Mechanism checks (KF)
- Log line `FIXES_ENV {...}` printed.
- Yielded-turn share in KF < K's (the yield fix on this stack).

## Submission rule (one slot/day; needs Pranav's go)
Levels on 25 public games, sd of a delta ~4:
- Both ran: submit **KF** unless K beats KF by >= 5 levels; then submit K.
- Only one ran to completion: that one.
- Neither: no candidate today; diagnose from logs.
Also record Keith's score.json from each arm (his local public-demo figure was 6.76; different measurement).

## Cost
Two RTX PRO 6000 sessions, ~2.5–3 h each (135 GB model load + one 7920 s wave). 0 submission slots.

## Status 2026-09-25 15:19Z
Dataset uploaded and verified (marker, watchdog, patches, F tool_agent 94300 B). Both kernel pushes
refused: "Maximum weekly GPU quota of 30.00 hours reached." Quota resets Saturday 00:00 UTC.
Launch with `./launch.sh` after reset. Nothing ran; no GPU hours spent on this card.

## OpenRouter smoke 2026-09-25 16:15Z (openrouter/smoke.py, out-kf-smoke1, kf-smoke1.log)
KF shipped tree against `qwen/qwen3.8-flash` (hf Qwen/Qwen3.8-Flash-Next, Alibaba upstream), sb26 + tr87,
8 min each, 2-way. PASS on wiring: tool_agent loaded from the bundle, yield 300 / retry / review on;
no crash; tool calls parse; sb26 **L1 at step 13** (64 actions), post-level prompt carried the review
text and the model wrote `Cross-level notes:`; game-over suffix rendered. tr87 made 2 actions: 12
upstream 429s (Alibaba shared pool) starved it. Cost $0.08.
Not a level measurement: different host, no preserve_thinking, full-precision weights, rate-limited.
A 25-game pair would cost ~$30 against a $10 key limit and be confounded by 429s — not run.

## Scheduled launch 2026-09-25 16:50Z
User systemd timer `flash-next-launch.timer` fires `scheduled_launch.sh` at 2026-09-26 00:05Z (Persistent,
linger on). It refreshes the Kaggle token, pushes each arm once (marker `launched/<arm>.ok`), retries quota
refusals every 15 min until 04:00Z, then logs status 5 min later. Log: `launched/launch.log`.
Cancel: `systemctl --user stop flash-next-launch.timer`.

## Launch outcome 2026-09-26
- Timer fired at 13:17Z, not 00:05Z: the laptop was asleep, and the timer (Persistent) fired on wake. DNS
  wasn't up yet, the push failed, and RETRY_UNTIL (fixed 04:00Z) had passed, so it gave up. Fixed: the
  deadline is now start+4 h.
- Pushes to ids `duck-flash-next-k` / `-kf` (v1–v4) all died about 10 s in at cell 2: pip said the
  competition wheel dir "is ignored: non-existing path". The mounts were not the problem: probes on the
  same image, as script/notebook and CPU/GPU, all saw the path. `mount-probe/knb` (K's own first cells
  plus a mount listing, full K metadata, new id) ran pip fine. So the failure follows the kernel id.
  Likely cause (unverified): both ids were first created by the quota-refused pushes on 2026-09-25 15:19Z.
- Running arms: `duck-flash-next-k2` (pushed ~22:10Z) and `duck-flash-next-kf2` (~22:14Z). Metadata is
  identical to k/kf except id/title. Both RUNNING past the failure point.

## Result 2026-09-27 (k2 / kf2, both COMPLETE; outputs in out-k2/, out-kf2/)
| | K (Keith as-is) | KF (K + F fixes) |
|---|---|---|
| levels (public 25, 1 pass, 7920 s) | 36 | **44** |
| top game dropped | 33 | 39 |
| games with a level | 22 | 21 |
| local score.json | 5.99 | **10.15** |
| turns acted / yielded | 51 % / 47 % | 77 % / 20 % |
| MTP acceptance, gen tok/s | 60 %, 251 | 60 %, 248 |
Mechanism checks pass: FIXES_ENV printed in KF; yielded share 47 % -> 20 %. Rule: K doesn't beat KF by
>= 5, so **KF is the candidate** (needs Pranav's go). Reference: dense-27B arm F 25 / F2 20 levels on
the same 25 games. Both arms end with a serving_teardown RuntimeError ("did not reach the bounded
terminal gate", 1 GPU survivor) after results are written; the kernel is still COMPLETE and
submission.parquet exists, the same as in Keith's notebook.

## Submitted + incumbent 2026-09-28
KF (`duck-flash-next-kf2` v1) submitted 15:46Z as ref **56645750** (PENDING). By Pranav's decision, KF is the incumbent from now on: new arms clone `kf2/` and use KF as control.

## Arms R and S launched 2026-09-28 (plan: ~/projects/arc-agi-3/analysis/consolidation-20260928/NEXT-STEPS.md)
Both vs KF, public-25, 1 pass, 7920 s, same image/model/runtime; fresh ids.
- **R** `duck-flash-next-r1` v1 (pushed 16:56Z), dataset `pranavagrawai/duck-flash-next-r-src` = KF source + 2 edits:
  `action_names.py` ACTION7→ACTION7 identity (was unmapped → "Unknown action"); `tool_agent._extract_labeled_blocks`
  accepts qualifiers ("(revised)", "update", "L2", "v4", "(verified)") and markdown. Offline replay over out-kf2
  (`r/qualify_r.py` → `r/qualify_r.json`, PASS): world model filled 188→456 of 463 authored blocks, cross-level
  notes 27→104 of 104, 0 regressions; ACTION7 dispatches via real arcengine, ACTION8 still rejected. Readback
  sha256 of both edited files from Kaggle matches local.
- **S** `duck-flash-next-s1` v1 (pushed 16:55Z), KF notebook with profile `kv5-fp8-mtp3-c16-cg64`:
  KV dtype auto→fp8 (same 5 GiB), max_num_seqs 8→16, cudagraph cap 32→64 (MTP3 ⇒ 16 seqs × 4 tokens).
  Rationale: KF ran 5 / waiting ~20 at 88–99 % KV; seqs never bound. Nobody public has run fp8 KV on this stack.
Mechanism checks before levels count: R — executed ACTION7 > 0 in ar25/bp35/lf52/sb26/sk48/su15; S — server
loads, "GPU KV cache size" ≈ 2× 105,202 tokens, median Running > 5, gen tok/s up, tool-call parse errors not up.
Rule: an arm replaces KF only if it beats mean(KF, C) by ≥ max(5, 2·|KF−C|) levels (C = KF replicate, next wave).
- **S failed 16:59Z** (~3 GPU min): `NotImplementedError: Qwen3.8-Flash-Next QSA requires a BF16 main KV cache`
  (vllm/models/qwen3_8_flash_next/nvidia/qsa.py:187). fp8 KV is impossible on this model/runtime — don't retry.
  Log in out-s1/. The KV lever now needs freed VRAM (pruned checkpoint, more PLE offload) or fewer tokens/seq.
- **C** `duck-flash-next-c1` v1 pushed 16:59Z into S's slot: KF notebook byte-identical, only id/title changed.
  First same-stack noise estimate (KF vs C).

## Results 2026-09-29: KF hidden score, C (control), R (repair bundle)
- **KF sub 56645750 COMPLETE: 2.96 public** (F 1.58 → ×1.87; local levels 44/25 = ×1.76, so local level counts
  track the hidden ratio). Local score 10.15 → hidden 2.96 = ÷3.4.
- `readout.py out-c1 out-r1`: KF 44 / **C 42** / **R 41** levels; games with a level 21/21/22; KV, running (3),
  waiting (22), gen tok/s (~248) identical, so the serving stack did not differ between arms.
- **Noise (C is byte-identical to KF):** 16/25 games disagree, tr87 2 vs 5, m0r0 2 vs 0. Per-game diff sd 1.2 →
  **sd of a 25-game total diff ≈ 6 levels**. |KF−C| = 2 was a lucky draw. The rule max(5, 2·|KF−C|) = 5 is ~0.8σ
  and would pass pure noise ~20 % of the time. At 2σ, a single-pass arm needs about +12 levels.
- **R mechanisms both live:** ACTION7 executed in bp35 2, lf52 4, sb26 10, sk48 9, su15 3 (0 in KF/C). The carried
  world model appeared in 511 of 837 turns (KF 315/866, C 220/851), cross-level notes 407 (KF 143, C 130).
  **Levels: R − mean(KF,C) = −2**, ACTION7 games 7 vs 6/6. The repairs work and buy nothing measurable. This is the
  third time memory carriage has failed to predict levels (see duck-memory-audit).
- Verdict: R does not replace KF. S is dead (fp8 KV), so RS = R and should not be run.

## Arm G (graft) — pre-registered 2026-09-29, before launch
**Change:** KF notebook + one inserted cell (after the setup cell): Scott Le Grand's image block from
`taaf-flashnext-sheetu12b-0922` cell 10, verbatim below forced switches. The block is F1 (history images
kept only for the newest 2 messages, flat 120-token image estimate, safety margin 6144), F13 (animation frames
in the sandbox), F19 (contact sheet on the newest turn), ARM P (board image 12×, i.e. 768 px) and F9 timing
(observation only). Every other agentfix switch is OFF (memory, result, loop, noimpact, ACTION7, stall, dedup,
ledger), so KF's F fixes are untouched. Serving is unchanged (c8, 32K). The "16K context" is NOT grafted:
Scott's own run served at 32768, and 863/1045 requests exceeded 16384, so it never applied there either.
Build: `g/build_g.py`. Offline proof: `g/qualify_g.py` (installs on KF source, guard passes) and
`g/e2e_g.py [--control]` (real bp35 + KF session/agent + stub model). G sends 768² board + 792×560 sheet
+ "Frame sheet:" line after an animating action, the sandbox sees 4 frames, and the old sheet is stripped.
KF sends 256² and the sandbox sees 0 frames.
**Runs:** `duck-flash-next-g1` and `-g2`, byte-identical notebooks (two passes), public-25, 1 pass each, 7920 s.
**Control:** KF 44, C 42 → mean 43 (two passes already banked).
**Mechanism checks (must pass before levels count):** "ARM G LIVE" in the log; AGENTFIX_EVENT sheet lines
> 0 in the high-animation games (bp35 lf52 sk48 su15 tu93); no context-length 400s; max AGENTFIX_TIMING
ptok < 32768; running/waiting median not worse than KF (3 / 22).
**Decision rule:** mean(G1, G2) − 43 ≥ +8.5 levels (≈2σ for 2-vs-2 at run sd ≈ 4.2) → G is the submission
candidate (needs Pranav's go). Otherwise KF stays.
**Attribution prediction:** if the sheet/frames help, the gain concentrates in games that animate (Scott's
log: bp35 73/73 actions animated, lf52 92/92, sk48 68/80, su15 55/58, tu93 94/111, g50t 31/41, sb26 33/67,
r11l 26/77). A gain only in non-animating games (wa30, re86, vc33, ka59…) points to 12× / image trimming
instead. Prior: most likely a tie (Scott's full package = KF within noise; community animation p = .92).

## Arm G result 2026-09-29 (g1, g2 COMPLETE 13:04Z)
- **Levels: G1 41, G2 40 → mean 40.5 vs control mean 43 = −2.5.** The rule needed +8.5. **KF stays.**
- Mechanism checks all pass: ARM G LIVE in both; sheet events 313 / 330; 0 context-length errors; ptok median
  ~21K, max 27,606 / 30,369 (< 32,768, but g2 came within 2.4K); running/waiting 3 / 22 as in KF; gen tok/s 258 vs 248.
- Attribution split (actions in parentheses):
  | | KF | C | G1 | G2 |
  |---|---|---|---|---|
  | animating 8 (bp35 lf52 sk48 su15 tu93 g50t sb26 r11l) | 9 (682) | 8 (717) | 7 (730) | 10 (979) |
  | other 17 | 35 (2802) | 34 (2545) | 34 (1710) | 30 (1858) |
  The sheet games show no lift. The **other** games lost ~30 % of their actions (2.5–2.8K → 1.7–1.9K) and 1–5 levels,
  which is the cost of the heavier every-turn board image. So the frames are delivered, but the model does not convert them.
- Scott's sb26/sp80 hint did not replicate: sb26 and sp80 are unchanged from KF in both G passes.

## Arm P (prune-at-load, REAP-k448) — probe 2026-09-30 PASSED
Scope: `~/projects/arc-agi-3/analysis/kv-lever-scope-20260930/SCOPE.md`; code `prune/` (plugin, tests, `build_p.py`).
Probe v1 failed on my own completeness assert (vLLM calls the text model's load_weights once per shard run; the
routers sit in separate BF16 shards). Fixed with cumulative totals checked at the top-level load; multi-call test added.
Probe v2 `duck-flash-next-p-probe` (KF + plugin, KV 12 GiB, max_num_seqs 12, cudagraph 48, 1,200 s/game):
- plugin totals: dropped 36,864 / renumbered 258,048 / routers 48 / 3 calls; 48 blocks at 448.
- Model loading 73.9 GiB (KF 81.8); KV 252,658 tokens (KF 105,202, ×2.4); vLLM max concurrency 7.71× (KF 3.21×).
- First 20 min of load: running median 9 (KF 4), waiting 16 (21), gen tok/s 388 (274, +41 %), MTP acceptance
  58.6 % (58.6 %, unchanged); request latency 71 s (145 s).
- At 1,200 s per game: **14 levels / 613 actions** vs KF 10/432, C 9/391, R 12/455, G1 11/372, G2 12/369.
**Pre-registered full test:** `duck-flash-next-p1`/`-p2` (built, not launched), two passes at 7,920 s.
Rule: mean(P1, P2) − 43 ≥ +8.5 levels → P is the submission candidate. Mechanism: guard LIVE, running median ≥ 7.
