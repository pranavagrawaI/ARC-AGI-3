# Milestone 2 notebook audit — 2026-10-01

The strongest reuse candidate is the **serving and context-retention package**, especially lordhansolo's published vLLM runtime. All three donors have working FP8 KV in downloaded logs, roughly one million or more KV-token slots, and history policies that preserve a reusable prefix between large trims. This changes the feasibility of long-context Duck. It does not establish that any one feature caused their leaderboard results.

For our next bet, qualify the published lordhansolo stack and its context policy, retaining P as the comparison and KF as fallback. Do not transplant the P pruning plugin into this new runtime without checking its different model/loader interfaces. Smaller independent candidates are function/module retention and programmatic animation access. No GPU run, submission, publication, or change to our solver was performed in this audit.

## Evidence and scope

Live Kaggle CLI pulls of all three notebooks match the preexisting local downloads byte-for-byte. Notebook hashes, input identities, Git revision, downloaded-file hashes, demo computations, and a CPU retention probe are in [evidence.json](evidence.json). Recompute with `python3 analysis/m2-winners-20261001/audit/audit_sources.py`.

Primary sources:

- [dfranzen notebook](https://www.kaggle.com/code/dfranzen/arc-agi-3-milestone-2-solution), [write-up](https://github.com/da-fr/arc-agi-3-solution/blob/main/WRITEUP.md), and [serving sources](https://github.com/da-fr/arc-agi-3-solution/tree/main/serving). Repository snapshot: `10882e332d883af3e61616df8c8a17d03a328a75`.
- [lordhansolo notebook](https://www.kaggle.com/code/lordhansolo/arc-agi-3-milestone-2), [actual source bundle](https://www.kaggle.com/datasets/lordhansolo/taaf-kaggle-source), and [runtime dataset](https://www.kaggle.com/datasets/lordhansolo/vllm-main-e975732-arc3). The notebook itself is mostly deployment wiring; reading it alone misses the solver. Source bundle reports commit `ca1bd02`.
- [sirikilohit notebook](https://www.kaggle.com/code/sirikilohit/arc-agi-3-duck-18-1gc-submit). Its executable patch order matters: M90 disables the old reasoning ledger; several earlier comments describe inactive predecessors.

The screenshot's 27.89 / 23.84 / 22.53 scores are supplied leaderboard claims, not scores reproduced here. Franzen also reports 27.89 in his write-up. Tentative winner status was not independently adjudicated. Fresh pulls identify content, not a pinned scored notebook version: do not silently equate latest public source with the exact scored artifact.

## What the source and logs show

| Property | dfranzen | lordhansolo | sirikilohit |
|---|---|---|---|
| Target weights | Intel AutoRound W4A16; BF16 PLE | Mixed NVFP4/FP8 checkpoint | Intel AutoRound W4A16 with FP8 PLE borrowed from NVFP4 checkpoint |
| Runtime | Pennyroyal SGLang 2.5.3 + patches | vLLM e975732 + 32-file pinned overlay | Pennyroyal SGLang 2.5.0 + patches |
| KV dtype / logged capacity | FP8 / 1,011,264 tokens | FP8 / 1,417,100 tokens | FP8 / 1,004,288 tokens |
| Server / harness context | 139,264 / 131,072 | 147,072 / 127,488 | 69,632 / 69,632 |
| History policy | 59,392-token drain; 150 assistant-turn cap with 30-turn drain/coalescing | Soft target 81,536; removes old whole turns after ceiling exceeded; lifts separate turn cap when target is active | High 57,344 → low 45,056; effectively lifts old assistant-turn cap |
| Active server limit | 10 | 14 | 16 |
| Harness admission/allocation | All competition games admitted to a priority gate; ten resident streams | 14 concurrent games | 28 game sessions, shared pool with UCB extensions |
| Code persistence | Eligible functions/imports retained for a game | Explicit saved Python modules, checked against frozen observation tests | Many historical patches; M90 swaps the sandbox and disables ledger. Do not assume M59 remains effective |
| Animation evidence | Deduplicated frames and timelines in Python; transient-pixel hints | Frame views plus compact animation summaries | `transition.frames`, bounded retention, plus selectively attached animation images |

Capacities are runtime allocator reports, not equal-compute benchmarks or a measurement of model quality. Across runtime/weight changes, the accounting and memory layout differ. The larger pools cannot be attributed solely to FP8.

Published demo evidence recomputed from `benchmark.json`:

| Downloaded run | Coverage | Wall time | Levels | Actions | Interpretation |
|---|---|---:|---:|---:|---|
| dfranzen | Ten selected public games | 25.1 min | 40 | 1,327 | Strong short-demo observation; selected subset, not public-25 |
| lordhansolo | 25 entries, only 14 with actions | 15.0 min | 25 | 871 | Short deadline leaves eleven games unplayed; unsuitable as full-suite score |
| sirikilohit | 25 games | 20.0 min | 32 | 1,302 | All terminal states are `cancelled`; bounded demo, not finished competition evidence |

The sirikilohit notebook header still says two games/six minutes, but live code and outputs use 25 games/20 minutes. Prefer executed settings and artifacts over that comment. Empty embedded notebook outputs did not imply absent downloadable logs.

## 1. Reuse the runtime package before inventing more memory workarounds

Our P1 log has 252,658 KV-token slots, against KF's 105,202. P reached 51 and 58 levels in two full public-25 passes; KF/C were 44/42. See [our P report](../../arm-p-full-20260930/REPORT.md). The recent speed screen was INVALID, with S0 baseline invalid; it provides no qualified P+ candidate.

lordhansolo's live log reports 1,417,100 KV-token slots, about 5.6 times P's reported capacity. Over 90 periodic samples in the short demo, median running requests were 14, median waiting zero, and median aggregate generation was 976.4 tokens/s. The final logged prefix hit rate was 94.1%. The benchmark accounts for 839,898 generated tokens over 900.54 seconds, or 932.7/s. These are short-run serving observations, not a controlled speedup over P or proof of late-run performance.

Relevant reusable pieces in [expanded setup](lord-bundle/setup_commands.expanded.txt):

- New runtime supporting FP8 main QSA KV and FP8 indexer KV, with BF16 recurrent state. The old Keith error, “QSA requires a BF16 main KV cache,” is a restriction of that pinned build.
- Fine-grained prefix matching at 128 tokens plus scheduler/worker Mamba alignment overlays. Default vLLM alone is not the donor configuration.
- GDN RecoverSSM avoids allocating per-draft state blocks. The worker/config logs confirm activation.
- CPU offload of `embed_tokens`, in addition to PLE offload; launcher documents about 1.18 GiB reclaimed.
- Profile the vision encoder at the actual 256×256 board-image maximum instead of a huge default image. This trades supported image size for lower activation reservation and more KV capacity.
- Restricted 32,000-token **draft** vocabulary, while target verification uses the full vocabulary; SM120 GEMM changes and correctly sized CUDA graphs.

Downloaded only the small runtime documentation, manifest, and 1.7 MB overlay, not the multi-GB image layers or weights. Overlay SHA256 `e3a6fe0d9f010bc1fb66e53e7dc6fde43272824d5200da5a49b5c4456065e0cb` matches the launcher's pin. Its author-provided validation covers targeted CPU/GPU tests, not full-model bitwise equality; documentation also contains older “not measured” notes. Our downloaded Kaggle log supplies direct evidence of execution, but not an independent reproduction.

**Judgment:** first donor for a vLLM migration. Reproduce its published weight/runtime combination before considering P's pruned weights. Treat a full weight/runtime migration as a bundled candidate; to attribute a speed gain to one runtime change, hold weights and workload fixed. A speed-only result cannot qualify FP8 or new quantization for gameplay by itself.

## 2. Long history and cache reuse must be designed together

The donors do not merely raise a context-limit flag. They count image tokens separately, retain reasoning, and allow history to grow unchanged between coarse trims. This preserves the exact token prefix needed for reuse. Our existing short-history/turn-cap behavior can invalidate that prefix repeatedly.

Franzen additionally preserves the useful final-prefill Mamba checkpoint, avoids filling the recurrent-state cache with intermediate checkpoints, and refreshes its eviction recency after decoding. With a hybrid recurrent model, retaining attention KV without the corresponding recurrent checkpoint does not suffice. His [patch](dfranzen-repo/serving/patches/patch-sglang-mamba-final-prefill-v2.py) is pinned to Pennyroyal; it is not a vLLM patch.

His write-up reports 93.43% token-weighted prompt reuse over 5,132 requests in an **earlier**, slightly different demo run. This is author-reported, not recomputed here. Admission control is also part of the design: a game keeps its slot through tool calls and yields it for priority reconsideration when history eviction already forces a new prefill.

**Judgment:** reuse blockwise trimming plus accurate image accounting with the qualified donor runtime. Do not increase P to 128K on the old pool and expect the same behavior. Preserve thinking; the reported 512-token reserve in lordhansolo is not a 512-token generation cap (`LOCAL_ANALYZER_MAX_OUTPUT=0`).

## 3. Preserve code that worked, without preserving a stale board

Franzen's [retained_functions.py](dfranzen-repo/ARC3-Inference/inference/utils/retained_functions.py) is the smaller transplant. It restores eligible definitions and supported imports, discards ordinary variables, rejects unsafe defaults/unavailable dependencies, and refreshes game state. A CPU probe here retained a board-parameterized helper, obtained correct results on two different inputs, and rejected both an old-board dependency and a captured `current_frame` default. This checks one mechanism, not full sandbox integration or gameplay benefit.

lordhansolo's [python_workspace.py](lord-bundle/src/ARC3-Inference/inference/agent/python_workspace.py) goes further: `save_module`/`load_module`, frozen observation snapshots for regression examples, and validation before replacing stored source. Existing tests rerun on their original snapshots; environment actions are unavailable during validation. Its prompt asks for checks against observations and warns against tautologies. The host enforces checks running and passing, not that the model's chosen tests are scientifically adequate.

**Judgment:** promising for reducing repeated BFS/simulator/helper generation and losing working code when conversation text is trimmed. Start with function retention if seeking a narrow harness change; the tested-module interface is a larger behavioral intervention. Neither has an isolated gain established by these notebooks.

## 4. Expose animation as data, selectively

The source offers a more useful candidate than sending bigger contact sheets: expose intermediate frames to Python so the agent can compare trajectories, collisions, or transient symbols. Franzen gives a special hint when intermediate pixels contain information absent from both endpoint boards; automatic timelines and composites were not in his successful final configuration.

Our G1/G2 contact-sheet/upscale bundle returned 41/40 levels against KF/C 44/42 and failed its gate. That is evidence against blindly repeating that bundle, not proof that selective programmatic animation access cannot help. Franzen's positive description is author experience without a matched ablation; his automatic composite-image experiments were negative or inconclusive.

**Judgment:** a bounded animation-access test is still defensible. Avoid bundling it with a global resolution increase. lordhansolo retains 4× images, so high resolution is not common to all three plausible winners.

## 5. Scheduling and cross-level memory are later candidates

Franzen's priority combines estimated action-efficient immediate reward, remaining-level value, and a penalty for actions/tokens spent without completing a level. Future value fades in the last 20% of the run. He describes calibration from previous level attempts; the resulting priority remains a heuristic. Crucially, handover happens at history eviction, preserving cache residency through ordinary tool calls.

sirikilohit instead uses a shared runtime pool, an initial slice for every game, and UCB extensions based on lifetime/recent weighted progress per generated token. The current slice is 2,400 seconds; old 300-second comments are stale. Admission is still 28 game sessions over 16 server slots. Its solved-level memory stores observed winning action tails separately from the model's proposed rule. The downloaded run reports 32 levels recorded and 27 rules captured, demonstrating that the mechanism fired, not that the rules were correct or improved score.

**Judgment:** first obtain adequate throughput and retention. Allocation on 110 competition games may matter more than public-25, but copying the scheduler simultaneously would make a runtime comparison hard to interpret. P already benefits from sustained reasoning on difficult games; do not replace it with indiscriminate stall resets.

## Things not established, or not worth copying blindly

- **UNDO:** all relevant repairs must reach actual dispatch and useful semantics. Our R repair bundle executed ACTION7 and improved note capture but returned 41 levels, versus C's 42. Franzen reports a competition improvement for UNDO **bundled with** action information, level transfer, and function retention; it is not an isolated UNDO result. Public-25 may still underrepresent its value.
- **Delete world-model memory:** Franzen disables parsed sections with much longer history; sirikilohit disables its old ledger but adds a separate solved-level memory. This does not justify deleting our memory under an unchanged 32K regime.
- **No-op guards:** Franzen's interior-board test excludes a four-cell border. It can mistake meaningful border changes for no progress. His source/write-up acknowledge this limitation; guards need game/action evidence before adoption.
- **10× images:** an author-reported benefit for Franzen, not a universal prerequisite. Our own larger-image bundle failed its gate.
- **Summaries and critics:** these sources do not supply evidence for a new critic. Franzen found no clear gain from automatic summaries or more prescriptive verification.

## Concrete next decision

One bet: qualify **lordhansolo's published runtime plus prefix-preserving history** as a donor stack. First verify startup, actual FP8/cache behavior, correct model/tool parsing, request retirement, and behavior after enough turns to exercise trimming. Then compare a frozen candidate against P at equal full-game wall-clock budgets. Preserve P and KF artifacts. If changing weights, report the result as bundled adoption rather than pure speed causality. Do not launch another instance of the already-invalid speed screen as part of this research request.

Franzen's complete notebook is the strongest alternative reproduction target if the objective becomes adopting the highest reported scorer outright. Reproduce it intact before selectively editing its many coupled controls; its selected ten-game demo is insufficient to promote it over P locally.
