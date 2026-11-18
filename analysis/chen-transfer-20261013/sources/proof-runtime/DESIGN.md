# proof_agent v2 — streaming pool loop

離線 Kaggle 數學證明 agent loop（32B olmo3_sink，sm120 單卡，每題 1 小時 + 30min grace，
**離線**、輸出 `id,prediction`、**永遠要吐一份 best-available proof**）。

單一量化 32B 用不同 prompt 扮演 **prove → verify → merge-refine → select**，跑在一個
**streaming 引擎**（即時 loop 偵測 + 時間到強制收尾）+ **並發 Gate** 之上。

> 本文檔對齊 2026-06-24 的現行 code。最後一節「已知問題」是 2026-06-24 review 找到、**尚未修**的洞。

---

## 1. 檔案 / 元件

| 檔 | 內容 |
|---|---|
| `agent.py` | `ProofAgentV2` 入口——持有一個 `StreamClient`，每題建 `StreamingEngine` + `ConcurrencyGate` 跑 `solve_pooled` |
| `pool_loop.py` | `PoolSolver`——連續 pool orchestrator（seed/drive/drain/select + realtime events dump） |
| `stream_engine.py` | `StreamClient`（SSE streaming + /generate salvage + abort）、`ConcurrencyGate`、`StreamingEngine`（停止策略 + salvage） |
| `zlib_runaway_detector.py` | **主 loop 偵測**：`RunawayDetector`（zlib 滑窗壓縮比，HARD/SOFT 兩階）。1008 clean 0% FP / 53 loop 100% catch；抓得到逐字 **與語義** loop |
| `loopguard.py` | `find_loop_cut()`（25 字 chunk 局部密集偵測，純函式）——只用於 salvage 的**逐字 cut 精修**；偵測主力已換 zlib |
| reuse 上一層 | `parser.py`（容錯 XML parse + validity）、`bundle.py`（refine/select bundle）、`clean.py`、`rank.py`、`prompts.py` |

---

## 2. 生命週期（`PoolSolver.solve`）

**ACTIVE phase** `[0 .. active_deadline]`（`active_deadline = deadline − reserve`）：
1. seed `init_provers` 個 prover（同一 prompt，多樣性靠 sampling）。
2. prove 完成且 valid → spawn `verify_k` 個 verifier → 進 pool（拿到分數）。
3. **refine driver**（背景）：維持 ~`gen_cap` 個 prove/refine 在飛（gate 內）。pool 有 ≥2 個 verified
   候選 → `_pick_refine_inputs`（**mean → 同分選自然停（非 salvaged）→ 剩下隨機抽**，取 top-`refine_inputs`）
   → merge-refine；材料不足 → spawn 新 prover。
4. refine 完成且 valid → spawn verifier → 重進 pool（可被再 merge，形成 `gen` 世代 + `parents` lineage）。

**SELECT phase** `[最後 select_reserve_s]`：
5. `_drain`：`active=False`、cancel 所有在飛 task（10s timeout）。
6. 取 rank 最高的 `select_bundle_n` 個 **verified** 候選 → `num_selectors` 個 selector 多數決 → final。
7. 無有效票 → 最高分候選；無 verified → 最長且**非退化**的 raw proof；全無 → `fallback_preamble()`（永不空）。

---

## 3. Streaming 停止策略（`StreamingEngine.generate`）

一條 SSE stream（`/v1/chat/completions` `stream:true`，**text streaming**），逐 chunk 累積
`reasoning_content`/`content` 文字；每個 delta 餵進 zlib `RunawayDetector`：

| 條件 | 偵測 | 處置 |
|---|---|---|
| **LOOP** | `RunawayDetector.feed(delta)` — zlib 滑窗（W=12000/step=1000）壓縮比：**HARD** ratio<0.05 即 abort（退化 token loop）；**SOFT** ratio<0.18 持續≥20 窗 → abort（段落/語義 loop） | abort → salvage（`verdict` 帶 onset 供 cut） |
| **TIME** | `content` 還沒出現 `<solution>`（仍在 think）**且** `remaining < finalize_reserve_s`(180)，每 `_CHECK_EVERY_CHARS`(2000) 字檢查 | abort → force-close salvage（注入「given the time…</think><solution>」） |
| **natural** | server `finish_reason`（stop/length） | 原樣返回；mid-solution 被切也有 partial 可用（lenient parser） |

> **為何用 zlib 而非逐字 chunk**：舊 `degenerate()`（25 字 chunk）只抓**逐字**重複，**漏掉語義 loop**（換句話說重試同一 approach，實測 19% 屬此類），且「16k 窗數總次數」會對散布的合法復現/小-case 檢查**誤判**。zlib 的 fuzzy LZ77 + 持續性兩階規則：1008 clean 0% FP / 53 loop 100% catch（列舉/算術 max_low_run 5/5/9 vs 真 loop ≥62）。abort 只截斷 doomed 生成、不改取樣分佈 → on-policy 安全。

- **abort** = 跳出 SSE 迴圈關閉 httpx stream（client disconnect → sglang abort）**＋** best-effort
  `POST /abort_request{rid}`（rid 從首 chunk 抓）。
- **無 est_tps**：force-close 續寫長度用「**這條 stream 實測 char-rate**」算，唯一 wall-clock 常數是
  `finalize_reserve_s`。
- **TIME force-close 順帶解掉 drain-loss**：每個 call 的 deadline = `active_deadline`，還在 think 的
  prover 會在窗尾前自己 force-close 落地，不再整條被 drain 丟。

---

## 4. Salvage（force-close-think）

截斷/loop 的 CoT 救成 `<solution>`：重建 prefix `chat_template([sys,user], add_generation_prompt)`
（結尾 `<think>`）+ encode(乾淨 reasoning + steer) → native `/generate` 續寫。

- **loop 在 content**（已過 `</think>`）：直接截到 loop 起點（乾淨前綴），不續寫。
- **loop 在 reasoning / time-forceclose**：留乾淨 reasoning + steer「…write ONLY the rigorous proof…`</think><solution>`」。
- **cut 點**（`_loop_onset`）：逐字 loop 用 `find_loop_cut`（精準）；語義 loop（無逐字 cut）用 **zlib `verdict` 的 onset 估計**（`position − WINDOW − soft_run×STEP`）。
- **cap** = `_salvage_cap`：`max(_MIN_SALVAGE_TOK=2048, min(_MAX_SALVAGE_TOK=4096, 剩餘時間×live_rate))`
  —— salvage 是「短收尾」不是「重寫」，4096 上限避免續寫又 ramble/loop。
- `_apply_salvage`：對續寫結果跑 **`scan().abort`**（zlib）guard（退化就 `find_loop_cut` 截）。
- native `/generate` **不可帶 seed**（此 build 會 500，`client.py` 已避）。

---

## 5. 並發 Gate（`ConcurrencyGate`）— **實際行為（誠實版）**

```python
gen()    准入：_gen < gen_cap  AND  _tot < total  AND  _vw == 0
verify() 准入：_tot < total          ← verify 沒有 sub-cap
```
- `total`(12) 綁總並發；`gen_cap`(6) 只限 prove/refine。
- **verify 無上限 + 絕對優先**：`_vw` = 排隊中的 verifier 數；只要 `_vw>0`，**所有 gen 被凍結**。
  每個 proof 完成 spawn `verify_k`(3) 個 verifier → verifier 成群湧入。
- → **不是乾淨的「6 gen + 6 verify」**：verify 可佔到 12 slot，gen 在 verify 波期間暫停 →
  pool 在「狂 prove」「狂 verify」間**震盪**。**自我限制**（gen 停 → 無新 proof → 無新 verifier →
  `_vw` 歸零 → gen 恢復），不會永久餓死，但降低總產出。⚠️ 見已知問題 #1。

---

## 6. Realtime events dump

`dump_path + ".events.jsonl"`（append-only，`_emit` 每次 `open(...,"a")+flush`）：每完成一個
- `prove`/`refine`（含 **proof 全文** + stop_reason/salvaged/proof_len/ctok）
- `verify`（score + cand_mean）
- `final`（final_source/selected_id/final_proof）

外部 `tail -F …events.jsonl` 即時看完成的證明，不用等 run 結束。完整 dump（含每候選 verifs + 每 call
時間戳 `t_start/t_end` + `parents` lineage）在 run 結束寫 `dump_path`。

---

## 7. 現行參數（預設）

| 參數 | 值 | 來源 | 意義 |
|---|---|---|---|
| `temperature` | **0.6** | agent.py | prove/refine temp（role_temps 未命中時） |
| `verify_temp` / `select_temp` | 0.6 / 0.6 | agent.py | verify/ 與 select/ 的 temp |
| `top_p` | 0.95 | agent.py | |
| `call_cap` / `max_tokens` | **100_000** | agent.py | 每 call token 上限 |
| `max_concurrent` (total) | 12 | agent.py | 總並發 |
| `gen_cap` | **6** | agent.py | prove/refine 同時上限 |
| `finalize_reserve_s` | 180 | agent.py | 還在 think 且剩 <此 → force-close |
| `init_provers` | 6 | PoolSolver | seed prover 數 |
| `verify_k` | 3 | PoolSolver | 每候選 verifier 數 |
| `refine_inputs` / `select_bundle_n` / `num_selectors` | 4 / 4 / 5 | PoolSolver | |
| refine seed 選擇 | mean → 同分自然停 → 隨機 | `_pick_refine_inputs` | 取 top-`refine_inputs`；同分用 `rng` 隨機抽（`diversity_p`/`refine_min_score` 已棄用） |
| `budget_s` | **4200**（70min） | run.py | 每題 wall-clock（Kaggle 1h + 30min grace） |
| `select_reserve_s` | **900**（15min） | run.py | 尾段保留給 selector |
| salvage cap | 2048–**4096** | stream_engine | |
| **loop 偵測（zlib）** | W=12000 / step=1000 / HARD<0.05 / SOFT<0.18×≥20 窗 | zlib_runaway_detector | 主偵測；validated 0% FP / 100% catch |
| salvage cut（逐字精修） | chunk 25 / step 5 / 門檻 >8 / span 1500 | loopguard | 只用於 cut，非主偵測 |
| `_MAX_CONSEC_ERR` / `_MAX_GENS` | 24 / 2000 | pool_loop | circuit breaker / 絕對 spawn 上限 |

---

## 8. 時間預算 / 1hr 硬上限

- `reserve = min(select_reserve_s, budget × (1−0.7))` → 4200 下 = min(900, 1260) = 900（沒被截）。
- `active_deadline = deadline − reserve` → active 55min、select 最後 15min。
- 每個 call 的 engine deadline = `active_deadline`；selector phase 把它推到完整 `deadline`。
- **最終保險**：`agent.py` 外層 `asyncio.wait_for(coro, budget_s + 120)` → 72min hard slack，撞到回
  `fallback_preamble()`。Kaggle 1h + 30min grace → 70min OK。

---

## 9. ⚠️ 已知問題（2026-06-24 review，**未修**）

依 severity（修法見括號）：

1. **verify 餓死/震盪（gate）**：verify 無 sub-cap + `gen()` 要 `_vw==0` → prove/refine 在 verify 波
   期間凍結，pool 停止長新候選。非 hang（自我限制）但傷產出。`stream_engine.py` ConcurrencyGate。
   （修：verify sub-cap = `total−gen_cap`、硬優先改軟。）
2. **select 可爆 hard cap**：`sel_budget = max(_MIN_SELECT_S=60, deadline−now−30)` 的 **60s 硬下限** →
   貼 deadline 時硬撐 60s → 觸發外層 `hard_timeout` → 回 `fallback_preamble`，**丟掉整小時驗好的 proof**。
   `pool_loop.py:_select_phase`。（修：沒時間直接走 `fallback_top_scored`，別硬撐。）
3. **salvage cap 違反時間預算**：`_salvage_cap` 的 `max(2048,…)` floor 可超過剩餘時間 → 尾段
   time-forceclose 的續寫在慢 server 下 systematically timeout 丟失。（修：剩餘不足直接放棄續寫。）
4. ~~salvage degenerate guard 盲區~~ **已大幅改善**：loop 偵測換成 zlib `RunawayDetector`（抓逐字+語義
   loop、0% FP），salvage guard 改用 `scan().abort`、cut 用 `_loop_onset`（逐字 `find_loop_cut`，語義用
   verdict onset）。殘留：content-path 的逐字 cut 對純語義 content loop 仍可能不精準（但 zlib guard 會擋）。
5. **半死 stream 拖時間**：`<solution>` 出現後 time-forceclose 永久關閉；或 stream 停滯不到 2000 字檢查點
   → 一條 call 拖到 `deadline+30`（httpx timeout）才結束，壓縮 drain/select。（修：時間檢查加「每 N 秒」
   雙觸發 + solution 後仍保留絕對 deadline break。）
6. **記憶體無界**：`self.calls` 留每個 call 完整 `reasoning_content`（force-close 前可數萬字），數百 call
   累積數十~上百 MB，極端下 OOM。（修：parse 完丟 reasoning，只留長度。）
7. **circuit breaker 形同虛設**：要 24 個**連續** error 才開、且任一成功歸零 → server 半死（時好時壞）時
   永遠開不了 → 空耗到 deadline。（修：改看 window error-rate / health 失敗直接 bail。）

> 結論：1hr 硬上限的最終保險還在，但多條路徑會把時間擠到貼 deadline，疊加觸發 `hard_timeout` 就回
> preamble、放棄全部成果（#2 直指此，最該優先修）。

---

## 10. v1 vs v2

| | v1（`proof_agent/{agent,pipeline,pool_loop}.py`） | v2（本目錄） |
|---|---|---|
| 生成 | 非 streaming，一次性 POST | **streaming SSE** |
| 截斷處理 | watchdog 用 `est_tps` 換算 token cap、salvage 只在 `finish=length` | 即時偵測 loop/time → abort → salvage，**無 est_tps** |
| loop 偵測 | 無（生成端） | **即時 zlib `RunawayDetector`**（逐字+語義 loop、0% FP）→ abort |
| 並發 | 單一 sem，verify/refine 共用 | **ConcurrencyGate**（gen cap + verify 優先） |
| 觀測 | 結束才 dump | **realtime events JSONL** |

✅ **提交路徑已走 v2**：`notebook/proof_pilot_submit.ipynb` cell8 呼叫 **`notebook/run_v2.py`** →
`from agent import ProofAgentV2`，參數已是 v2（`--call-cap 100000 --gen-cap 6 --concurrency 12 --temperature 0.6`，
無 est-tps）。budget 4200/900（hidden）、300/60（public example set）。run_v2 pre-seed 每 id fallback +
每題重寫 submission.csv → wall-clock 被砍也留完整 grader-valid 檔。
（舊 `notebook/run.py`＝v1 `ProofAgent`，已不被 notebook 使用。）
