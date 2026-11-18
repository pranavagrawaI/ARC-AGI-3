# WORLD-MODEL-UTILITY — once-only run ended HOLD

## Observation

The released run executed once and exited2: `provider/accounting stop; no more spend`. CONSUMED and every raw receipt remain intact. No retry, fallback, extra call, source/control edit, cancellation, or submission occurred.

| Phase | Outcome | New actions | Official levels | Known actual cost |
|---|---|---:|---:|---:|
| Constructor | Typed freeze after4 inspections and2 submissions, with one syntax/serialization repair | n/a | n/a | $0.045218654 |
| Control | Interrupted while waiting for request20; last official state NOT_FINISHED |13 (21 including prefix)|0|$0.125239356 |
| Treatment | Never started |not run|not observed|$0 |

Total known actual cost: **$0.170458010** from25 validated responses. One additional dispatched control request has **unknown actual cost**, with **$0.345637232 unresolved reserved exposure**. Known plus reserved exposure is **$0.516095242**, not a statement of actual total billing. No complete response/hash or zero bill is fabricated for the interrupted call.

Constructor reasoning ON is supported by outgoing requests and4714 reported reasoning tokens. Control OFF is supported by outgoing requests and zero reasoning tokens/no payload on its19 completed responses; the twentieth response is unresolved. The source hash is `fd46b2b4db763deeca910161920adabf41dd3e4d1eceec3a14c7bd8e111fd107`. The first submission failed with an unclosed-parenthesis syntax/serialization error; the second passed typed visible replay. This is executable construction, not predictive accuracy.

**Query exposure: zero.** Treatment never started, so its forced initial RIGHT query was never attempted. Constructor visible replay is not actor query exposure. No prospective query accuracy or model utility result exists.

## Actual stop and cleanup

Control's twentieth request returned HTTP200 headers, then its response body remained incomplete. The absolute watchdog interrupted I/O at **1805.008783 active seconds** for the1805s phase hard ceiling. The request's recorded latency was **1375.877264s**. SUPERVISION records hard_timeout:true, SIGKILL returncode-9 and reaped:true. The provider parent exited. Unknown usage stopped all further admissions; treatment was never entered. This does not claim provider-side cancellation.

The host's file wall timestamps span **25177.284s**, substantially more than the supervisor's active-clock duration; the observed host boot-time/monotonic offset confirms accumulated suspend time exists. These observations are consistent with host suspend/clock effects, but do not identify why the provider body failed to complete. The1805s result is an active-clock stop, not a claim that less than1805 calendar seconds elapsed.

All21 executed transition grids/states/level counts match the corresponding22-row official recording (initial reset plus21 actions). Last state is NOT_FINISHED with0 completed levels. The killed worker did not close/publish a final SCORECARD.json; **official final score is unavailable, not inferred**. No clean control result was promoted. Treatment outcomes are absent, not zero-valued observations.

## Interpretation / separate design limitation

**The utility comparison was not completed. There is no positive or negative world-model efficacy finding.** A successful constructor freeze and a partial control trajectory cannot establish whether query access helps.

The PM's read-only budget concern was confirmed separately: parent admission requires used prompt tokens + serialized outgoing UTF-8 bytes +4096 framing <=1500000. At19 completed control requests,552870 prompt tokens were used; the current request reserved53288, and the observed mean was29098 tokens/request. Holding those quantities fixed projected about50 total control requests, not100 actions. This was a projection, not an exact future cutoff. A parent-side cumulative-token rejection would cause a nonclean control stop and HOLD before treatment. That limit was **not reached**: the observed stop was the incomplete twentieth response. No frozen behavior was altered to rescue the comparison.

## Evidence / allowed next action

`SUMMARY.json`, `RECEIPT-AUDIT.json`, `EXECUTION.json`, and `EVIDENCE-HASHES.json` bind the outcome. Raw evidence lives under `../live/`, particularly `RESULT.json`, `ledger.json`, `control/SUPERVISION.json`, `control-raw/020.receipt.json`, constructor root FROZEN/RESULT and child construction/tool events, control transitions/action joins and the official recording. The repaired inventory remains `ac86862b5de9ddb3598205580417655ce3349ba3bc9309ee793e9b15892ea71b`.

Allowed next action: independent read-only outcome and operational review. Preserve CONSUMED and all evidence. No resume, retry, repair, new pair, submission, or R10b clearance is implied.
