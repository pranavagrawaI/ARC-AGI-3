# Resume of the authorized speed plan

The handoff's "nothing launched" statement was stale. The existing screen was
pushed at 2026-09-30 16:09:04 UTC as `duck-flash-next-speed-screen` version 1.
Its notebook, driver, sessions and card remain unchanged. Remote cells match.

The launched card permits greedy mismatches under a noise allowance; the user's
NEXT-PLAN requires identical greedy outputs. Retain both verdicts. `policy.py`
requires exact successful responses on all four prompts in both attempts before
admitting prefix caching. For S3 with caching, also require its greedy responses
to match. Missing responses, worker flags, align evidence or windows cannot pass.

The collector verifies remote source/version and original frozen hashes, saves
raw outputs, then evaluates the card and user-plan gates. A negative or invalid
screen preserves P and spends no validation quota. A positive screen creates
two concrete P+ notebooks/cards before launching, changing only the selected
serving environment in P1's notebook. At least 6.0 GPU hours must remain.
Each validation has a hard Kaggle timeout of 10,800 seconds (3 GPU hours), no
retry. The source, masks, prompt, history, weights, game horizon and resource
settings otherwise remain P. Selected MTP4 retains the screen's capture size 60.

Validation retention: both complete passes and all 25 games; mean levels >=48.5;
busy generation median >=422.4 tokens/s in each pass. Busy means >=20 running
plus waiting during the 7,920-second gameplay horizon, exactly as P's 384 tok/s
baseline was measured. Operational teardown is reported separately, as for P.
Any missing/invalid evidence preserves P. Adoption is local; competition
submission/publication are separate authorized actions.

Attempt receipts are written before each push. An interrupted/uncertain launch
is never automatically repeated. Source/version drift, a quota shortfall, a
three-hour screen monitoring limit, and collection failure stop continuation
and notify the user. Raw evidence and status persist under this directory and
the original `bench/execution`. Failed runs are preserved. No input or weight
experiments launch before next week's quota.
