# FLARE stage-3e smoke: card (frozen before launch, 2026-10-20)

## Why
3d (RESULT.md there) ran clean at r = 0.45 (U_overlap 214 vs 477.6) with a prefix-cache hit of 0.37–0.42.
Per-request analysis: misses are not capacity eviction (a hit and a miss both come about 2 s after the
same game's previous response). Cached lengths equal whole earlier prompts, so Flare snapshots at
request boundaries. The moving 4-image window rewrites the previous turn's image every new turn, so the
next prompt diverges inside the last snapshot and only the 4,662-token system-prompt snapshot matches.
Follow-ups within a turn keep their images unchanged, and they hit.

## Change (one)
Every image part becomes the fixed text "[grid image omitted from history]", so prompts are prefix-stable
across turns: a no-image harness (MULTIMODAL_CONTEXT unset) would look like this. S=10 and S=14 only.
This measures Flare's throughput ceiling with its cache able to work. It is not a candidate
configuration until a quality screen shows no images doesn't hurt levels.

## Decision
- r = S=10 U_overlap / 477.6 (the full-image SGLang M3A reference). The prompts are shorter still, so the
  bias favours Flare even more than in 3d.
- r < 1.10 → **FLARE closed**: even with its cache working and the shortest prompts, it isn't faster.
- r ≥ 1.60 → FLARE has a path. Next: a same-workload SGLang no-image reference and a quality screen of
  no-image vs F1x on SGLang (useful in its own right).
- 1.10 ≤ r < 1.60 → a same-workload SGLang no-image reference before deciding anything.
- Reported: cached_fraction_block (should rise toward SGLang's ~0.9 if the snapshot diagnosis is right;
  if it doesn't, the diagnosis is wrong, and that is recorded).
