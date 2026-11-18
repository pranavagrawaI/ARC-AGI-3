# FLARE stage-3d smoke: result (kernel flare-smoke-3d-20261020, notebook 13:03–13:25Z)

## Ran clean
- Inputs staged in 18.5 min; route bench A gcsfuse 168 MB/s, B staged parts 908 MB/s → B.
- `--verify-weights` OK (94.9 s); weights 72.4 GiB in 5.3 s; healthy 120 s after start. The only
  graph attribute skipped was id 11, PREFERRED_CLUSTER_DIMENSION = 0,0,0.
- Image window (ARC3_HISTORY_IMAGE_KEEP=0 transform): 2,352 requests, max 3 images, none over 4.
- Functional: 6/6 rows returned a `python` tool call with nonempty code and reasoning; no fields
  stripped; image inputs accepted (usage reports multimodal image tokens 400–800).

## Throughput: r = 0.45 against the full-image SGLang M3A reference
| S | responses in 480 s | U_resp | U_overlap | mean latency | errors |
|---|---|---|---|---|---|
| 10 | 70 | 231.0 | 214.1 | 67 s | 0 |
| 14 | 77 | 229.4 | 211.0 | 84 s | 0 |
| 16 | 67 | 231.9 | 208.5 | 108 s | 0 |

Throughput is flat across concurrency. The prefix cache served only 37–42% of prompt tokens: some
requests reused 47–74K tokens, others only the shared 4,662-token system prompt. Flare was therefore
prefilling about 6.4–7.2K uncached tokens/s and decoding only about 230 tok/s. SGLang's replay hit 0.92.
Flare reports 128 GPU snapshot slots, and a 64 GiB host prefix cache holding 57,044 pages + 140 rows.

## Reading under SMOKE3D-CARD.md
r = 0.45 < 1.10 → **KILL** on the pre-registered rule, before the same-workload SGLang reference
(sglang-keep0-ref-20261020, running). That reference decides whether the stripped workload itself
breaks prefix caching for any server, or only for Flare.
