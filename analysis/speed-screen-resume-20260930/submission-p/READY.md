# P submission ready for approval

Submit the already completed `pranavagrawai/duck-flash-next-p1`, **version 1**, output
`submission.parquet`. P1 is a fixed representative of the retained P configuration;
P2's higher stochastic level count is not a reason to choose its trajectory.

`preflight.json` records exact remote source agreement, COMPLETE status and the
local output hash. P's registered comparison passed with 51/58 levels versus
44/42 for KF/C; see `analysis/arm-p-full-20260930/REVIEW.md`.

Concrete command, only after submission authorization:

```bash
kaggle competitions submit arc-prize-2026-arc-agi-3 \
  -k pranavagrawai/duck-flash-next-p1 -v 1 -f submission.parquet \
  -m 'Retained arm P: KF plus load-time REAP-k448, 12 GiB BF16 KV, 12 sequences, MTP3; public-25 validation 51/58 levels, mean 54.5 vs KF/C 43. Fixed P1 v1 source.'
```

Submission spends the daily competition submission allowance. It does not
require a new notebook push. No submission has been performed by this resume.

Milestone 2 additionally requires open source by 2026-09-30 23:59 UTC.
Both P1 and its `pranavagrawai/duck-flash-next-kf-src` dependency are currently
private; the source dataset metadata reports license `unknown`.
Submitting privately alone does not establish prize eligibility. Publication
and licensing need an explicit user decision. Sources checked live:

- https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/713634
- https://arcprize.org/competitions/2026/arc-agi-3
