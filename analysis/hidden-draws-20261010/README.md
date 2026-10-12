# Hidden draws: the daily submission as the instrument

Started 2026-10-10. Each daily submission scores 110 hidden games (55 on the public board).
Alternate a candidate with the unchanged control so that every candidate draw has a neighbouring control draw.
Deadline 2026-11-02; merger deadline 2026-10-26.

Prior single draws of the unchanged dfranzen M2 code: 23.87 (ours, 56747536), 27.62 (affectify), 27.89 (dfranzen).
31.47 is the best of N on his page, so it is excluded.
One draw is about ±10%. A candidate needs ≳ +20% on a single pair, or a consistent sign over 3–4 pairs.

| Draw | Date (UTC) | Arm | Kernel / version | Ref | Hidden | Quota used before → after |
|---|---|---|---|---|---|---|
| D1 | 2026-10-10 10:07 | candidate: m2_evidence | pranavagrawai/m2-evidence-pair1-treatment v1 | 57039090 | 26.24 | 0 s → ? |
| D2 | 2026-10-11 00:30 | control: dfranzen raw | pranavagrawai/dfranzen-m2-raw v1 | 57056746 | 29.44 | quota-before-d2 = D1 after |
| D3 | 2026-10-12 (timer 00:30Z, gated) | candidate: B1 + c14 | pranavagrawai/wave2-b1-c14-20261011 v1 | | | |

Rules:
- Submit only versions that already ran COMPLETE remotely with the same code cells. The candidate's TRUE_SUBMISSION branch is byte-identical to raw; the only difference is the overlay cell.
- Record the quota (API `timeUsed`) before and after each draw. D1 started from a fresh week at 0 s with no other kernel running, so its delta answers whether scoring reruns are billed.
- Commit messages go in the submission description, together with the public-25 evidence.

Automation: `submit_draw.py` (idempotent: date guard, duplicate tag, slot check, quota receipt) driven by the
systemd user timer `arc-hidden-draw-d2.timer` (hourly :30 on the day, Persistent). Outcomes go to `draws.log`.
Hold GPU launches until quota-before-d2.json exists, so the D1 billing delta stays clean.

Readout 2026-10-11: two draws of the byte-identical control are 23.87 and 29.44 (+23%); D1 m2_evidence 26.24 lies between them. A single pair cannot resolve less than ~25%.
