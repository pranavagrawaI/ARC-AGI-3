# D9 F1 hidden draw — 2026-10-18 UTC

Question: What hidden-set score does the screened F1 bundle achieve?

Authority: Pranav, 2026-10-17: "It would be of use to use f1 on a hidden set run."
Today's D8 slot was already consumed by the unchanged baseline, so this uses the next daily slot.

Posture: One exploratory hidden draw. F1 passed its mechanism/no-harm public screen
(62 levels versus B1's 65; permitted deficit 7), without demonstrated gameplay benefit.

Candidate: `pranavagrawai/m2-f1-screen1-f1`, version 1, kernel 139171846.
This is original screened F1: B1-fixed API wording plus module preload plus 16K output.
Screened launch notebook SHA256: `2e746dbc5f259d001b7ce8816d62a992cf2d9bbae1afd5374ce3ae54c997bbcd`.
All 13 remote code cells match the screened launch. Executing the actual notebook's
deadline block with TRUE_SUBMISSION=True yields soft_end=None; the 60-minute screen
limit applies only to interactive execution. No new notebook push is needed.

Contrast: No concurrent matched control. Historical unchanged raw-v1 scores include
23.87, 29.44, 24.97 and 29.38. B1 alone scored 26.96. These draws have uncontrolled
variation and cannot establish a causal F1 effect.

Prediction: A small friction intervention may remain within historical variation;
no directional score gain is preclaimed.

Measure: Kaggle ListSubmissions entry tagged `Hidden draw D9 `, exact kernel/version,
terminal status, publicScore and privateScore if exposed. Missing hidden per-game
logs prevent mechanism attribution even if the aggregate score is high.

Cap: Exactly 1 competition submission on 2026-10-18 UTC. 0 separate GPU launches,
0 paid runs, 0 training. Inherit the notebook's standard 32,400-second target budget
and competition per-game cap of 532 minutes; do not extend either. The existing daily
timer invokes the duplicate-tag/remaining-slot-checked submission helper.

Stop: Stop submitting after acceptance or unresolved acceptance; retain receipts.
No automatic additional draw, launch, retry or promotion under this card.

Invalid if: Wrong kernel/version, source drift, competition branch not active,
submission ERROR, or missing score. Preserve failures; do not reinterpret them as
evidence of F1's gameplay quality.

Result map: A valid score >=30 makes F1 eligible for a separately selected confirmation
under the campaign plan; it does not establish improvement. A valid score <30 gives
no compelling reason to promote F1 and preserves the incumbent. An invalid or missing
result preserves the incumbent and triggers read-only diagnosis; no automatic retry.

Known uncertainty: The independently authorized screen-2 F1x ablation is running,
testing the API-wording change after slower early action rates were observed in F1
and the lexicon treatment. That suspicion is unreplicated. This card pins the original
F1 requested in this conversation; it neither selects F1x nor changes that experiment.
Any subsequent candidate substitution needs its own recorded decision/card.

Outputs: `analysis/hidden-draws-20261010/receipt-d9.json`, `quota-before-d9.json`,
`draws.log`, and this directory's `remote/`, `remote-identity.json`,
`VERIFICATION.json`, `QUEUE-RECEIPT.json`, and frozen input hashes.
