# 0004 LS20 L2 objective acquisition E4B v1

- **Question:** Can E4B acquire the valid objective when prefix machinery leaves seven scale hypotheses?
- **Code:** `453439d`.
- **Contrast:** Machinery acquires levels 1/3/4 but abstains on level 2.
- **Prediction:** Three identical prefix-only calls select pair 32, `integer_scale_objective`, factor 2.
- **Measure:** Contract, compilation, signature stability, hidden terminal support.
- **Cap:** 3 local E4B calls; 0 environment actions; $0; 10 minutes.
- **Stop:** After 3 calls or first transport/parser/model-drift error.
- **Invalid:** Scorer leakage, source/model drift, or incomplete responses.
- **Map:** Stable supported selection → planning gate; stable abstention → discriminator; wrong/unstable → stop; invalid → repair/new card.
- **Boundary:** This tests objective-relation acquisition, not route construction or live play.
