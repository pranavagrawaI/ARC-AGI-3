# 0006-NOOA deletion log

The verifier is the contract-test suite plus the offline frozen-worker crossing.
These are intentional temporary deletions made on 2026-08-01 and immediately
restored after the stated failure. They are not hypothetical arguments for
keeping code.

| Pass | Temporary deletion | Verifier failure | Minimal restoration |
|---|---|---|---|
| context selection | Removed `requested_evidence` from the fixed context packet | `test_model_claim_becomes_a_commitment_then_a_public_mismatch` raised `KeyError: 'requested_evidence'` | Restore the selected-evidence field only; no retrieval/ranking added |
| reconciliation gate | Bypassed `_validate_reconciliation` in action admission | The mismatch test accepted a second action without a `ContractError` | Restore the explicit all-mismatch coverage check only |
| frozen action binding | Allowed JSON in `reason` to override `action` or `data` | The final-action boundary test accepted an overriding action | Restore the two-key rejection only |

An earlier pre-admission simplification was retained: cell-coordinate assertions
were deleted because the actual public observation is multi-layer and the
verified treatment needs only shape-neutral changed-cell, state, and
level-count predicates. A cell ontology has not earned re-admission.
