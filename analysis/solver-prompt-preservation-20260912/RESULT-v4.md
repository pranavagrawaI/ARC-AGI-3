# Result: solver-prompt-preservation-local-v4

Status: valid diagnosis.

Observation:
- The one targeted execution reached the equality assertion after the real path delivered ACTION1; pytest failed in 5.96s as predicted.
- Both sides contained three requests. The structural diff found exactly 12 differences.
- Every difference was the same: a recorded assistant tool-call message retained `"content": null`, while the HTTP JSON omitted that field. No text, role, tool call, tool result, ordering, or cache-control value differed.
- External model calls: 0. ARC environment calls: 0. Spend: $0.

Interpretation:
The provider-bound serializer excludes null-valued fields after NOOA's cache-control rendering. The correct local representation is therefore the prepared message list with null dictionary values removed recursively. Capturing earlier is semantically close but not byte-structurally faithful enough for the requested replay evidence.

Next action:
Normalize null fields from the frozen request evidence, freeze v11 again, and qualify exact HTTP equality under a new card.
