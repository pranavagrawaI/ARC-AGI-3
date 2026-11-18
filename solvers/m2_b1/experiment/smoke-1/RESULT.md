# m2-b1-smoke-1 result (2026-10-10)

Invalidity: none. COMPLETE on an RTX PRO 6000; "M2 B1 installed" at t=6 s; 3 games; about 0.44 GPU-h charged (budget.json). Raw output: analysis.json (analyze_b1.py smoke).

| game | levels | requests | pinned requests | trims | trims with pin | fallbacks | max pin tokens |
|---|---|---|---|---|---|---|---|
| ft09 | 4 | 93 | 70 | 4 | 3 | 0 | 12,073 |
| lf52 | 2 | 145 | 127 | 6 | 5 | 0 | 12,208 |
| s5i5 | 2 | 74 | 59 | 2 | 2 | 0 | 10,877 |

All card predictions held. The pin fired in 3 of 3 games with 0 fallbacks; python_timeout_s was 60 and static rules were on; there were no "No user query" errors.
The exception types are the same family as the m2_evidence smoke (ValueError 8, TypeError 5, NameError 5, StaleState 5, KeyError 3). Levels were 2/2/4, identical to the m2_evidence smoke (n = 1, smaller window; not a comparison).
"Produced no tool call": 2 genuine events, both in s5i5 (the third match is the model quoting the message), against 1 in the m2_evidence smoke. Not clearly above, so static rules stay on for the pair.
The format rule also remains in the system prompt's "Tool session rules".

Defect found while reading the traces: the system prompt's tool-session rules still said "Each `python` tool call has a hard time limit of 30 seconds" while the enforced limit was 60.
Fix (after smoke, before pair freeze): `_effective_tool_timeout()` sets both the limit and the stated value. The gate checks "limit of 60 seconds" is in the system prompt and "30 seconds" is not. Re-qualified 10/10. This prompt-text change was not GPU-smoked.

Per card: all predictions hold → launch pair-1, with the post-smoke fix recorded in pair-1's card.
