# 0158 Gemma LS20 grounding v1
Question: Can frozen 0158 get `google/gemma-4-31b-it` through fresh LS20 level 1?
Success: `levels_completed >= 1`; gates are not success.
Input: public observations/actions only; no LS20 roles, mechanics, coordinates, route, map, tracker, or semantic state.
Caps: 44 actions after reset; 200 generations including retries; 4096 output tokens/generation; 5,000,000 reported tokens; 90 active minutes excluding review.
Tokens: sum reported `input_tokens + output_tokens`; record reasoning/cache separately. Missing required usage invalidates and stops. Spend unknown; record it.
Retry: one byte-identical retry only before response output; it and all reported usage count.
Control: claim -> commitment -> one action -> verdict. First consequential failure permanently closes the scored run; review configures only a new run/card.
Stop at G2 transition eligibility for review; expose no new API.
Execution authority: NONE until a separate approval digest is supplied.
