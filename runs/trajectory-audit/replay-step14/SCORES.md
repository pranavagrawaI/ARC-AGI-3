# Scores — replay of ft09_p0 step 14 (read against RUBRIC.md; quotes from trials/*/transcript.txt)

C1 = BR centre derived from code output this turn. C2 = mapping checked on all visible references
(TL, TR, BL) before rejecting/applying. Decode = the stated BR target. Precommitted set = pairs
0,1,2,4,5,6 (pair 3 was a 429 on both arms with no model output; re-run as pair 6). The smoke pair
(`trials-smoke/00-*`, identical conditions, run before the set) is reported separately.

| trial | C1 | C2 | decode stated | acted on | sim level | s |
|---|---|---|---|---|---|---|
| 00-control | ✗ "The center was W g g / W r W / g g W" (recalled) | ✗ | none (XOR) | XOR | no | 49 |
| 00-treatment | ✗ no code before acting | ✗ | none (XOR) | XOR | no | 28 |
| 01-control | ✗ "all 4 center patterns appear identical" (recalled) | ✗ | none (all-red) | all-red | no | 44 |
| 01-treatment | ✗ no code before acting; "TL center: W g / g w" (recalled) | ✗ | none (TL⊕BL) | TL⊕BL | no | 125 |
| 02-control | ✗ recalled centres | ✗ | none | click centre | no | 75 |
| 02-treatment | ✗ "all centers are the same!" (recalled) | ✗ | none (AND) | AND | no | 100 |
| **04-control** | ✓ printed TL/TR/BL/BR by code (segmentation, then ascii rows 44–49) | ✓ "TL … ✓✓✓ YES! Center encodes outer"; "BL … ✓ YES!"; "BR … ✗ … encodes the TARGET" | **R b b / R ? R / R b b** | 7 toggles | **completed** | 808 |
| 04-treatment | ✗ recalled; "BR center: W g g / W R W / g g W" | ✗ | none (all-red) | all-red | no | 45 |
| 05-control | ✗ recalled "W g / g g" | ✗ | none (XOR) | XOR | no | 53 |
| 05-treatment | ✗ no code before acting | ✗ | none (XOR) | XOR | no | 101 |
| 06-control | ✗ "All 4 centers identical" (recalled) | ✗ | none (XOR) | XOR | no | 41 |
| 06-treatment | ✓ printed four centres by code (BR rows correct: `WWgggg…WWgggg`) | ✗ chose "W=blue, g=red" with no reference check | b R R / b ? b / b R R (inverted) | 2 toggles | no | 155 |

**Precommitted set:** control C1 1/6, C2 1/6, both 1/6, correct decode 1/6, level 1/6.
Treatment C1 1/6, C2 0/6, both 0/6, correct decode 0/6, level 0/6.

Smoke pair (extra, same conditions): control C1 ✓ (printed all four centres), C2 ✓ (checked TL first, decoded
"BR center is W g g / W R W / W g g → target R b b / R ? R / R b b"), decode correct, but acted on a *recalled*
current outer pattern (bottom row "R R b", actual "R R R") and missed one toggle → not completed. Treatment C1 ✓,
C2 ✗ (tried "W→blue, g→blue, R→red"), wrong decode.

Including smoke: control both-criteria 2/7, treatment 0/7. In 5 of 7 treatment trials the model executed
no code at all before acting — the added line was not acted on; in the other 2 it printed and skipped validation.
