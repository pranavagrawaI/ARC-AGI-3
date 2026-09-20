# Offline memory-path audit — Duck alone (periodic-v1 control arm, 20 episodes, 858 actor attempts)

Question: where does previously authored knowledge stop reaching Duck? Method: replay the deployed
note store (`_extract_scientist_note` + the level/game-over clearing policy) over the authoritative
per-game analyzer transcripts, attempt by attempt, and compare with what each USER PROMPT carried.
`replay.py` reproduces the table; the counts below use the next prompt's own wording
("You have progressed to a new level!" / "The game is over.") as the clear signal.

## Findings
| link in the chain | result |
|---|---|
| note in host store but absent from the actor request | **0 cases.** 12/858 replay/prompt disagreements remain, all at attempt-retry boundaries within one step; none is a note present in state and missing from the prompt |
| destructive clears | 28 notes destroyed (15 at level transitions, 13 at same-level GAME_OVER). Recovery: Duck re-authored a note on the very next attempt in 18/28, within 2 attempts in 26/28; two stretches of 14 and 20 attempts without a note (su15_p2, lf52_p0) |
| labeled note authored elsewhere (reasoning, code) | not found: the recognized labels appear only in assistant content when they appear at all |
| **the actor never authors a usable note** | **10/20 episodes: ≤1 labeled note in the whole episode** (sb26 3/4 passes at 0/68–72 attempts; lf52 3/4; ls20 2/4; ft09 1/4; su15 1/4). The other 10 episodes author on nearly every attempt (E22, E26, E21 runs). Same game, same prompt, different pass → different mode; decided in the first attempts and then stable |

So the dominant mechanism is per-episode **mode lock at the first turns**, not clearing and not
transport. Clearing accounts for 28 losses, almost all repaired immediately; it is real but small.

## Does note availability predict outcome? (control arm)
| episodes | n | levels/episode | actions/episode |
|---|---|---|---|
| note-authoring (assistant-text rate ≥ 0.5) | 9 | 0.89 | 92 |
| tool-only (rate < 0.5) | 11 | 1.18 | 112 |

The one L3 (ft09_p1) came from a tool-only episode. n is tiny, but there is no sign that carrying a
structured note helps this model reach levels; the memory that matters may be the message history and
the sandbox `history`/`transitions`, not the labeled block.

## Implication for the critic diagnosis
The critic's "designated structured belief source" was empty in 57/85 firings because Duck itself
had authored nothing in those episodes — not because the adapter dropped it (the packet reads the
same store the prompt does). Any critic that needs Duck's beliefs is blind in half the episodes by
construction of the actor, and repairing that is a change to the actor's generation, not transport.

## What this selects
From the candidate table: "the actor never authors a usable note" → the smallest candidate is an
explicit note field in the existing tool-call interface (no extra call). But the outcome table says
the prior for that helping levels is weak; it should be run as its own two-arm screen against frozen
Duck with a written prediction, and the critic stays off. Not started.
