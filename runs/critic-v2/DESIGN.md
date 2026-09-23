# Critic v2: the questioner (design note, 2026-09-23)

## Why v1 lost (evidence, not opinion)
v1 "question mode" (`inference/critic/question.py`) said "you never solve; you ask", but what
reached the actor (`report.py:render_report_block`) was, in order:
1. `Summary: assumption you may be stuck on: <X>` -- a diagnosis asserted as fact,
2. `Recommended actions: [<probe>]` -- an action to take,
3. `Open questions: <question>`.
The `probe:` field turned the question into advice. Step 3 ft09: the same "large charcoal
square" click was recommended up to 6x in 15 actions and the actor took it 7/14; Duck alone
reached L3 there (14-11 loss). The "do not repeat" guard showed only the last 4 questions,
enforced by the model, with no notion of whether a question had been answered.

## User's principle (2026-09-23)
The critic does not know better than the actor, so it must never advise. It asks the question
that breaks the actor's current worldview -- a zen master's question: why do it that way? could
it be this instead? why was this not considered? is that really the most beneficial way? It
never repeats an answered question.

## Design consequences
- Output = one question. No probe, no recommended action, no asserted diagnosis. The host
  rejects output containing an action token and renders only the question.
- Aim at the actor's reasoning, not the board. The critic needs the actor's recent stated plan
  and reasons (thinking/plan text), which v1 never saw (it saw beliefs + 700 chars of code).
- Prefer questions that expose a premise ("why X?", "what would you expect if X were false?")
  over questions that plant a hypothesis ("what if Y?"). Planting is advice in question form and
  is what repeated on ft09; allow it, but never the same planted hypothesis twice.
- Question ledger per game: each question, the actor's next response, and whether the actor
  engaged (answered / tested / dismissed). Answered questions are never re-asked; the ledger is
  shown in full, and near-duplicates are filtered host-side, not trusted to the model.
- Cadence: the user's manual critic fired every turn. At 3.8's ~192 s/turn, a question must be
  short; cadence is an experimental variable (every turn vs on stall), priced in lost turns.
- Grading offline first: replay recorded 3.8 stall points; a question passes if it targets the
  premise that the eventual level-unlocking turn overturned. The user's human-critic
  transcripts are the reference for what a good question looks like.
