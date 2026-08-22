```markdown
confidence: low
model: transition(RIGHT) -> C4 x-=3, C5 stays; transition(DOWN) -> C4, C5 y+=3
dynamics: simulation_accuracy=0.00; matched=0/evaluated=16/observed=18 current-attempt transitions; current_level=L0 matched=0/evaluated=16/observed=18; archived_unchecked=0 transitions; coverage=n/a
outcome: ongoing | hypotheses still open: unknown | unknown
outcome_verified: no
plan: start=L0 turn 0; validated=no; length=none; assumptions=none
recommended_action: RIGHT
subgoal: none
note: render() still diverges from observations; need to refine the C5 shape mask and clearing logic to achieve simulation_accuracy > 0.
```