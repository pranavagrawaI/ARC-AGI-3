# 0000 Random Agent

## Question

What performance floor does uniform random interaction establish through the
public-interface harness?

## Intervention

At each step, sample uniformly from the action IDs supplied by the current
observation. The pseudorandom generator has an explicit seed. For a parameterized
action, sample coordinates uniformly within the rendered frame.

## Expected signal

The run should validate the complete harness-to-agent-to-environment loop and
produce a reconstructable Admissible Trace. On LS20, the expected competence
signal is zero completed levels; any completion would be recorded rather than
assumed impossible.

## Measurement

- terminal state and stopping reason;
- actions taken and counts by action ID;
- levels completed;
- full public observation/action trace.

## Result

Measured on LS20 with seed 0 and a 500-action ceiling:

- 135 actions before the environment reached `GAME_OVER`;
- 0 levels completed;
- action counts: A1=44, A2=31, A3=32, A4=28;
- stopping reason: environment terminal state, not the harness action ceiling;
- trace: `experiments/artifacts/0000-random-agent/ls20-seed-0.jsonl`.

The trace contains 137 records: one initial observation, 135 transitions, and one
end record. Its first observation exposes only the harness's five public fields;
no private game identifiers, state, or oracle labels were recorded.

## Conclusion

The common harness executes a complete seeded contestant run and preserves enough
public data to reconstruct it. Uniform random action selection establishes the
expected zero-level LS20 floor. One seed validates plumbing, not a performance
distribution; more seeds are unnecessary until another agent needs a statistical
baseline comparison.
