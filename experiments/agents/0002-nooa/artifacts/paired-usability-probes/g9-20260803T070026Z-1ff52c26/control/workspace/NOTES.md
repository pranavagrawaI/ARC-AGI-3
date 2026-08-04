Usability probe, not game evidence: before choosing the
first action, use Python to create G9_CAPABILITY_PROBE.json with a JSON object whose
first_step is the current step. Before choosing the second action, use Python to read
that same file, add second_step with the current step, and save it. Then choose game
actions normally. This requirement is identical for both arms and supplies no game
semantics.