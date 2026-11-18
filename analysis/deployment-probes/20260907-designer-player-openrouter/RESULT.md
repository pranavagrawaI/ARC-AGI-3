# Invalid first contact; no solver judgment

Both arms received HTTP 404 on their first upstream attempt; zero actions and no
successful responses. Error body was not retained by v1. Do not infer the exact
404 cause from the status alone.

Independent request inspection found the two rendered prompts identical: the
shared transport allowlist dropped ARC_SOLVER_ARM, so both workers took the loop
default. This invalidates the intended contrast independently of the HTTP fault.
Local method-call tests did not catch this transport omission.

Endpoint metadata also advertises no parallel_tool_calls support among available
providers, while the relay required every parameter. That is a concrete routing
incompatibility to remove in a new card, not a demonstrated explanation without
the missing error body.

Reserved spend: $0.07906388 (conservative reservation, not actual billing).
Original evidence and runner.py retained. No retry occurred within either arm.
Next: regression-check actual worker arm; assert policy in the outgoing body;
pin one supported FP8 provider, remove only the unsupported optional flag, retain
error response bodies, and run v2 within the original cumulative $2 cap.
