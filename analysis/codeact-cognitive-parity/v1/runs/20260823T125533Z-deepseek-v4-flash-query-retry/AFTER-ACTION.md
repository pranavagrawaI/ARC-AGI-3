# After action: malformed CodeAct arguments aborted the screen

## Disposition

**Invalid as a corpus screen; no comparative claim.** The first packet's native arm exhausted its call budget, then CodeAct returned an unterminated JSON tool argument that escaped the controller.

## Observed trace

- Native exhaustion was preserved nonfatally and CodeAct was still attempted, confirming the prior packet-boundary repair.
- CodeAct eventually returned an `execute_python` tool call whose arguments were unterminated JSON. Tool selection validation passed, but argument decoding raised `JSONDecodeError` outside the existing malformed-turn recovery.
- Thirteen calls cost 0.0104286 USD. Including prior attempts, observed cumulative spend is 0.036462495 USD.
- No canonical mutation or ARC action occurred.

## Causal diagnosis

Malformed provider turns were recoverable only when the number or name of tool calls was invalid. Malformed JSON or argument schemas passed through a second unguarded decoder. Provider syntax quality again determined whether the corpus survived.

## Next discriminating move

Preserve this run. Treat tool argument decoding/schema errors as the same T1-authored rejected provider turn, with bounded corrective feedback and no execution. Test malformed JSON recovery, then authorize a new immutable retry after debiting the thirteen calls and observed cost.
