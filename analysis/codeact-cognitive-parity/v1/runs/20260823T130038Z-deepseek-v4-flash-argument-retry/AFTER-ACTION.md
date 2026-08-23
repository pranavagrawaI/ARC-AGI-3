# After action: malformed native return arguments aborted the screen

## Disposition

**Invalid as a corpus screen; no comparative claim.** The first native session terminated when `return_result` contained the single unterminated character `{`.

## Observed trace

- Three native calls cost 0.00413412 USD. Including prior attempts, observed cumulative spend is 0.040596615 USD.
- The controller had recovery for a typed answer that decoded successfully but failed answer validation. It did not cover failure to decode the outer `return_result` argument object.
- No CodeAct call, scored report, canonical mutation, or ARC action occurred.

## Causal diagnosis

Native evidence-query arguments and CodeAct arguments were now recoverable, but native return argument decoding still occurred outside its validation guard. The same class of provider syntax error therefore remained fatal on one path.

## Next discriminating move

Preserve the run. Move native outer return decoding inside the existing rejected-return boundary, test malformed JSON followed by a valid return, and authorize one new immutable retry after debiting these calls and cost.
