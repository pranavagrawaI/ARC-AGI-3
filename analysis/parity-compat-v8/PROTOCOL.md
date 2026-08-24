# Parity compatibility v8

## Intervention

V8 is the final discoverability treatment. `repr(evidence)` now enumerates only the capabilities granted by the execution manifest. With the current authority it displays only `query`, embedding that leaf's manifest-derived admitted invocation and live attempt/disclosure state. It cannot advertise workspace, write, terminal, future, or other ungranted powers.

The container delegates to the same query object used by `repr(evidence.query)`, `help(evidence.query)`, and the concrete signature. Live budgets therefore update after rejected attempts and successful disclosures without creating a second source of state. These values remain advisory projections; T1 exclusively validates, meters, normalizes, and discloses evidence.

## Frozen comparison and stopping rule

DeepSeek V7 inspected the opaque parent container, then leaf help, then successfully queried and retrieved L0:T3 LEFT, but had no call left to return. V8 changes only the parent representation. Gates, prompt contract, leaf representation/help/signature, evidence, semantics, authority, and three-call ceiling remain fixed. Scripted controls establish `repr(evidence) -> query -> return` and direct `query -> return`.

If DeepSeek receives the complete container representation and nevertheless chooses further `dir`, `help`, signature, or other introspection instead of querying, discoverability is falsified and no further interface accommodation is allowed. Gate 1 passage establishes protocol eligibility only; Gate 2 then tests representation.
