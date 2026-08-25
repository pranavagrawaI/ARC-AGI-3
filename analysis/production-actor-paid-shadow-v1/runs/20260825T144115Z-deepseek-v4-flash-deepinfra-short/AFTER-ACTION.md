# Invalid pre-dispatch attempt

The short authorization stopped before any provider call. `TychoNooaAgent`
initialization raised `SandboxError` because `TYCHO_SANDBOX_RUNTIME` was not
bound or configured. The durable ledger contains `RunArmed`, `GateStarted`, and
`RunStopped`; it contains no `CallPlanned` or later call event. Actual provider
spend is therefore $0.

This attempt is infrastructure-invalid and carries no model-behavior result. It
motivated binding the host+bubblewrap sandbox treatment in the manifest and
preflight before issuing a fresh single-use authorization.
