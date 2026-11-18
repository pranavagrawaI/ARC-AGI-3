# Action-tempo comparison v4: invalid provider-timeout prefix

V4 began baseline-first. The baseline executed two actions in 184.59 seconds after
14 successful Actor calls, then an OpenRouter read timed out with a request reservation
still unresolved. The frozen card therefore invalidated the run and prohibited starting
the critic arm. No retry occurred under this card.

The first action occurred after six Actor responses, so the persistent instruction to
act by response two already failed its prompt-uptake criterion. This prefix is diagnostic
only. It cannot rank the tempo critic because that arm is absent and the accounting did
not settle.

The unchanged candidate was subsequently tested under the new critic-first V5 card.
Raw requests, responses, action grids, the transport failure, snapshots, frozen hashes,
and partial RESULTS.json remain in this directory.

