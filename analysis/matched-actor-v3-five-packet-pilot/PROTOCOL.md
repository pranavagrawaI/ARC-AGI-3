# Protocol

Matched Actor V3 reuses the five frozen V2 packets and the same repaired
checkpoints. The sole intervention is receipt congruence in the sterile
execution path: `record_count` replaces the false `records` attribute in the
receipt representation, and the representation advertises
`evidence_cache.load(handle)`.

Both arms retain the same evidence semantics, immutable-handle memory, pure
computation, answer schema, six-call ceiling, and lack of canonical or ARC
authority. The package remains disabled until separately authorized by hash.
