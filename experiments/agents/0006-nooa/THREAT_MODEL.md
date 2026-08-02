# What the 0006-NOOA evidence chain proves, and against whom

Pinned 2026-08-02. Cited by `CONTRACT.md` and by review rubrics. A finding that
assumes a stronger attacker than the one declared here is out of scope by
construction, not by concession — see §4 before filing it.

A parallel document governs the capability-environment line at
`experiments/agents/0006-capability-environment/THREAT_MODEL.md`. The two lines
share no code; this file is authoritative for `0006-NOOA` and cites this line's
own evidence.

## 1. What is being verified

A run produces a two-action evidence chain:

1. the model sees an observation;
2. the model authors a competence commitment;
3. the host admits an action;
4. the environment produces a successor;
5. the model sees the result and acts again.

The offline verifier reconstructs that chain and rejects forged or inconsistent
records. What it must establish is that **model-owned state survived across both
decisions without host semantic invention** — the host preserves and checks the
model's material, and never creates, ranks, or revises a hypothesis.

## 2. The attacker

**Declared in scope.** An attacker, or an accident, that alters part of a
completed run directory without regenerating the rest consistently:

- records that disagree with their producer;
- projections asserting their own type or shape rather than deriving it from a
  producer contract;
- identifiers rebound in one place and not the others;
- source files reached through a path that is not the path recorded;
- audits whose top-level fields drift from the canonical request they claim;
- external runtime targets whose type, mode, path, or bytes differ from record;
- source-snapshot additions, including bytecode caches, after execution.

Each of these was a real finding, and each is fixed. This is the class the
verifier is built for, and it is the class that ordinary experimental error
actually produces.

**Declared out of scope.** An attacker with full write access to the completed
artifact who rewrites every record and recomputes every unkeyed hash
consistently.

## 3. Why out of scope is a result, not a concession

For any verifier `V` over a mutable artifact using unkeyed hashes, an attacker
with full write access and knowledge of `V` can compute an artifact `V` accepts.
`V` is computable; the attacker runs it.

Every additional internal cross-link enlarges the object the attacker must
regenerate. None introduces a value the attacker cannot compute. Resistance to
this attacker is unreachable from inside the artifact at any amount of effort,
and requires an authority the artifact editor does not hold: a key-backed
signature, a parent-held secret MAC, an externally stored signed digest, or an
append-only remote log.

None of those is warranted here. This is a prototype whose purpose is to find
out whether an agent given a competence protocol plays the game better, and it
is operated by one person. An attacker who can rewrite the artifact is that same
person, who would also hold any signing key — so the key buys assurance against
nobody. The adversary that actually threatens this work is a silent
inconsistency, not a forger.

## 4. The stopping rule

Adversarial review against an unbounded post-run editor has no fixed point.
Every round produces a finding, and every finding is real under an attacker
nobody bounded. That is a specification failure, not an implementation failure,
and no verifier change terminates it.

So: a proposed finding must name which capability in §2 the attacker used. A
finding that requires coordinated rewriting of the whole artifact is answered by
this document and closed. A finding that shows a §2 attacker slipping through is
a defect and gets fixed.

## 5. The anchors that do live outside the artifact

Three, already collected, none costing anything further. They are **asymmetric**:
inputs are anchored strongly, outputs only in existence and shape.

**The approval digest — inputs.** Execution is two-phase. `--prepare-approval-
request` writes a no-spend request binding run identity, transitive source
hashes, the frozen `0002-NOOA` revision and lockfiles, runtimes, the provider
request profile, and every budget into one digest. A human reads that file and
types the digest back as `--confirm-live-run`. **The digest travels through the
operator, outside the artifact**, and the accepted request is atomically renamed
`.consumed` before the run root is created, so the grant is single-use. This is
the strongest anchor the run has, and it covers what went in, not what came out.

**Provider-side records — outputs, partial.** Each completed generation is
recorded by its OpenRouter ID and re-fetchable independently. Across the five
live runs on record: 7 generations, `$0.00592847` total, the one-action success
at `20260801T195450Z` costing `$0.00175948` over 2 generations. A local editor
can rewrite the stored copies; they cannot rewrite OpenRouter's server-side
history. Re-fetching binds provider identity, `endpoint_id`, model permaslug,
`created_at`, latency, finish reason, cost, `upstream_id`, `session_id`, and
token counts.

It does **not** bind content: the generation endpoint returns metadata only, with
no request or response bodies. The one partial content binding is arithmetic —
token counts are a lossy function of prompt content, so a fabricated chain must
tokenize to exactly the counts OpenRouter recorded.

**Committed and pushed bundles.** Publishing a bundle's digest to a remote that
is not rewritten gives a third-party timestamp. A weak transparency log, and
genuinely external.

So the honest scope statement is: *run inputs are anchored by an operator-
transported digest; within-artifact consistency is proved by the verifier;
artifact existence, provenance, and approximate shape are anchored by provider
records; record contents are not externally attested. An audit cites all three.*

## 6. What would change this

If a result from this work were to be relied on by someone who did not run it,
under conditions where a party benefits from the outcome, the out-of-scope
attacker becomes real and §3's list of external authorities becomes the required
work. Nothing short of that motivates it. Revisit when a result leaves the
prototype, not before.
