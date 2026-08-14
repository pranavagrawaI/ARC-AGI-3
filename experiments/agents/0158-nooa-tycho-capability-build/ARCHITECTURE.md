# Architecture

```text
public ARC observation
        |
protected Python 3.14 host
        | content-addressed serialized evidence
fresh Python 3.12 NOOA process + fresh CodeAct sandbox
        | GroundingClaim draft (no action)
host schema/evidence validation + host claim_id
        |
fresh Python 3.12 NOOA process + fresh CodeAct sandbox
        | ProbeCommitment (one action, anchored effect, alternative-action effects)
host legality and temporal binding
        | exactly one action authorization
successor observation
        | exact non-oracular effect verification
observed effect + reciprocal prospective contrast
        | action specificity supported / contradicted / inconclusive
G0/G1/G2 or review stop
```

Tycho-derived evidence conventions keep ordinary decision frames and transitions separate from terminal, death, and animation events. Attempts and levels cannot be stitched into ordinary transitions. The model receives one coherent immutable snapshot through `inspect_evidence()`, including the current frame, matched current transition/probe, visible history, accepted claim, and legal actions.

Lower-capability sandboxes do not mount advanced implementations. Forbidden capabilities are absent from the runtime source and exercised namespace, rather than hidden through prompting.
