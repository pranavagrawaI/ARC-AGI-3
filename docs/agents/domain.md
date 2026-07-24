# Domain Docs

How the engineering skills should consume this repository's domain documentation when exploring the codebase.

## Before exploring, read these

- `CONTEXT.md` at the repository root.
- Relevant decisions under `docs/adr/`.

If either is absent, proceed silently. The `/domain-modeling` skill creates domain artifacts lazily when terms or decisions actually resolve.

## File structure

This is a single-context repository:

```text
/
├── CONTEXT.md
├── docs/adr/
└── experiments/
```

## Use the glossary's vocabulary

When output names a domain concept—in an issue title, proposal, hypothesis, or test—use the term defined in `CONTEXT.md`. Do not drift to synonyms the glossary explicitly avoids.

If a needed concept is absent, reconsider whether the new language is necessary; if it represents a real distinction, resolve it through `/domain-modeling`.

## Flag ADR conflicts

If proposed work contradicts an existing ADR, surface the conflict rather than silently overriding it.
