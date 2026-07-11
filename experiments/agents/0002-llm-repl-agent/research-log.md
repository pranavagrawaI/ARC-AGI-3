# Research log: 0002 generic LLM baseline

## 2026-07-11 — baseline contract

Question: how much structure can a capable local multimodal LLM invent during
interaction when given public observations, transition history, and a generic
Python workspace, but no imposed ontology or planner?

The policy contract is intentionally small. The process measurement contract is
large but post hoc: retain full model/tool/action evidence, then annotate when a
compact representation appeared, whether it remained stable without new
evidence, and where actions were wasted.

No hybrid mechanisms are included. Color permutation is the first contamination
probe. Spatial/layout permutation remains blocked until a behavior-preserving
public action transformation is specified and tested.
