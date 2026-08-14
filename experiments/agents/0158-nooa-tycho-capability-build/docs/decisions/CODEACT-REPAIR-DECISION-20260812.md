# CodeAct repair decision — 2026-08-12

Raise the per-artifact CodeAct error budget from one to four while retaining the 12-iteration ceiling.

The prior live run reached a nearly valid claim, but its 8-column bbox carried 7-character mask rows. Pydantic generated exact validation feedback; `max_retries=1` then terminated immediately, contradicting the intended in-artifact repair boundary. Four errors permit bounded schema/code correction without changing ARC action authority, provider routing, output limits, or the non-resumable run boundary.

The error budget counts CodeAct generation/execution/validation errors. It is separate from the five HTTP 429 transport retries per provider call. A repair remains inside the same artifact and before any action is issued.
