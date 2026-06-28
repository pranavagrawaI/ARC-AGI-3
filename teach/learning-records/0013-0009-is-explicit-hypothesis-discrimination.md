# Lesson 0009 is EXPLICIT hypothesis discrimination (fail-loud), not implicit latent disagreement

Session 6 (2026-06-23). Target A locked; #4 accepted as *budgeted abduction with
the world model as falsifier*; 4a strengthened to *priors-as-hypothesis-family-
with-knobs* (all in MISSION + LR 0012). Pranav handed me the one fork he wouldn't
decide: does 0009's test-time loop discriminate among an **explicit, enumerable
rule-hypothesis set**, or an **implicit** one (the world model's predictive
disagreement under perturbation)? Decision: **explicit, for 0009.** Reasons,
strongest first:

1. **The locked metric presupposes it.** "Hypotheses-killed-per-action" is only
   definable over an explicit set. Implicit latent-disagreement yields a scalar
   info-gain proxy but never a crisp "this rule died." The discipline Pranav just
   locked *requires* legible hypotheses to be measurable.
2. **Explicit fails LOUD; implicit fails SILENT — and silent late failure is the
   benchmark's signature trap.** When the explicit hypothesis set is *all
   refuted but the goal isn't solved*, that empty surviving-set is an unambiguous
   signal: the grammar is too small. The implicit version can't distinguish "my
   space is missing the mechanic" from "noise" — it fails by being murky. So
   explicit isn't just the safer first brick; its failure mode is *observable*,
   which is exactly what you want when the enemy is quiet generalization failure.
   This directly answers Pranav's "too small and never recovers" worry: it
   recovers loudly (detect grammar gap → extend grammar), it doesn't silently
   stall.
3. **Maximally fits "amortize little, infer much."** Going explicit changes *what
   the world model IS* early: an ensemble of cheap **symbolic** simulators (one
   per surviving hypothesis), with ~no learned weights. "Discriminator-first"
   then means **probe selection that maximises disagreement among surviving
   hypotheses** — not a predictive-loss CNN. The neural frame-predictor is
   deferred to its real early job (and the implicit signal's real job): a
   *proposer* of new primitives when the grammar must grow. Synthesis:
   **explicit falsifier always; implicit proposer later.** Implicit is where it
   ends up to be general (Pranav's read), reached by making the grammar
   *extensible*, not by surrendering legible falsification.

**Guardrail (or this becomes the LR-0009 cheat):** the hypothesis grammar is
GENERIC — trigger = contact with some object-class × effect = an operator
(+k mod n, set, toggle, block, …) on an avatar attribute (colour/shape/rotation/
position). We do NOT seed ls20's actual rules; the agent discovers which
(trigger, effect) instantiations hold by falsification from play. GIVEN = the
causal-interaction vocabulary (agentness/causality made productive); LEARNED =
the specific rules. Source stays unread.

**So 0009 concretely:** (a) a generic interaction-hypothesis grammar over the
objects 0008 already segments; (b) a surviving-hypothesis set, each a cheap
symbolic forward-simulator; (c) a probe selector that picks the next action to
kill the most hypotheses per action (info gain), anchored to agency-grounded
interactions (LR 0011) so noisy-TV effects earn nothing; (d) the empty-set
detector that flags a grammar gap. First brick of the adapter: built to refute
rules cheaply before it is ever asked to plan through them.
[[0012-scale-the-learner-not-the-player]] [[0011-the-curiosity-trap-and-agency-grounded-exploration]] [[0009-the-real-cheat-is-hardcoded-rules]]
