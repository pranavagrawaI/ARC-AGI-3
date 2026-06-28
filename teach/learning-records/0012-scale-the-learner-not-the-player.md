# Scale the learner, not the player: meta-learning is the real target

Session 5/6 (2026-06-22/23). Pranav's "build for 1 game, tune to 2, ... get all"
hope, and the long critique of it (his or sourced), converge on a target-level
decision. The critique is largely correct; recorded with my sharpenings.

**The critique's core (accepted):**
- The plan's weight rests on "tune." ARC-AGI-3 games are built mutually novel,
  Core-Knowledge-only, distribution-shifted per level — task-specific transfer is
  adversarially ~0 *by design*. So "1→2→…→all" is not a ramp; it's 135 climbs
  sharing a parking lot.
- Sequential fine-tuning ⇒ catastrophic forgetting (fit 2, lose 1).
- "Train on many until covered" smuggles pretraining/amortization back in, and
  forks: zero-shot on game 135 = the $700K nobody can do; else = 135 specialists
  on a shared backbone (useful, but not "general player").
- **The orthogonality:** the benchmark scores *learning-curve slope*, not
  asymptote. A model that masters every game after a million runs each scores ~0.
  "Eventually gets all games" and "does well on ARC-AGI-3" are nearly orthogonal.
- **The rescue:** the unit that overfits per-game is the *weights/policy*. The
  unit the benchmark *permits* to transfer is Core-Knowledge structure
  (objectness/geometry/agentness) + the exploration-and-hypothesis-testing
  *procedure*. So scale the **learner**, not the player. Target quantity:
  sample-efficiency at mastering game N+1 *falls* as games 1…N are seen =
  meta-learning = Chollet's skill-acquisition efficiency = what scores.
- **Honest ladder:** sequential tune ⟶ joint multi-task ⟶ meta-learn the learner.

**My deltas (where I sharpen or correct it):**
1. **It's *test-time* adaptation, not "first action."** The agent DOES interact
   with and learn the novel game at evaluation; it just can't have pretrained on
   *that* game, and pays per action. So the unit to generalize is an online
   adapter that climbs the competence curve in few test-time actions.
   Pretraining on 1…134 is allowed and good — iff what you extract is the
   adapter, not game-specific policy. This strengthens the meta-learning frame.
2. **The proposed experiment tests the wrong unit.** Warm-starting game-2 from
   game-1 *weights* measures weight-transfer — the very thing predicted to be
   flat. Keep it as the null baseline, but the real test is *procedure/prior*
   transfer: learner-with-priors vs same-learner-without, x-axis = number of
   prior games the *procedure* was tuned on, y-axis = test-time acquisition cost
   on a held-out game. Down-slope ⇒ hope is real and is the scored quantity;
   flat ⇒ we've quantified the benchmark's no-transfer claim (still a finding).
3. **"Joint > sequential" only on the forgetting axis.** For mutually-novel
   games, joint training exposes *task interference / capacity dilution*: a small
   model co-fitting 135 adversarial games tends to mediocrity-at-all or to
   carving disjoint subnets (= specialists-on-a-backbone again). Joint kills
   forgetting but not interference. The true escape is meta-learning the adapter,
   not co-fitting policies.
4. **Architectural consequence for US (resource-constrained):** amortize little,
   infer much. Keep learned weights small; put intelligence in an explicit,
   fast, *test-time* inference/search/abduction procedure over the
   object-centric relational world model we're building. ARC history rewards
   test-time search / program-induction over big amortized policies — and it fits
   our no-compute constraint. We do NOT meta-learn objectness via MAML on a GPU
   farm; we *hand-build* the Core-Knowledge structure (0008 already did) and tune
   the exploration/hypothesis procedure on a few games.

**The decision this forces (pending Pranav's explicit confirm):** set the
north-star target to **sample-efficiency / fast test-time adaptation** (the
leaderboard quantity), and demote "eventually beat all games" to its honest
meaning — *acquisition cost on game N+1 falls*. If he instead wants
"eventually-all after per-game tuning," that's a legitimate but different,
near-0-leaderboard goal; the two pull apart and he must pick.

**Curriculum impact:** near-term lessons unchanged (0009 dynamics induction,
0010 goal induction, 0011 planning — the single-game learner), but reframed as
building the *adapter*, with priors held fixed and only the procedure tuned. The
transfer experiment (delta #2) becomes a payoff lesson once two games' learners
exist. Baselines/floor: StochasticGoose + official random agent (see RESOURCES).
[[0011-the-curiosity-trap-and-agency-grounded-exploration]]
[[0010-mission-pivot-to-a-learning-model]]
