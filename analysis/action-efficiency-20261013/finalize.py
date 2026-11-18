"""Assemble the requested reviewable findings and machine-readable result from verified receipts."""
import collections
import json
import re
import statistics
from pathlib import Path
from analyze import HERE, ROOT, RUNS, sha, parse


def auxiliary():
    out={}
    for name,(rel,role) in RUNS.items():
        directory=ROOT/rel; count=0; suffix=[]
        for path in (directory/'transcripts').glob('*.txt'):
            for m in re.finditer(r'\[TOOL RESULT: python\]\n(.*?)(?=\n\[[A-Z]|\Z)',path.read_text(),re.S):
                count += 'StaleStateActionError' in m.group(1)
        for game in json.loads((directory/'benchmark.json').read_text())['game_runs']:
            hist,apl=map(parse,(game['history'],game['actions_per_level']));k=0
            for n in apl[:game['levels_completed']]:
                last=max([i for i,h in enumerate(hist[k:k+n]) if h['generated_tokens']>0],default=0)
                suffix.append(n-last);k+=n
        out[name]=dict(stale_error_tool_result_blocks=count,
                       stale_scope='all reached levels, text-result blocks; unexecuted actions excluded from action score',
                       solved_levels_with_generation_free_final_suffix_2_or_more=sum(n>=2 for n in suffix),
                       median_generation_free_final_suffix_actions=statistics.median(suffix),
                       generation_free_suffix_definition='From last positive-generated-tokens action to completion inclusive; no claim about rule knowledge')
    (HERE/'auxiliary-observations.json').write_text(json.dumps(out,indent=2)+'\n')
    return out


def main():
    a=json.loads((HERE/'accounting.json').read_text())
    chain=json.loads((HERE/'chain-summary.json').read_text())
    v=json.loads((HERE/'verification.json').read_text())
    aux=auxiliary()
    plans=json.loads((HERE/'plan-audit.json').read_text())
    rows=list(map(json.loads,(HERE/'levels.jsonl').open()))
    game_rows=list(map(json.loads,(HERE/'games.jsonl').open()))
    phase=collections.defaultdict(collections.Counter)
    for r in map(json.loads,(HERE/'phase-accounting.jsonl').open()):
        phase[r['run']].update(r['disjoint_observation_partition'])
    failures=[]
    for r in map(json.loads,(HERE/'deletion-checks.jsonl').open()):
        for mode,t in r['trials'].items():
            if t['removed_candidates'] and not t['solves']:
                failures.append(dict(run=r['run'],game_id=r['game_id'],level=r['level'],mode=mode,
                                     removed_candidates=t['removed_candidates'],trial=t))
    original=json.loads((ROOT/RUNS['control'][0]/'benchmark.json').read_text())
    bases=[b for r in original['game_runs'] for b in r['base_actions_per_level']]
    mechanisms=[
        dict(waste='before correct rules/full plan',mechanism='Execute discriminating probes selected by the existing simulator; retain its tested transitions and counterexamples in Python data.',
             extra_model_tokens='None for bookkeeping/execution; changing which probes are informative remains model-dependent.',
             evidence='KA59, SB26, CD82 explicit full-plan witnesses arrive late; no proof that a fixed generic probe policy can halve induction actions.'),
        dict(waste='deaths/RESET episode replay',mechanism='Validate the model-provided path in its own simulator, including timer, hazards and reset semantics; replan in Python when observed state disagrees.',
             extra_model_tokens='None if simulator/rules are already correct and retained.',
             evidence='Hindsight reset-prefix deletion saves 94-625 solved-level actions; source-free live correctness is unproven, and TN36 reset-prefix deletions fail.'),
        dict(waste='post-plan execution',mechanism='Compile and execute the model-generated plan with early terminal stop; shorten cycles using already-validated inverse transitions.',
             extra_model_tokens='None for path compilation/cycle simplification.',
             evidence='TR87 r2 L4: 18 to 9 actions via DOWN inverses; model already executes its own planners and batched routes in other audited cases.'),
        dict(waste='visual no-op/repeat/stale',mechanism='Cache tested full-state/action transitions with animation and phase evidence; use reversible advisory dedupe and a model override. Refresh Python state references after action calls.',
             extra_model_tokens='None for cache lookup/reference refresh; refusal/recovery can add tokens and must be measured.',
             evidence='Repeated visual no-op deletion earns 0-0.049701 mean points. LF52 is a replay counterexample to hard suppression based on cropped pixels. Existing stale/batch guards already refuse actions.')
    ]
    maxrel=max(r['max_direct_relative_gain_percent'] for r in a['runs'])
    result=dict(schema_version=1,date='2026-10-13',actor='codex-action-efficiency',
        question='Is action efficiency on levels already solved worth >=20% hidden score?',
        answer='NO as a supported investment claim; hidden-specific headroom is not identified.',
        public_verdict='All-action perfect-efficiency ceiling is +0.7956 to +6.5982 percent at fixed completions; post-insight optimization is a subset.',
        hidden=dict(measured_gain=None,rigorous_hidden_specific_upper_bound=None,
                    explanation='Hidden per-level baselines/actions/completions are unavailable. Public bounds do not establish a universal hidden bound.',
                    conditional_transfer_max_relative_gain_percent=maxrel,
                    conditional_hidden_score_at_27_baseline=27*(1+maxrel/100),
                    relative_gain_required_percent=20,required_gain_at_27_baseline_points=5.4,
                    minimum_hidden_to_largest_public_relative_headroom_ratio=20/maxrel,
                    additional_completions_or_time_savings='not bounded by this fixed-completions analysis'),
        scoring=dict(level='min(115, 100*(baseline_actions/actions)^2)',
                     game='min(sum((L+1)*level_score)/sum(1..N),100*sum(1..K)/sum(1..N))',
                     event_score_semantics='levels_completed, not efficiency score',
                     baseline_source='version-exact environment metadata.json, reconciled to every benchmark',
                     public_unique_level_baseline_median=statistics.median(bases),public_unique_levels=len(bases)),
        runs=a['runs'],whole_game_hindsight_deletion_scenarios=chain,
        waste_phase_disjoint_partition={k:dict(val) for k,val in phase.items()},
        plan_audit=plans,mechanisms=mechanisms,
        coverage=dict(games=v['games_replayed'],solved_level_observations=len(rows),solved_actions=sum(r['actions'] for r in rows),
                      frames_verified=v['frames_compared'],independent_scorer_game_scenarios=v['scorer_game_scenarios'],
                      globally_certified_shortest_level_observations=sum(r['global_oracle_actions'] is not None for r in rows),
                      manually_anchored_full_plan_boundaries=len(plans),insight_unknown_solved_levels=len(rows)-len(plans),
                      deletion_failure_trials=len(failures)),
        deletion_failures=failures,auxiliary_observations=aux,
        unknowns=['Hidden action-efficiency distribution and hidden completed-level ceiling.',
                  'Earliest correct-rule insight for all levels; five explicit full-plan witnesses were audited, not a blind causal classification.',
                  'Exact optimal paths outside TU93 L0-L2; prior LP85/VC33 graphs are conditional on frame keys/click candidates.',
                  'Counterfactual information lost by removing exploration, and tokens/time recovered or additional completions.',
                  'Production gain of the mechanisms; local hindsight replay supplies rules/routes that may not yet be known online.'],
        authority=dict(spend_usd=0,model_calls=0,training=False,gpu_launches=False,competition_submissions=False,
                       authored_artifacts_directory='analysis/action-efficiency-20261013',
                       prescribed_board_post=True,
                       setup_exceptions='Initial board inventory used python3 without -B. First installed scorer import created a temporary matplotlib cache under /tmp; later Python commands use -B and plotting imports use .runtime-cache within the work directory.'),
        artifacts=dict(levels='levels.jsonl',games='games.jsonl',actions='actions.jsonl',phase='phase-accounting.jsonl',
                       replay='replay-checks.jsonl',deletions='deletion-checks.jsonl',composed_routes='chain-checks.jsonl',
                       exact_bfs='exact-bfs.json',manual_plans='plan-audit.json',validation='verification.json',
                       inputs='input-manifest.json',sources='source-manifest.json',
                       superseded_deletion_evidence='evidence-pre-death-reset-fix/'))
    if (HERE/'board-receipt.json').exists():
        result['board_post']=json.loads((HERE/'board-receipt.json').read_text())
    (HERE/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    table=[];waste=[];bounds=[];deletion=[]
    for r,c in zip(a['runs'],chain):
        m=r['means'];obs=r['waste_observations'];name=r['run']
        table.append(f"| {name} | {r['solved_levels']} | {r['levels_capped_at_115']} | {r['games_at_completion_ceiling']}/25 | {m['actual_score']:.6f} | {m['fixed_completions_ceiling']:.6f} | +{m['perfect_efficiency_gain']:.6f} | +{r['max_direct_relative_gain_percent']:.3f}% |")
        bounds.append(f"| {name} | {m['baseline_efficiency_score']:.6f} | {m['known_plan_at_entry_optimal_execution_score_upper']:.6f} | {m['exploration_halved_most_generous_score_upper']:.6f} | {m['global_certified_oracles_only_score']:.6f} |")
        waste.append(f"| {name} | {r['solved_actions']} | {obs['fatal']} / {obs['reset']} | {r['reset_episode_prefix_actions']} | {obs['interior_unchanged']} / {obs['full_frame_unchanged']} | {obs['repeated_observed_noop']} | {aux[name]['stale_error_tool_result_blocks']} |")
        x=c['modes']
        deletion.append(f"| {name} | {x['remove_reset_prefix']['removed_actions']} / +{x['remove_reset_prefix']['direct_mean_gain']:.6f} | {x['remove_interior_unchanged']['removed_actions']} / +{x['remove_interior_unchanged']['direct_mean_gain']:.6f} | {x['remove_repeated_noops']['removed_actions']} / +{x['remove_repeated_noops']['direct_mean_gain']:.6f} |")
    ptable=[]
    for p in plans:
        ptable.append(f"| {p['run']} {p['game']} L{p['level']+1} | {p['pre_full_plan_actions']} | {p['original_execution_actions']} | {p['verified_execution_actions']} | {p['verified_post_plan_saving']} | +{p['direct_public_mean_gain_this_case']:.6f} |")
    losses=sorted((g for g in game_rows if g['run']=='control' and g['perfect_efficiency_gain']>1e-9),key=lambda g:-g['perfect_efficiency_gain'])
    lossline=', '.join(f"{g['game_id'][:4]} {g['perfect_efficiency_gain']:.6f} game points" for g in losses)
    content=f'''# Action efficiency on already-solved levels — 2026-10-13

**Decision: no, (b) is not supported as a >=20% hidden-score lever.** Even perfect action efficiency on every already-solved public level can increase mean score by only **0.377048–3.019823 points (0.796–6.598%)** across the five logged runs. The original incumbent control is **52.481598 → at most 53.215584 (+1.399%)**. Optimizing only execution after insight has no larger ceiling. This is a public, fixed-completions bound; hidden-specific headroom remains unknown. Evidence: [accounting.json](accounting.json), [games.jsonl](games.jsonl), [RESULT.json](RESULT.json).

## What the scorer actually pays

For completed level index L (zero-based), `q_L = min(115, 100*(b_L/a_L)^2)`. Uncompleted levels score zero. A game with N levels and K completed levels scores

`min(sum((L+1)*q_L)/sum(1..N), 100*sum(1..K)/sum(1..N))`.

The second term caps credit by completed depth. The brief omits the factor 100 in the level formula and the second game cap. A 1.3x action cut multiplies an uncapped level score by 1.69, but it need not increase the game score at all. Baselines come directly from each exact game/version's `metadata.json`, and match every logged `base_actions_per_level`; the median across the {len(bases)} distinct public levels is {statistics.median(bases):g}. Event `score` is a completed-level count, not action-efficiency credit. RESET and fatal actions are retained in action counts; the initial seeded RESET is not in history. Evidence: [levels.jsonl](levels.jsonl), [source-manifest.json](source-manifest.json), [verification.json](verification.json); primary scorer `/home/pranav/projects/arc-agi-3-reference/dfranzen-m2/tufa-arc-agi-framework/src/taaf/game.py:381-413` and installed `arc_agi/scorecard.py:150-209`.

## Whole public-25 score ceiling

| Run | Solved levels | Level score already 115 | Games at depth cap | Actual mean | Fixed-depth ceiling | Gain, points | Gain, relative |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

Evidence: [accounting.json](accounting.json), [games.jsonl](games.jsonl). Primary old controls have different configured horizons (7,254 and 6,000 seconds); later c14 is a serving reference, and b1-c14 changes prompt/history too. These are descriptive per-run ceilings, not a paired intervention estimate. E1 Swift uses a different model and is excluded. Exact input paths and SHA-256 hashes: [input-manifest.json](input-manifest.json). Original-control residual credit is confined to four games: {lossline}; divide their sum by 25 to obtain the mean ceiling gain. In particular, raw level inefficiency in VC33 contributes **zero** original-control game headroom because other level bonuses already reach its depth cap.

## Baseline, optimal execution, and halved exploration

| Run | All solved levels at human baseline | Post-insight optimum: generous upper bound | Exploration halved: generous upper bound | Only globally certified TU93 routes substituted |
|---|---:|---:|---:|---:|
{chr(10).join(bounds)}

Evidence: [games.jsonl](games.jsonl), [accounting.json](accounting.json). Human-baseline efficiency sets every completed level to 100, so it equals the depth ceiling. For post-insight execution, assume knowledge is available at entry and an unrealistically short one-action finish on every solved level: that still cannot exceed the displayed depth ceiling. The true knowledge boundary and global optimal route are usually unknown, so this is an upper bound, not a measured counterfactual total. For exploration E <= A, halving E removes at most `floor(A/2)` integer actions. Replacing every solved level's A by `ceil(A/2)` (including necessary execution actions) supplies an intentionally overgenerous exploration-halving bound; realistic E gives smaller gains. These bounds hold completions fixed and do not assign extra completions to time saved.

All {len(rows)} solved (run, level) observations have actual action counts, exact human baselines, level scores, baseline-efficiency scores, and known/unknown oracle fields in [levels.jsonl](levels.jsonl). An independent BFS using exact timer, all sprite properties, internal phase and direction queues certifies **TU93 L1/L2/L3 shortest distances 18/10/19**, checking all four allowed actions. The same semantic roots and routes verify across all five runs: **15 globally certified level observations**. RESET cannot improve a shortest route in this deterministic game because it returns to the same start at positive cost. Evidence: [exact_bfs.py](exact_bfs.py), [exact-bfs.json](exact-bfs.json), [verification.json](verification.json). Original TU93 actions are 18/19/24: its L1 path is already shortest, L2 has 9 excess actions, and L3 has 5. These first-three route substitutions add **zero** original-control game credit.

The archived BFS also marks LP85 L1 distance 5 and VC33 L1 distance 3 as exhausted. We retain their conditional distances and scores, but do not call them globally certified: its keys merge states by final rendered frame, and mouse candidates cover components/logged clicks rather than all screen coordinates. SC25 L1 is exhausted with no goal despite logged solves, which further prevents treating the exhaustion flag alone as proof of a global oracle. Evidence: `analysis/oracle-progress-20261009/oracle.py`, graph paths in [levels.jsonl](levels.jsonl), and [oracle-traces.jsonl](oracle-traces.jsonl). Optimal actions outside the three independently certified TU93 levels are unknown.

## Waste accounting and the knowledge boundary

| Run | Solved actions | Deaths / RESETs | Prefix through last RESET | Interior / full-frame unchanged | Repeated visual no-op | Stale-error result blocks |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(waste)}

Evidence: [actions.jsonl](actions.jsonl), [levels.jsonl](levels.jsonl), [auxiliary-observations.json](auxiliary-observations.json). Prefix counts cover entire attempts discarded by later RESETs; they may have taught essential rules and are not all safely avoidable. No-op/repeat keys crop one boundary cell on each edge; that is an observational convention, not complete engine-state identity. These columns overlap and must not be added. Stale blocks cover all reached levels and count text-result blocks reporting `StaleStateActionError`, not uniquely identified refused calls or scored actions. A refused action creates no action receipt and adds no action-efficiency penalty, though recovery can cost tokens/time.

[phase-accounting.jsonl](phase-accounting.jsonl) provides a disjoint partition for every solved level: fatal/RESET actions first, other interior-unchanged actions second, remaining moves before/after a manually audited full-plan witness where available, otherwise phase unknown. **Only five full-plan boundaries were manually validated; earliest correct-rule insight and the exploration-versus-execution split for the other 520 solved observations are unknown.** A correct-looking thought does not establish a correct transition model. Even in audited cases, the prefix is before the explicit full-plan witness, not proof that every prefix action was wasted exploration. Global excess over shortest distance is known only on the certified TU93 levels.

| Explicit full-plan witness | Actions before witness | Actual finish | Verified finish | Execution saving | Public mean gain |
|---|---:|---:|---:|---:|---:|
{chr(10).join(ptable)}

Evidence: [plan-audit.json](plan-audit.json), original transcript paths and exact action/analysis-step anchors therein. VC33's final click aligns its bar with the gap; KA59 corrects its goal to both rings occupied; SB26 corrects its recursive reading/permutation before six swaps and SPACE; CD82 corrects the SW stencil from strict to inclusive before its own backward painting search succeeds. These are concrete cases of late goal/rule/model correction followed by existing planner execution. The 7-, 13- and 22-action finishes were replayed, but their global optimality is not certified; zero shortening found is not proof of zero waste.

**A concrete positive execution example:** TR87 repeat L5 has 82 actions before the corrected tile-versus-glyph plan, then 18 to finish. Replace five UPs with two DOWNs, four UPs with three DOWNs, and six UPs with one DOWN; preserve cursor movements. The 9-action replacement completes from the exact logged checkpoint. It cuts that finish by 50%, but the whole level only from 100 to 91 actions and earns **+0.071177 mean public points**. This is one verified compression, not a globally optimal suffix certificate. Source: game `tr87.py:974-1005` (bidirectional cycle); routes, replay receipt, and score calculation: [plan-audit.json](plan-audit.json).

Existing execution already spends no new model generation between many finishing moves: original control **82/107** solved levels have a final suffix of at least two such actions (median suffix **6 actions**). Evidence: [auxiliary-observations.json](auxiliary-observations.json); each suffix starts at the last action charged positive generated tokens. This measures batching/execution receipts, not insight, and means generic “execute the model's own path” largely duplicates current behavior.

## Replay-verified deletion scenarios

These hindsight changes shorten successful recorded routes, preserve failed candidates, and then replay the composed routes from each game's initial state. Original routes are retained wherever a deletion trial fails. They are achievable route counterfactuals under already-known logs, not evidence that the online model can avoid the probes while learning the same rules. The three scenarios are separate and must not be summed.

| Run | Remove discarded RESET prefixes: actions / mean gain | Remove visual no-change candidates: actions / mean gain | Remove repeated visual no-op candidates: actions / mean gain |
|---|---:|---:|---:|
{chr(10).join(deletion)}

Evidence: [deletion-checks.jsonl](deletion-checks.jsonl), [chain-checks.jsonl](chain-checks.jsonl), [chain-summary.json](chain-summary.json). All retained composed scenarios preserve the recorded completed levels. Savings can exceed the initial candidate count when the shortened sequence completes earlier and its remaining actions are omitted.

There are **{len(failures)} rejected deletion trials**. Counterexamples include LF52 repeat L2: removing 10 visually unchanged actions, or just two repeated visual no-op actions, prevents completion; and TN36 c14 L3: removing its 62-action pre-RESET prefix prevents the remaining nine actions from completing. Exact failures remain in [deletion-checks.jsonl](deletion-checks.jsonl) and [RESULT.json](RESULT.json). They invalidate blanket visual dedupe and the assumption that every RESET reproduces the original entry state. The exact underlying hidden-state cause was not established for every failure.

The first deletion checker incorrectly stopped at any death, including intentional death followed by recorded RESET. It was fixed to allow that continuation, all checks rerun, and the superseded outputs retained in [evidence-pre-death-reset-fix](evidence-pre-death-reset-fix/). Final figures above use the corrected checks.

## Concrete mechanisms within the zero-token constraint

1. **Exploration:** retain tested transitions and failed hypotheses in the model's existing Python simulator; select distinguishing probes in code when candidate models are already available. This can prevent repeated probes without another model call. There is no demonstrated generic mechanism that halves the work of inducing missing rules.
2. **Deaths/RESETs:** execute only simulator-validated paths with explicit hazard and remaining-action state; refresh observations and replan using that simulator after divergence. It helps when the model is correct; an incorrect simulator certifies incorrect plans. Track reset semantics rather than assuming reset erases every hidden state.
3. **Execution:** compile the existing planner's route, stop immediately on death/level completion, and simplify cyclic moves through already-validated inverses. TR87 supplies a concrete receipt. Model-owned planner execution already exists; adding another planner does not automatically remove rule induction.
4. **No-ops/staleness:** dedupe known transitions with phase/animation-aware state, retain an override, and refresh local state references after every action. Avoid hard suppression from identical interior pixels. Existing stale and batch guards already refuse actions; a more aggressive refusal can add reasoning/recovery tokens.

No production changes were made. These are mechanisms and qualification requirements, not claims of measured online gains. Routes/engine source used for diagnosis must not become a hidden-game oracle.

## Hidden-score decision and limits

If hidden relative headroom resembled even the largest observed public ceiling, a 27 hidden baseline would rise only to **{27*(1+maxrel/100):.3f} (+{maxrel:.3f}%)**; transferring the original control's ceiling gives about **27.378 (+1.399%)**. These are conditional illustrations, not hidden forecasts. A >=20% lift from 27 needs **5.4 hidden points** and at least **{20/maxrel:.3f} times** the largest public relative headroom. We do not have hidden per-level actions, baselines or completed-depth credit to test that possibility. There is no rigorous hidden-specific <20% bound in these public logs.

Therefore **NO for treating action polishing on solved levels as a demonstrated >=20% hidden lever**. The public bound is small even under omniscient, unrealistic execution. Avoiding a death or saving decode time could unlock new levels; those gains are outside this fixed-completion question and are unknown, not zero. The findings support prioritizing earlier correct rule induction and additional completions over generic post-plan action polishing.

## Verification, scope, and reproduction

Checked **125 game histories, {v['frames_compared']:,} per-action frames and level counts**, **525 solved level observations**, **500 game/scenario calculations against the installed scorer**, and **15 source-state-matched certified oracle routes**. Composed deletion receipts are in [chain-checks.jsonl](chain-checks.jsonl); five exact-checkpoint model-plan finishes are in [plan-audit.json](plan-audit.json). Evidence: [verification.json](verification.json), [input-manifest.json](input-manifest.json), [source-manifest.json](source-manifest.json). No training, model calls, spending, GPU launches or competition submissions occurred. Authored files are within this directory, and the summary is posted through the brief's prescribed board operation. One initial dependency import created a temporary Matplotlib cache under `/tmp`; subsequent imports were redirected into this directory's `.runtime-cache`.

From the repository root, run these commands in order (all Python invocations use `-B`):

```bash
python3 -B analysis/action-efficiency-20261013/analyze.py
solvers/tycho_nooa/.venv/bin/python -B analysis/action-efficiency-20261013/exact_bfs.py
solvers/tycho_nooa/.venv/bin/python -B analysis/action-efficiency-20261013/replay_audit.py
solvers/tycho_nooa/.venv/bin/python -B analysis/action-efficiency-20261013/chain_audit.py
solvers/tycho_nooa/.venv/bin/python -B analysis/action-efficiency-20261013/plan_audit.py
solvers/tycho_nooa/.venv/bin/python -B analysis/action-efficiency-20261013/verify.py
python3 -B analysis/action-efficiency-20261013/finalize.py
```

`auxiliary-observations.json` records the transcript-block and final-suffix counters, recomputed by `finalize.py`. `plan-candidates.jsonl` is unvalidated triage material: analysis transcripts can span a level boundary, so automatic candidate labels are not used as evidence of when a rule was known.

Setup exception: the first shared-board inventory command used `python3` without `-B`; later Python commands use `-B`.
'''
    (HERE/'FINDINGS.md').write_text(content)
    if (HERE/'board-receipt.json').exists():
        with (HERE/'FINDINGS.md').open('a') as f:
            f.write('\nThe prescribed one-paragraph summary was posted as `codex-action-efficiency` in shared-board thread **685**; receipt: [board-receipt.json](board-receipt.json).\n')
    artifacts=['analyze.py','replay_audit.py','exact_bfs.py','chain_audit.py','plan_audit.py','verify.py','finalize.py',
               'RESULT.json','FINDINGS.md','accounting.json','levels.jsonl','games.jsonl','actions.jsonl',
               'verification.json','deletion-checks.jsonl','chain-checks.jsonl','chain-summary.json','plan-audit.json',
               'exact-bfs.json','phase-accounting.jsonl','auxiliary-observations.json']
    if (HERE/'board-receipt.json').exists():
        artifacts.extend(['board-receipt.json','board-summary.txt'])
    (HERE/'artifact-manifest.json').write_text(json.dumps([dict(path=n,sha256=sha(HERE/n)) for n in artifacts],indent=2)+'\n')
    print(json.dumps(dict(answer=result['answer'],coverage=result['coverage'],max_public_relative_ceiling=maxrel)))


if __name__=='__main__':
    main()
