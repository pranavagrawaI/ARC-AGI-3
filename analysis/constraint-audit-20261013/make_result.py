"""Assemble compact, evidence-bound results from the raw audit."""
import hashlib
import json
import statistics as S
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
data={k:json.loads((HERE/(k+'_metrics.json')).read_text()) for k in ['c14','b1_c14','base_c1','base_pc']}
c=data['c14'];gs={g['game']:g for g in c['games']}
def exposure(levels):
    levels=set(map(tuple,levels)); games={g for g,L in levels}
    return dict(games=len(games),game_denominator=25,game_fraction=len(games)/25,
                levels=len(levels),level_denominator=c['reached_levels'],level_fraction=len(levels)/c['reached_levels'],
                affected_games=sorted(games),affected_levels=sorted(levels),
                direct_remaining_level_ceiling=sum(gs[g]['total_levels']-gs[g]['levels'] for g in games),
                direct_public_mean_score_ceiling_pp=sum(100-gs[g]['score'] for g in games)/25,
                ceiling_assumption='All affected games become perfect; all other outcomes fixed. Includes downstream levels and existing action efficiency. Not a prediction or a global counterfactual upper bound with shared scheduling.')
constraints=[]
def add(id,levels,evidence,classification,bound,confidence,removal=None,smoke=None,weight=None):
    row=dict(id=id,exposure=exposure(levels),evidence=evidence,classification=classification,
             gain_bound=bound,confidence=confidence,proven_positive_gain_lower_bound=0,minimal_removal=removal,smoke_40min=smoke)
    if weight is not None:
        row['ranking_confidence_weight']=weight
        row['ranking_index']=row['exposure']['direct_public_mean_score_ceiling_pp']*weight
    constraints.append(row)
alllv=c['request_levels'];ev='analysis/constraint-audit-20261013/c14_metrics.json'
add('hard_output_ceiling',c['length_levels'],[ev+'#/length_rows',ev+'#/completion_tokens'],
    'Actually hits 12288-token ceiling; 31/2787 responses, 30 without parsed calls; continuation exists, net harm unproven.',
    dict(cap_hit_tokens=380928,share_of_successful_response_tokens=380928/c['generated_tokens'],
         interpretation='At most 7.02% of measured response tokens are inside cap-hit replies. This bounds that token pool, not score gain; retained reasoning is not necessarily wasted.'),
    'High for ceiling incidence; medium-low for useful gain.',
    'Raise LOCAL_ANALYZER_MAX_OUTPUT 12K -> 16K and analyzer context 92K -> 96K together, preserving 81408-token input budget; keep server 100K, 14 streams, model and prompts fixed.',
    'One Claude-coordinated <=2400s smoke including startup. Replay 8 saved pre-length inputs in paired 12K/16K batches with counterbalanced order, then short public gameplay if time remains. Require >=1 incomplete call/thinking-only cutoff becomes a valid Python call, valid tool execution, unchanged input-budget receipt, no context rejection/retraction/OOM. Score efficacy remains unknown. Reject if only longer reasoning or more repeated cutoffs.',0.5)
add('context_eviction',c['eviction_levels'],[ev+'#/text_evictions',ev+'#/opening_losses',
    'analysis/constraint-audit-20261013/b1_c14_metrics.json#/opening_losses'],
    '168 text evictions; 54 first level openers lost on 54 levels in 23 games; no rolling summaries or server-overflow recoveries observed. B1 mitigates opening loss, not all text loss.',
    dict(eviction_events=168,median_messages_lost=39,median_text_chars_lost=109908,
         lost_openers=54,probation_b1_lost_openers=2,causal_score_gain=None,
         interpretation='Information loss is observed. Whether it removed correct rules needed later is not established.'),
    'High for physical deletion; low for score harm, with null/confounded bundle comparisons.',
    'Reduce ARC3_CONTEXT_DRAIN_TOKENS 40K -> 20K, keep input/output limits and 14 streams fixed. This removes excess eviction rather than adding memory prompts, summaries or tools. Use retained c14 or B1 source explicitly; do not call the existing pin new.',
    'One Claude-coordinated <=2400s smoke including startup. CPU-replay logged trim-trigger inputs first to require strictly more old evidence retained with valid user/tool pairing and budget. GPU replay same selected game checkpoints with one knob changed; instrument evictions, uncached prefill, throughput, retractions and same-input rule recall. Require retained factual evidence actually used, no corruption/overflow/retractions, aggregate throughput within 5% of control. No trim exercised => INVALID_EMPTY. Forty minutes cannot establish hidden gain.',0.25)
tls=[(x['game'],x['level']) for x in c['tool_rows'] if x['timed_out']]
add('sandbox_timeout_and_helper_erasure',tls,[ev+'#/tool_rows','analysis/constraint-audit-20261013/retention_replay.json',
    'solvers/m2_b1/src/inference/agent/tool_agent.py:4698'],
    '9 hard 30s timeouts; all 9 erase previous helpers, 256 helper-name loss notices. B1 already preserves snapshots and uses 60s calls; its 3 timeouts erase 0 helpers.',
    dict(timeout_seconds=270,helper_loss_notices=256,affected_terminal_unsolved_levels=3,
         retained_c14_gain=None,probation_b1_additional_helper_preservation_gain=0,
         interpretation='270s bounds time spent in the nine timeout calls, not rewrite cost or recoverable GPU time. Overall score effect unknown.'),
    'High for loss/replay, medium for avoidable relearning cost, unknown for score.',
    'Extract the existing B1 check: if keepable_functions is absent, retain last committed helper set. Commit explicit empty sets when the child actually reports them. Do not raise timeout or add the evidence toolbox in the same comparison. Already fixed on B1.',
    'One Claude-coordinated <=2400s smoke including startup. In the real sandbox, define a helper, force a short interrupted CPU call, then call the helper and verify any already executed actions remain counted. Also verify an explicit completed empty snapshot removes helpers. Require zero erased helpers after missing snapshots, valid subsequent call and no action rollback. Include sk48/cn04 late checkpoints if available; ordinary gameplay with no timeout is not qualification.',0.75)
add('per_game_wall_cap',[(g['game'],g['levels']+1) for g in c['games'] if g['state']=='gave_up'],
    [ev+'#/games','analysis/constraint-audit-20261013/c14_notebook.py:10728'],
    '17/25 public games end at 7254s plus bounded overrun; hidden branch sets 31920s, not 7254s.',
    dict(public_max_extra_levels=76,observed_levels_last_30min=10,public_linear_40min_tail_extrapolation_levels=13.3333333333,
         hidden_gain=None,interpretation='Tail extrapolation is a scenario, not upper bound. Public per-game cap cannot be blamed for hidden 9h results.'),
    'High for public censoring; no hidden per-game attribution.')
add('startup',alllv,[ev+'#/startup','analysis/constraint-audit-20261013/c14_notebook.py:10103'],
    'All games lose the same approximately 558.6s notebook-to-benchmark delay.',
    dict(startup_s=558.624376436,nine_hour_time_fraction=558.624376436/32400,
         optimistic_playtime_gain=558.624376436/(32400-558.624376436),score_gain=None),
    'High for this public log; hidden startup timing unlogged. Deleting all startup is impossible.')
add('no_action_and_yield',c['touched_levels']['no_action_invocations'],[ev+'#/counters',ev+'#/event_counts'],
    '680/1882 analyzer invocations have no action, but 674 token-budget yields continue work. Deduplicating reused analysis_step gives only 16/1218 logical turns that never act.',
    dict(never_acted_tokens=228837,share_of_response_tokens=228837/c['generated_tokens'],
         theoretical_pool_playtime_gain=228837/(c['generated_tokens']-228837),
         interpretation='Terminal unbanked reasoning pool equals response-minus-history token difference. Not proof of waste; no-action computation can establish rules.'),
    'High after deduplication; no gain proven for deleting yields.')
add('tool_errors',c['touched_levels']['tool_errors'],[ev+'#/errors',ev+'#/tool_rows'],
    '257 traceback-bearing results, plus timeout/error-only results. Unique visible request results have 278 errors out of 2742; 60 NameError, 25 ImportError, 42 StaleStateActionError tracebacks.',
    dict(causal_score_gain=None,interpretation='Error count is not decode-time loss. Correct models and intentional guards can also error.'),
    'High for counts, low for a general removal; time import permission is already relaxed by B1.')
add('tool_output_limit',[(x['game'],x['level']) for x in c['tool_rows'] if x['truncated']],
    [ev+'#/tool_rows','analysis/constraint-audit-20261013/c14_notebook.py:10179'],
    '3072-token tool output ceiling truncates 3 visible results in 2 games/2 levels.',
    dict(causal_score_gain=None,interpretation='Only two levels directly expose this limit in this run; opening history and image data are separate.'),
    'High for count, low for consequential information loss.')
add('model_request_timeout',c['touched_levels']['request_timeout'],[ev+'#/examples/request_timeout'],
    '10 read timeouts with 5.86-78.49s remaining-time limits, all below the configured 900s analyzer ceiling; history preserved.',
    dict(gain_from_removing_independent_900s_ceiling=0,overlap_with_per_game_wall_cap=True),
    'High for logs; no independent 900s ceiling observed.')
add('prompt_and_image_representation',alllv,[ev+'#/png_dimensions',ev+'#/image_count',
    'analysis/m2-winners-20261001/audit/dfranzen-repo/ARC3-Inference/inference/agent/vision_context.py:71'],
    'Full 64x64 grid rendered losslessly as 640x640 nearest-neighbor PNG; ASCII and segmentation accessible. One request carries median 17 historic images, not just current frame. Raw numeric frame API absent, but recoverable from ASCII.',
    dict(proven_missing_current_frame_cells=0,score_gain=None,interpretation='Full frame is available; model perception/rule inference and lost animation facts are not proven adequate. Oracle-route result supports execution, not universal perception competence.'),
    'High for submitted PNG/ASCII representation; unknown for model internal perception.')
add('action_api',alllv,[ev+'#/batch_sizes',ev+'#/batched_calls'],
    'Multiple actions per Python call and multiple calls per snippet already supported; 292 multi-action batches, max 36 actions.',
    dict(gain_from_adding_batch_support=0,guard_gain=None),
    'High; removing transition-stop or stale/no-op guards without a false-refusal trace is unsupported.')
add('game_over_handling',c['event_levels']['game_over'],[ev+'#/event_counts','analysis/constraint-audit-20261013/c14_notebook.py:10133'],
    '25 deaths on 16 levels in 10 games; auto RESET preserves completed levels and fatal-action metadata. No terminal game_over/crashed games in this cohort.',
    dict(gain_from_fixing_premature_game_over_termination=0,death_avoidance_gain=None),
    'High for no premature terminal GAME_OVER here; cannot bound preventable deaths without counterfactual policies.')
add('level_up_memory_loss',[(x['game'],x['level']) for x in c['level_transitions']],
    [ev+'#/level_transitions','analysis/constraint-audit-20261013/c14_notebook.py:10208'],
    'All 99 observed nonfinal level transitions retain earlier-level user text. 9 also trim old text; helpers scoped to game. No wholesale level-up wipe observed.',
    dict(gain_from_fixing_wholesale_level_up_history_wipe=0,overlap_with_context_eviction=True),
    'High for visible history; arbitrary snippet locals do not persist between ANY calls.')
add('end_of_run_idle_and_stream_cap',alllv,[ev+'#/serve','analysis/model-serving-20261011/wave-1/RESULT.md'],
    'Eight games win early but 17 unfinished games remain for 14 slots. Decode continues until 2.21s before benchmark end. Median 13 running requests; no retractions. KV p90 0.89 and capacity 911424 tokens constrain larger context/concurrency.',
    dict(trailing_no_logged_decode_s=2.2132139205932617,actual_idle_gpu_s=None,tail_fraction=2.2132139205932617/(c['serve']['benchmark_end']-c['serve']['benchmark_start']),
         spare_request_slot_fraction=0.08936662678252683,
         optimistic_linear_16_vs_14_streams_token_gain=16/14-1,
         interpretation='Trailing interval after last logged decode is not direct GPU idle measurement. Spare-slot seconds and early-finisher game-seconds are NOT idle GPU seconds. 16/14 is only a fixed per-stream-rate scenario; per-stream slowdown and KV losses invalidate it as a real upper bound.'),
    'High for public tail/occupancy; hidden tail utilization unknown, no verified new breadth gain.')
ranking=sorted([x for x in constraints if 'ranking_index' in x],key=lambda x:x['ranking_index'],reverse=True)
for i,x in enumerate(ranking,1):x['rank']=i
result=dict(status='COMPLETE_OFFLINE_AUDIT',actor='codex-constraint-audit',date='2026-10-13',
    authorized_operations=dict(spend_usd=0,training=False,gpu_launches=0,submissions=0,write_directory=str(HERE.relative_to(ROOT))),
    scope='Four complete public-25 runs (100 game-runs); retained c14 and probation B1+c14 distinguished from old base controls. No hidden trajectories available.',
    main_finding='No supported constraint-removal route to a 1.4x hidden score. Actual losses remain, but causal gain is unmeasured and two are already mitigated by probation B1.',
    premise_corrections=['Current retained source is c14/92K, not ten-stream raw base.',
                        'Measured SGLang runs load Intel AutoRound W4A16, not NVFP4; stack path and server_args establish this.',
                        'Score formula includes x100 and level weights; level count is not score.'],
    ranking_method='Direct affected-game public-score opportunity ceiling x explicit subjective confidence weight. 0.25=observed deletion with weak harm attribution; 0.5=direct cutoff with preserved continuation; 0.75=CPU-reproduced loss plus raw notices. These are prioritization weights, NOT measured probabilities, predictions, or causal score estimates. Cross-game scheduling effects excluded from direct ceilings.',
    ranked_top3=[dict(id=x['id'],rank=x['rank'],index=x['ranking_index']) for x in ranking],constraints=constraints,
    cohort_summary={k:{field:v[field] for field in ['evidence','completed_levels','reached_levels','generated_tokens','requests','counters','event_counts','curve','states']} for k,v in data.items()},
    supplementary_current_campaign=json.loads((HERE/'supplementary_checks.json').read_text()),
    unknowns=['Hidden per-game timelines, effective startup/cancellation and final GPU utilization.',
              'Score gain from any remaining removal; exposure is not failure attribution.',
              'Correct mechanistic facts lost at drains versus wrong hypotheses; full long-term raw environment history is still accessible.',
              'GPU forward-pass utilization, prefill/decode/vision timing decomposition and impact of duplicated historical images.',
              'Exact installed-byte source identity in hidden draws; local source and logs support mechanisms but no hidden hash receipt.',
              'Public-to-hidden transfer and stochastic variability; four runs are not randomized ablations.'],
    rejects=['New memory/world-model/critic/planner scaffolds','Unqualified wholesale context reset','Adding action batching that already exists','Treating yields as discarded reasoning','Calling early-finisher game-time idle GPU time','Rebranding existing B1 fixes as new discoveries'],
    reproduction=['python3 -B analysis/constraint-audit-20261013/audit.py','python3 -B analysis/constraint-audit-20261013/retention_replay.py','python3 -B analysis/constraint-audit-20261013/make_result.py'])
result['validation_artifact']='analysis/constraint-audit-20261013/validation.json'
if (HERE/'board-receipt.json').exists():
    receipt=json.loads((HERE/'board-receipt.json').read_text())
    result['board_post']=dict(thread=receipt['event']['thread'],seq=receipt['event']['seq'],
                            timestamp=receipt['event']['ts'],receipt_path='analysis/constraint-audit-20261013/board-receipt.json')
(HERE/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
paths=set()
for v in data.values():
    out=ROOT/v['evidence'];paths.add(out/'benchmark.json');paths.add(out/'serve.log')
    paths.update(out.glob('*_requests.jsonl'));paths.update((out/'transcripts').glob('*.txt'));paths.update((out/'artifacts').glob('*events.jsonl'))
paths.update(ROOT/p for p in ['incumbent/stack.json','analysis/model-serving-20261011/wave-1/e2-c14/kernel/wave1-e2-c14-20261011.ipynb',
    'analysis/model-serving-20261011/wave-2/b1-c14/kernel/wave2-b1-c14-20261011.ipynb',
    'analysis/m2-winners-20261001/audit/dfranzen-repo/ARC3-Inference/inference/agent/tool_agent.py',
    'solvers/m2_b1/src/inference/agent/tool_agent.py'])
for rel in ['analysis/pruning-transfer-20261011/control/output','analysis/pruning-transfer-20261011/candidate2/output']:
    paths.update(ROOT/rel/f for f in ['benchmark.json','serve.log'])
manifest=[]
for p in sorted(paths):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    manifest.append(dict(path=str(p.relative_to(ROOT)),size_bytes=p.stat().st_size,sha256=h.hexdigest()))
(HERE/'evidence_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(ranking=result['ranked_top3'],hashed_files=len(manifest)),indent=2))
