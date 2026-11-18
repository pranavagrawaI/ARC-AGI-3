"""CPU-only readout of the frozen D16 public screen and historical public references."""
import collections, datetime, hashlib, json, re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
SOURCES={
 'D16 merge':'solvers/m2_f1x_reap/experiment/screen-1/merge/output/benchmark.json',
 'F1x':'solvers/m2_f1/experiment/screen-2/f1x/output/benchmark.json',
 'Raw control':'solvers/m2_lexicon/experiment/screen-1/rawctl/output/benchmark.json',
}
checks=['solvers/m2_f1x_reap/experiment/screen-1/RESULT.json','solvers/m2_f1/experiment/screen-2/RESULT.json','solvers/m2_f1x_reap/experiment/screen-1/merge/output/serve.log','solvers/m2_f1x_reap/experiment/screen-1/merge/kernel/m2-f1x-reap14-screen1.ipynb']
runs={}; records={}; shared_ids=None
for name,rel in SOURCES.items():
 b=json.loads((ROOT/rel).read_text());games=b['game_runs'];g={x['game_id']:x for x in games};assert len(g)==len(games)==25
 if shared_ids is None:shared_ids=set(g)
 assert set(g)==shared_ids
 assert all(0<=x['levels_completed']<=x['number_of_levels'] for x in games)
 progress=[]
 for minute in range(0,61,5):
  levels=actions=0
  for x in games:
   history=x['history'];actions+=sum(y['wallclock_seconds']<=minute*60 for y in history)
   endpoint=0
   for length in x['actions_per_level'][:x['levels_completed']]:
    endpoint+=length;assert 0<endpoint<=len(history)
    levels+=history[endpoint-1]['wallclock_seconds']<=minute*60
  progress.append({'minute':minute,'levels':levels,'actions':actions})
 runs[name]={'date':b['start_time'][:10],'games':25,'levels':sum(x['levels_completed'] for x in games),'mean_score':sum(x['final_score'] for x in games)/25,'wins':sum(x['state']=='won' for x in games),'states':dict(collections.Counter(x['state'] for x in games)),'actions':sum(len(x['history']) for x in games),'progress':progress,'zero_games':sorted(x['game_id'].split('-')[0] for x in games if x['levels_completed']==0),'error_ending_notes':[x['solver_note'] for x in games if x['solver_note'] and not x['solver_note'].startswith('tokens=')]}
 records[name]=g
comparisons={}
for name in ['F1x','Raw control']:
 diffs=[records['D16 merge'][key]['levels_completed']-records[name][key]['levels_completed'] for key in shared_ids]
 comparisons[name]={'ahead':sum(x>0 for x in diffs),'tied':sum(x==0 for x in diffs),'behind':sum(x<0 for x in diffs),'total_level_delta':sum(diffs)}
 assert sum(diffs)==runs['D16 merge']['levels']-runs[name]['levels']
per_game=[{'game':key.split('-')[0],'total_levels':records['D16 merge'][key]['number_of_levels'],**{name:{'levels':records[name][key]['levels_completed'],'score':records[name][key]['final_score']} for name in SOURCES}} for key in sorted(shared_ids)]
result=json.loads((ROOT/checks[0]).read_text());assert result['arm']['levels']==runs['D16 merge']['levels']==80
assert runs['D16 merge']['progress'][-1]['levels']==80
log=(ROOT/checks[2]).read_text(errors='replace').splitlines()
warnings=[l for l in log if 'device-loaded after serving started' in l]
fatal=[l[:300] for l in log if re.search(r'OutOfMemoryError|CUDA error:|Traceback \(most recent|Watchdog timeout',l)]
public={'runs':runs,'comparisons':comparisons,'per_game':per_game,'screen_serving':result['serving'],'late_kernel_warning_count':len(warnings),'late_kernel_warnings':warnings,'fatal_pattern_matches':fatal,'scope':'Same 25 game IDs, one 60-minute offline screen each; historical references, no contemporaneous randomized pair. Not the hidden leaderboard game set or full competition runtime.','source_sha256':{rel:hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() for rel in [*SOURCES.values(),*checks]}}
assert public['source_sha256'][checks[3]]=='9d634d6accb024d53b875106c4d77ad12948d895ea485cd4be44b7b4f83f56c0'
(OUT/'public25.json').write_text(json.dumps(public,indent=2)+'\n')
print(json.dumps({k:public[k] for k in ['runs','comparisons','late_kernel_warning_count','fatal_pattern_matches']},indent=2))
