"""Synthesize the audited failure stages without claiming prevalence."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
T=json.loads((OUT/'tn36/engine-replay.json').read_text())
P=json.loads((OUT/'tn36/encoding-check.json').read_text())
C=json.loads((OUT/'request-context-audit.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})

fig=plt.figure(figsize=(11,8))
fig.text(.07,.95,'The experiment observed less than the model concluded.',fontsize=20,fontweight='bold',va='top')
fig.text(.07,.895,'tn36 L2 · engine replay matches all 387 recorded frames across four attempts',fontsize=11,color='#58666d',va='top')
op=T['missed_rotation_probe']['operation']
for i,(field,title) in enumerate([('before_shape','Before command'),('after_shape','During command')]):
    ax=fig.add_axes([.17+i*.42,.55,.24,.25])
    data=np.array(op[field]);grid=(data>=0).astype(int)
    ax.imshow(grid,cmap=ListedColormap(['#edf0f2','#ffde33']),vmin=0,vmax=1,interpolation='nearest')
    ax.set_title(title,fontweight='bold',fontsize=11,pad=10)
    ax.set_xticks([]);ax.set_yticks([])
    ax.set_xlabel(f"Position (45,24) · rotation {op['before' if i==0 else 'after']['rotation']}°",fontsize=10,labelpad=10)
    for spine in ax.spines.values():spine.set_visible(False)
fig.text(.07,.47,'The key did not translate. It rotated 90°. That rotated shape was present in the captured animation.',fontsize=10,color='#58666d')

labels=['Top bar only','Bottom stem only','Both switched on']
outcomes=['LEFT','No operation','UP']
for i,(probe,label,outcome) in enumerate(zip(P['probes'],labels,outcomes)):
    ax=fig.add_axes([.08+i*.30,.16,.22,.22])
    dy,dx=probe['delta_row_col']
    ax.scatter([0],[0],color='#8a949b',s=35)
    if dx or dy:
        ax.annotate('',xy=(dx,dy),xytext=(0,0),arrowprops={'arrowstyle':'->','color':'#176f67','lw':2})
    else:ax.scatter([0],[0],facecolor='none',edgecolor='#176f67',s=110)
    ax.set_xlim(-5,1);ax.set_ylim(1,-5);ax.set_aspect('equal')
    ax.set_xticks([-4,0]);ax.set_yticks([-4,0]);ax.grid(color='#edf0f2')
    ax.set_title(label+' → '+outcome,fontsize=10,fontweight='bold',pad=9)
    ax.set_xlabel('Column displacement');ax.set_ylabel('Row displacement')
fig.text(.07,.07,'Three offline mouse-action probes: the combined effect is not the sum of the isolated effects.',fontsize=9,color='#58666d')
fig.text(.07,.043,'These probes were not supplied to the model. They falsify the independent-row interpretation for this game.',fontsize=9,color='#58666d')
fig.savefig(OUT/'tn36-measurement.png',dpi=170,facecolor='white')
fig.savefig(OUT/'tn36-measurement.svg',facecolor='white')
plt.close(fig)

# Plot request message counts, showing actual eviction despite zero B1 counters.
fig,axs=plt.subplots(1,3,figsize=(12,4.3))
fig.subplots_adjust(left=.07,right=.96,top=.72,bottom=.24,wspace=.30)
fig.text(.07,.94,'Zero reported trims did not mean zero history loss.',fontsize=19,fontweight='bold',va='top')
fig.text(.07,.86,'B1 r1 · sampled adjacent requests at each observed L2 history reduction · opening message retained unchanged',fontsize=10,color='#58666d',va='top')
for ax,game in zip(axs,['cn04','s5i5','tn36']):
    drops=C['runs']['b1-r1/'+game]['observed_history_reductions_l2']
    for i,d in enumerate(drops):
        ax.plot([i-.13,i+.13],[d['messages_before'],d['messages_after']],color='#aab2b8',lw=1)
        ax.scatter([i-.13],[d['messages_before']],color='#b85b36',s=35)
        ax.scatter([i+.13],[d['messages_after']],color='#176f67',s=35)
    ax.set_title(game+' · B1 counter = 0',fontweight='bold',fontsize=11)
    ax.set_ylim(0,150);ax.set_xlim(-.5,len(drops)-.5)
    ax.set_xticks(range(len(drops)),[f"{d['messages_before']}→{d['messages_after']}" for d in drops],fontsize=9)
    ax.set_xlabel('Observed reductions in request order')
    ax.set_ylabel('Messages in request');ax.grid(axis='y',color='#edf0f2')
fig.text(.07,.105,'Counter scope: request-time trimming. Real commit-time eviction can remove messages without incrementing that counter.',fontsize=9,color='#58666d')
fig.text(.07,.055,'These counts measure retained messages, not information quality. Eviction and waiting are not evidence of model error by themselves.',fontsize=9,color='#58666d')
fig.savefig(OUT/'context-counter-correction.png',dpi=170,facecolor='white')
fig.savefig(OUT/'context-counter-correction.svg',facecolor='white')
plt.close(fig)

report={
 'scope':'Three selected game levels; exact local mechanisms, not failure prevalence or treatment efficacy',
 'verified_frames':102+237+sum(r['verified_actions'] for r in T['runs'].values()),
 'bounded_search_correction':{
  'source':'solvers/m2_b1/experiment/pair-1/b1-r1/output/transcripts/s5i5-18d95033_p0.txt',
  'definition_line':8190,'pruning_threshold':26,'maximum_checked_successor_depth':27,
  'validated_model_route_clicks':31,
  'statement':'The validated four-length model finds a31-click route, outside this BFS bound. The model defects are independently reproduced, but do not alone explain the bounded no-path result.'},
 'failure_stages':[
  {'stage':'Representation fidelity','case':'cn04 L2, B1 r1',
   'observation':'Independent normalization erases connector offsets; valid two-pair joins excluded.',
   'reproduction':'analysis/cn04-b1-inspection-20261012/geometry-check.json'},
  {'stage':'Reasoning-to-code fidelity','case':'s5i5 L2, B1 r1',
   'observation':'Correct coupling fact explicitly present in request 73, but growS does not implement it.',
   'reproduction':'analysis/s5i5-inspection-20261012/model-check.json',
   'prompt_evidence':C['feedback_checkpoint']},
  {'stage':'Measurement and experiment design','case':'tn36 L2, B1 r1',
   'observation':'No translation generalized to no effect, although rotation was visible. Isolated bits treated as independent directions.',
   'reproduction':'analysis/failure-audit-20261013/tn36/encoding-check.json'},
  {'stage':'Updating beliefs','cases':['cn04 L2','s5i5 L2'],
   'observation':'Negative search results used to weaken a goal before validating candidate models, coverage or depth bounds. s5i5 BFS prunes paths longer than26 and checks successors up to27; the validated-model recovery route has31 clicks.'},
  {'stage':'Whole-plan feasibility','case':'s5i5 L2, B1 r2',
   'observation':'Hand-planned route ends above a blocking wall; direct descent is impossible, although a recovery route exists.'}],
 'counterexamples':[
  'Successful control generates bugs too, then validates and repairs them.',
  'Historical agreement does not guarantee unseen-transition correctness; prior sim5 fails on its next action.',
  'A correct verbal explanation is not proof of a correct implemented model.',
  'A successful action sequence is not proof of a fully correct verbal explanation.'],
 'measurement_corrections':{
  'previous_zero_trim_claim':'Withdrawn: request-time counters do not cover commit-time history eviction.',
  'preserved_originals':'analysis/failure-audit-20261013/prior-artifacts',
  'replicated_counter_failure':'analysis/failure-audit-20261013/context-counter-reproduction.json',
  'request_context_evidence':'analysis/failure-audit-20261013/request-context-audit.json',
  's5i5_critical_fact_missing_at_request_73':False,
  'admission_wait_partition':'Unknown: these records lack per-game enqueue/admit timestamps. Long action gaps must not be reported as continuous reasoning time.'},
 'learned':[
  'Validate the representation and measurement contract before treating a negative result as evidence about the task.',
  'The BFS checks paths only up to27 actions, so it does not rule out the validated-model 31-click recovery route; simulator defects are not the sole isolated explanation for that negative result.',
  'Preserving an opening is narrower than preserving later decisive experiments.',
  'Direct observation, code output, model interpretation and terminal success are distinct evidence types.'],
 'next_discriminating_test':{
  'checkpoint':'s5i5 B1 r1 original request-log line 73',
  'arms':['Ordinary continuation','Raw past-observation reminder','Explicit candidate-prediction counterexample'],
  'measure':['Repair tokens','Held-out transition prediction','Executed check coverage','Recovery action feasibility'],
  'why_three_arms':'Distinguish reminder/attention effects from identifying an actual contradiction in the program.',
  'status':'Prepared offline inputs only; no new model inference or competition run.'}}
(OUT/'findings.json').write_text(json.dumps(report,indent=2)+'\n')
print('verified frame total',report['verified_frames'],'figures',2)
