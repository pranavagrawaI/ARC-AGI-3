"""Plot observed model repair separately from live completion outcomes."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE=Path(__file__).resolve().parent/'transcript-audit'
report=json.loads((HERE/'repair-probe.json').read_text())
episodes=[f'b1-{seed}-{arm}' for arm in ['control','reminder','counterexample'] for seed in [1701,2903]]
columns=[('past_at_checkpoint',13,'Past observations\n13 changing transitions'),
         ('other_selected_runs',120,'Other inspected runs\n120 changing transitions'),
         ('native_verified_recovery_route',31,'Known recovery route\n31 transitions')]
fig,ax=plt.subplots(figsize=(10.6,6.2))
for y,ep in enumerate(episodes):
    for x,(subset,total,_) in enumerate(columns):
        count=report['models'][ep][subset]['counts']['matched']
        ratio=count/total
        color=plt.cm.GnBu(.08+.55*ratio)
        ax.add_patch(Rectangle((x-.47,y-.43),.94,.86,facecolor=color,edgecolor='white'))
        ax.text(x,y,f'{count}/{total}',ha='center',va='center',fontsize=15,color='#102c3b')
labels=[f"{'Ordinary' if 'control' in ep else 'Reminder' if 'reminder' in ep else 'Counterexample'} / {ep.split('-')[1]}" for ep in episodes]
ax.set(xlim=(-.52,2.52),ylim=(5.55,-.6),xticks=range(3),xticklabels=[c[2] for c in columns],yticks=range(6),yticklabels=labels)
ax.xaxis.tick_top();ax.tick_params(length=0,pad=9,labelsize=11)
for spine in ax.spines.values():spine.set_visible(False)
fig.suptitle('A zero score gain hid substantial simulator repair',fontsize=17,x=.03,ha='left',y=.98)
fig.text(.03,.915,'Exact next-state predictions from each B1 continuation’s frozen final simulator. All six live runs failed L2.',fontsize=11)
fig.text(.03,.095,'The reminder / 2903 model also predicts a native-verified 29-action win from its final state.\nIts actual search stopped at depth 24.',fontsize=11,color='#143e51',weight='bold')
fig.text(.03,.025,'Retrospective diagnostic data, previously inspected; not a held-out efficacy test. No-op abstentions excluded:\n5 past observations and 6 transitions from other runs. Route predictions are tested one transition at a time.',fontsize=9,color='#444444')
fig.subplots_adjust(left=.235,right=.98,top=.78,bottom=.21)
fig.savefig(HERE/'learning-without-score.png',dpi=170)
fig.savefig(HERE/'learning-without-score.svg')
plt.close(fig)

rows=[]
for arm in ['control','reminder','counterexample']:
    paths=[HERE/f'{checkpoint}-{seed}-{arm}.json' for checkpoint in ['b1','control'] for seed in [1701,2903]]
    turns=[t for p in paths for t in json.loads(p.read_text())]
    full=sum(t['tokens'] for t in turns if not t['tools'] and t['tokens']==12288)
    partial=sum(t['tokens'] for t in turns if not t['tools'] and t['tokens']!=12288)
    rows.append((arm,full,partial,131072-full-partial))
fig,ax=plt.subplots(figsize=(9,3.8))
colors=['#b65240','#dec2ac','#c5d9df']
names=['Full 12,288-token cutoff, no tool call','Final partial reply, no tool call','Replies containing a tool call']
for y,(arm,*values) in enumerate(rows):
    left=0
    for k,v in enumerate(values):
        ax.barh(y,v/131072*100,left=left,color=colors[k],height=.6,label=names[k] if y==0 else None)
        if v/131072>.08:ax.text(left+v/131072*50,y,f'{v/131072:.0%}',ha='center',va='center',fontsize=11)
        left+=v/131072*100
ax.set(yticks=range(3),yticklabels=['Ordinary','Reminder','Counterexample'],xlim=(0,100),xlabel='Share of generated tokens (131,072 per arm; four continuations)')
ax.invert_yaxis()
ax.set_title('Eight full-length replies ended before emitting a tool call',loc='left',fontsize=15,pad=18)
ax.spines[['top','right','left']].set_visible(False)
handles,labels=ax.get_legend_handles_labels()
fig.legend(handles,labels,loc='lower left',bbox_to_anchor=(.02,.12),frameon=False,fontsize=9)
fig.text(.02,.02,'Reasoning may inform later replies; these categories do not measure causal waste. Tool-bearing replies include code and reasoning.',fontsize=8)
fig.subplots_adjust(left=.16,right=.97,top=.8,bottom=.46)
fig.savefig(HERE/'token-use.png',dpi=170)
fig.savefig(HERE/'token-use.svg')
plt.close(fig)
