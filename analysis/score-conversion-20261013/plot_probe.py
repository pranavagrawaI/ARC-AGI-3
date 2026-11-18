"""Compare past fit against later behavior and search horizon; no inference."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT=Path(__file__).resolve().parent
j=json.loads((OUT/'results.json').read_text())
assert j['status']=='COMPLETE'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,(ax,bx)=plt.subplots(1,2,figsize=(14,8),gridspec_kw={'width_ratios':[1.35,1]})
fig.subplots_adjust(left=.07,right=.98,top=.78,bottom=.36,wspace=.34)
fig.suptitle('Passing the past is not enough to turn a repair into score',x=.07,y=.96,ha='left',fontsize=19,weight='bold')
fig.text(.07,.895,'s5i5 L2 • Retrospective CPU tests of archived code and a human-authored one-rule repair',fontsize=11,color='#555555')

groups=['past_at_checkpoint','later_same_run','other_selected_runs']
labels=['Known past\n13 changing transitions','Later in same run\n2 changing transitions','Other selected runs\n120 changing transitions']
colors=['#b45540','#c49323','#2f766d']
names=['Original predictor','One-rule repair','Successful-run reference']
for index,(key,name,color) in enumerate(zip(['original','one_rule','reference'],names,colors)):
    matches=[];totals=[]
    for group in groups:
        rows=[r for r in j['coverage'][group][key]['rows'] if r['input']!=r['actual']]
        totals.append(len(rows));matches.append(sum(r['status']=='matched' for r in rows))
    x=np.arange(3)+(index-1)*.24
    values=np.array(matches)/np.array(totals)*100
    ax.bar(x,values,width=.22,color=color,label=name)
    for xx,v,n,d in zip(x,values,matches,totals):ax.text(xx,v+2,f'{n}/{d}',ha='center',fontsize=9)
ax.set_ylim(0,118);ax.set_yticks([0,25,50,75,100]);ax.set_ylabel('Exact successor predictions (%)')
ax.set_xticks(range(3),labels,fontsize=9)
ax.set_title('Repairing the witnessed error overfits the available checks',loc='left',fontsize=12,pad=20)
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.24),frameon=False,ncol=1,fontsize=10)
ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)

bx.set_title('Even the reference model needs a sufficient horizon',loc='left',fontsize=12,pad=20)
for y,label in enumerate(['b1-r1','b1-r2']):
    for x,horizon in enumerate([27,31,44]):
        rec=next(r for r in j['search'] if r['run']==label and r['horizon']==horizon)
        found=rec['status']=='FOUND'
        bx.add_patch(plt.Rectangle((x-.46,y-.36),.92,.72,color='#d1e8e1' if found else '#f1dfd7'))
        bx.text(x,y, f"{len(rec['path'])}-click path" if found else 'No path\nin bound',ha='center',va='center',fontsize=11)
bx.set_xlim(-.5,2.5);bx.set_ylim(1.5,-.6)
bx.set_xticks(range(3),['27','31','44']);bx.set_xlabel('Maximum search depth (actions)')
bx.set_yticks([0,1],['B1 r1\nafter 22 actions','B1 r2\nafter 46 actions'],fontsize=10)
for spine in bx.spines.values():spine.set_visible(False)
bx.tick_params(length=0)
fig.text(.57,.17,'Both recovery plans clear L2 in the native game.\nImmediate public-25 mean gain: +0.22 points each.\nLater-level benefit and model repair ability remain untested.',fontsize=11,linespacing=1.5)
fig.text(.07,.045,'No-op cases are excluded from the percentage bars: 12 total; reference abstains on all 12. One-rule repair falsely predicts change on 1.\nSelected, previously inspected cases; the reference uses a successful run. This is not an LLM treatment result or an independent transfer test.',fontsize=9,color='#555555',linespacing=1.5)
for ext in ['png','svg']:fig.savefig(OUT/f'repair-to-score.{ext}',dpi=180,facecolor='white')
print(OUT/'repair-to-score.png')
