"""Figures grounded in saved traces, engine replay, and geometry reproduction."""
from pathlib import Path
import json
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np

OUT = Path(__file__).resolve().parent
R = json.loads((OUT/'records.json').read_text())
E = json.loads((OUT/'engine-replay.json').read_text())
G = json.loads((OUT/'geometry-check.json').read_text())
ORANGE, TEAL, GRAY = '#b85b36', '#176f67', '#aab2b8'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})


def save(fig, name):
    fig.savefig(OUT/(name+'.png'), dpi=170, facecolor='white')
    fig.savefig(OUT/(name+'.svg'), facecolor='white')
    plt.close(fig)


series = {}
for label, run in E['runs'].items():
    xs, matched, visible = [], [], []
    for row in run['states']:
        k = row['local_action']
        xs.append(0 if k==0 else R[label]['actions'][k-1]['minutes_since_entry'])
        sprites = row['completed_level_sprites'] or row['sprites']
        assert sum(s['contacts'] for s in sprites)==12
        matched.append(sum(s['matched'] for s in sprites))
        visible.append(sum(s['visually_joined'] for s in sprites))
    if not R[label]['solved']:
        xs.append(R[label]['level_minutes']); matched.append(matched[-1]); visible.append(visible[-1])
    assert R[label]['solved'] == (matched[-1]==12)
    series[label]={'minutes':xs,'matched':matched,'visible_gray':visible}

fig=plt.figure(figsize=(12,8.5))
fig.text(.07,.96,'Same starting board. Different evidence loops.',fontsize=20,fontweight='bold')
fig.text(.07,.917,'cn04 L2 · compatible matches verified in engine replay · 12 connector endpoints must qualify',fontsize=11,color='#58666d')
axs=[fig.add_axes([.08,.46,.4,.35]),fig.add_axes([.57,.46,.37,.35])]
for ax, xmax, title in zip(axs,[12,120],['First 12 minutes','Whole attempt']):
    for label,color in [('b1-r1',ORANGE),('b1-r2',TEAL)]:
        s=series[label]
        ax.step(s['minutes'],s['matched'],where='post',color=color,lw=2,label=label+' compatible')
    ax.step(series['b1-r1']['minutes'],series['b1-r1']['visible_gray'],where='post',color=ORANGE,lw=1.5,ls='--',label='r1 visually gray')
    ax.set_xlim(0,xmax);ax.set_ylim(-.4,12.7)
    ax.set_yticks([0,2,4,6,8,10,12]);ax.set_xlabel('Minutes since entering L2')
    ax.set_title(title,loc='left',fontweight='bold',pad=12)
    ax.grid(color='#edf0f2');ax.axhline(12,color=GRAY,lw=.8,ls=':')
axs[0].set_ylabel('Joined endpoints / 12')
axs[0].annotate('≈5 min: faulty search returns 0;\nmodel doubts the all-ends goal',xy=(5.1,0),xytext=(1.2,6.9),fontsize=8.8,
                arrowprops={'arrowstyle':'-','color':GRAY})
axs[0].annotate('r1: 6 gray, only 2 compatible',xy=(10.69,6),xytext=(5.0,10),fontsize=8.8,
                arrowprops={'arrowstyle':'-','color':GRAY})
axs[1].text(14,11.9,'r2 solves at 8.54 min',color=TEAL,fontsize=10)
axs[1].text(75,1,'r1 ends unfinished\nat 118.71 min',color=ORANGE,fontsize=10)
fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.52,.874),ncol=3,frameon=False,fontsize=9)

ax=fig.add_axes([.14,.17,.80,.16])
for label,y,color in [('b1-r1',1,ORANGE),('b1-r2',0,TEAL)]:
    times=[a['minutes_since_entry'] for a in R[label]['actions']]
    ax.plot([0,R[label]['level_minutes']],[y,y],color=GRAY,lw=.9)
    ax.scatter(times,[y]*len(times),color=color,s=26,zorder=3)
    ax.scatter([R[label]['level_minutes']],[y],color=color,marker='x' if label=='b1-r1' else '*',s=65,zorder=4)
gaps=[]
actions=R['b1-r1']['actions']
for a,b in zip(actions,actions[1:]):
    gap=b['minutes_since_entry']-a['minutes_since_entry']
    if gap>30:
        lo,hi=a['minutes_since_entry'],b['minutes_since_entry']
        ax.fill_between([lo,hi],.82,1.18,color=ORANGE,alpha=.12,zorder=0)
        ax.text((lo+hi)/2,1.35,f'{gap:.1f} min without an action',ha='center',fontsize=8.5)
        gaps.append({'from_local_action':a['local_action'],'to_local_action':b['local_action'],'minutes':gap})
ax.set_xlim(0,120);ax.set_ylim(-.55,1.75)
ax.set_yticks([1,0],['r1 · 35 actions','r2 · 39 actions'])
ax.set_xlabel('Executed action timestamps · minutes since L2 entry')
ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
fig.text(.08,.074,'Gray and compatible are different: the engine renders both contact types red, but qualifies matches by type.',fontsize=9,color='#58666d')
fig.text(.08,.047,'Intervals between actions include reasoning, tools and scheduling. They are not measured GPU-compute durations.',fontsize=9,color='#58666d')
save(fig,'01-evidence-loop')

# Show the exact coordinate damage, using a common scale for all eight panels.
fig,axs=plt.subplots(2,4,figsize=(12,7))
fig.subplots_adjust(left=.07,right=.97,bottom=.16,top=.80,wspace=.25,hspace=.42)
fig.text(.07,.95,'The failed planner moved every connector to its own origin.',fontsize=18,fontweight='bold')
fig.text(.07,.90,'Same body shapes; connector positions are lost before search. Reproduced from the recorded code.',fontsize=10.5,color='#58666d')
names=['p','N','b','Y']
cm=ListedColormap(['white','#cbd3d7','#b85b36','#466daf'])
for j,name in enumerate(names):
    for row,variants in enumerate([G['corrected_variants'],G['original_variants']]):
        body,ends=variants[name][0]
        grid=np.zeros((28,28),dtype=int)
        for r,c in body:grid[r+4,c+4]=1
        for end in ends:
            for r,c in end:grid[r+4,c+4]=2 if row==0 else 3
        ax=axs[row,j]
        ax.imshow(grid,cmap=cm,vmin=0,vmax=3,extent=[-4,24,24,-4],interpolation='nearest')
        ax.set_xticks([0,6,12,18]);ax.set_yticks([0,6,12,18])
        ax.tick_params(labelsize=8,length=0)
        for sp in ax.spines.values():sp.set_visible(False)
        ax.set_title(f'{name}: {len(ends)} connectors',fontsize=11)
        if j==0:ax.set_ylabel('True offsets' if row==0 else 'Recorded faulty offsets',fontsize=10)
fig.legend(handles=[Patch(color='#cbd3d7',label='Body'),Patch(color=ORANGE,label='True connector positions'),
                    Patch(color='#466daf',label='All connectors collapse to (0,0)')],
           loc='lower center',bbox_to_anchor=(.53,.09),ncol=3,frameon=False,fontsize=9)
fig.text(.07,.053,'Original search: 0 admissible joint relations. Common coordinate frame: 176 relations, but still no complete solution.',fontsize=9,color='#58666d')
fig.text(.07,.030,'A second defect remains: the search rejects valid joins connecting two pairs at once. A coordinate fix alone is insufficient.',fontsize=9,color='#58666d')
save(fig,'02-geometry-defect')

# Real rendered boards. The win board is captured before the engine advances to L3.
palette=['#ffffff','#cccccc','#999999','#555555','#333333','#000000','#ff66cc','#ffb6c1',
         '#ff2222','#2266ee','#77ccff','#ffde33','#ee8833','#992222','#88ee99','#9944cc']
fig,axs=plt.subplots(1,3,figsize=(12,5.4))
fig.subplots_adjust(left=.03,right=.98,bottom=.20,top=.77,wspace=.10)
boards=[R['b1-r1']['entry_board'],R['b1-r1']['actions'][24]['board'],E['runs']['b1-r2']['winning_boards'][-1]]
titles=['Identical L2 entry','r1 at 10.69 min','r2 winning assembly']
subtitles=['0/12 compatible endpoints','6 gray endpoints; 2/12 compatible','12/12 compatible endpoints']
for ax,board,title,subtitle in zip(axs,boards,titles,subtitles):
    ax.imshow(board,cmap=ListedColormap(palette),vmin=0,vmax=15,interpolation='nearest')
    ax.set_title(title,fontsize=12,fontweight='bold',pad=10)
    ax.set_xticks([]);ax.set_yticks([])
    ax.set_xlabel(subtitle,labelpad=10,fontsize=10)
    for sp in ax.spines.values():sp.set_visible(False)
fig.text(.04,.94,'What the game showed, and what its win check counted',fontsize=19,fontweight='bold')
fig.text(.04,.885,'Boards from saved events and verified engine replay. Right board captured before the L3 transition.',fontsize=10.5,color='#58666d')
fig.text(.04,.065,'Selected pieces render white; color is not a stable piece identity. Visible gray alone is an incomplete success signal.',fontsize=9,color='#58666d')
save(fig,'03-board-comparison')

metrics={}
for label,rec in R.items():
    fragments=[f for b in rec['blocks'] for f in b['fragments']]
    outs=[f for f in fragments if f['type'].startswith('TOOL RESULT:')]
    metrics[label]={'level_minutes':rec['level_minutes'],'actions':len(rec['actions']),
                    'transcript_chunks':len(rec['blocks']),
                    'thinking_characters':sum(len(f['text']) for f in fragments if f['type']=='THINKING'),
                    'python_tool_calls':sum(f['type'].startswith('TOOL CALL:') for f in fragments),
                    'error_results':sum('Traceback' in f['text'] for f in outs),
                    'timeout_results':sum('Tool timed out' in f['text'] for f in outs),
                    'reported_b1_request_time_trim_counter':rec['mechanism_counters']['trims']}
findings={'scope':'cn04 L2, two saved B1 repetitions; one-based levels; no general efficacy claim',
          'same_entry_board':R['b1-r1']['entry_board_sha256']==R['b1-r2']['entry_board_sha256'],
          'verified_action_frames':sum(r['verified_actions'] for r in E['runs'].values()),
          'successful_segment_replays_from_failed_prefix':E['counterfactual'],
          'metrics':metrics,'progress_series':series,'long_action_gaps':gaps,
          'evidence':[
              {'claim':'Independent connector normalization destroys body-relative offsets',
               'source':R['b1-r1']['source'],'line':4784,'check':'geometry-check.json'},
              {'claim':'Failed search output used to reject the all-ends goal',
               'source':R['b1-r1']['source'],'line':5096},
              {'claim':'Search excludes observed valid simultaneous two-connector join',
               'source':R['b1-r1']['source'],'line':5034,'check':'geometry-check.json'},
              {'claim':'r2 corrects a rotation/placement prediction from board feedback',
               'source':R['b1-r2']['source'],'line':5813},
              {'claim':'Visible gray union differs from same-type compatibility win check',
               'source':E['engine_source'],'lines':[955,1001],'check':'engine-replay.json'}],
          'limits':['One selected case, not an estimate of prevalence or treatment effect',
                    'B1 trim counters omit commit-time eviction; see analysis/failure-audit-20261013/request-context-audit.json',
                    'No-action intervals do not distinguish scheduling, reasoning and tool time',
                    'Coordinate correction alone still returns zero; repair is not gameplay-qualified',
                    'Engine internals are inspection evidence, unavailable to the model during play']}
(OUT/'findings.json').write_text(json.dumps(findings,indent=2)+'\n')
print(json.dumps({'metrics':metrics,'figures':3,'verified_action_frames':findings['verified_action_frames']},indent=2))
