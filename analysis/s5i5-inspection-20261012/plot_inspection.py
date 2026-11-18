"""Render trajectory and model-counterexample evidence for s5i5 L2."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Patch
import numpy as np

OUT=Path(__file__).resolve().parent
R=json.loads((OUT/'records.json').read_text())
E=json.loads((OUT/'engine-replay.json').read_text())
M=json.loads((OUT/'model-check.json').read_text())
P=json.loads((OUT/'plan-verification.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
colors={'b1-r1':'#b85b36','b1-r2':'#8066ac','control-pc':'#176f67','evidence-pt':'#466daf'}


def save(fig,stem):
    fig.savefig(OUT/(stem+'.png'),dpi=165,facecolor='white')
    fig.savefig(OUT/(stem+'.svg'),facecolor='white')
    plt.close(fig)


fig=plt.figure(figsize=(12,9))
fig.text(.06,.96,'Same level, four attempts, two different routes.',fontsize=20,fontweight='bold')
fig.text(.06,.919,'s5i5 L2 · all 237 recorded action frames verified · target at row 31, column 52',fontsize=11,color='#58666d')
ax=fig.add_axes([.08,.39,.39,.43])
walls=np.argwhere(np.array(R['b1-r1']['entry_board'])==15)
for row,col in walls:ax.add_patch(Rectangle((col-.5,row-.5),1,1,color='#d7cfe4',lw=0))
for label in R:
    points=[s['tip_row_col'] for s in E['runs'][label]['states']]
    ys,xs=zip(*points)
    ax.plot(xs,ys,color=colors[label],lw=1.7,alpha=.85,label=label)
    ax.scatter([xs[-1]],[ys[-1]],color=colors[label],marker='*' if R[label]['solved'] else 'x',s=70,zorder=5)
ax.scatter([52],[31],marker='D',color='#992222',s=60,zorder=6)
ax.text(53,33,'target',fontsize=9)
ax.text(44,5,'r2 goes above the wall',fontsize=8.5,color=colors['b1-r2'])
ax.annotate('direct descent blocked',xy=(52,10),xytext=(36,20),fontsize=8.5,
            arrowprops={'arrowstyle':'-','color':'#aab2b8'})
ax.set_xlim(7,62);ax.set_ylim(49,-1)
ax.set_xlabel('Tip column');ax.set_ylabel('Tip row')
ax.set_title('Executed tip trajectories',loc='left',fontweight='bold',pad=12)
ax.set_aspect('equal')

bx=fig.add_axes([.57,.39,.37,.43])
all_distances=[]
for label in R:
    rec=R[label];points=E['runs'][label]['states']
    times=[0]+[a['minutes_since_entry'] for a in rec['actions']]
    distance=[abs(s['tip_row_col'][0]-31)+abs(s['tip_row_col'][1]-52) for s in points]
    all_distances.extend(distance)
    if not rec['solved']:times.append(rec['level_minutes']);distance.append(distance[-1])
    bx.step(times,distance,where='post',color=colors[label],lw=1.7)
bx.set_xlim(0,102);bx.set_ylim(-2,max(all_distances)+3)
bx.set_xlabel('Minutes since L2 entry');bx.set_ylabel('Manhattan distance to target (pixels)')
bx.set_title('Progress against elapsed time',loc='left',fontweight='bold',pad=12)
bx.grid(color='#edf0f2')
fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.52,.895),ncol=4,frameon=False,fontsize=9)

cx=fig.add_axes([.14,.15,.8,.13])
for y,label in enumerate(reversed(list(R))):
    rec=R[label]; times=[a['minutes_since_entry'] for a in rec['actions']]
    cx.plot([0,rec['level_minutes']],[y,y],color='#cbd3d7',lw=1)
    cx.scatter(times,[y]*len(times),color=colors[label],s=17)
    cx.scatter([rec['level_minutes']],[y],color=colors[label],marker='*' if rec['solved'] else 'x',s=55)
cx.set_yticks(range(4),[f'{label} · {len(R[label]["actions"])} actions' for label in reversed(list(R))],fontsize=9)
cx.set_xlim(0,102);cx.set_ylim(-.5,3.5);cx.set_xlabel('Executed action times · minutes since L2 entry')
cx.tick_params(axis='y',length=0);cx.spines['left'].set_visible(False)
fig.text(.06,.067,'Distance is geometric, not a measure of remaining difficulty: valid routes must detour around the walls.',fontsize=9,color='#58666d')
fig.text(.06,.042,'B1 attempts differ from control/evidence overlays. These selected trajectories diagnose failures, not treatment effects.',fontsize=9,color='#58666d')
save(fig,'01-four-attempts')

row=M['available_when_failed_predictor_was_written']['row']
states=[row['input_state'],row['outcomes']['failed_push_model']['prediction'],row['actual_successor']]
titles=['Recorded predecessor','Failed simulator predicts','Recorded successor']
bodycolors={'O':'#ee8833','S':'#77ccff','Y':'#ffde33','N':'#88cc99'}
fig,axs=plt.subplots(1,3,figsize=(12,4.4))
fig.subplots_adjust(left=.06,right=.97,top=.72,bottom=.29,wspace=.17)
fig.text(.06,.935,'The simulator failed an observation it already had.',fontsize=18,fontweight='bold',va='top')
fig.text(.06,.855,'Extend sky at recorded L2 action 16 · predicted green stays put; observed green moves up with yellow',fontsize=10.5,color='#58666d',va='top')
for ax,state,title in zip(axs,states,titles):
    for key,s in state.items():
        h=3 if s['axis']=='H' else s['L'];w=s['L'] if s['axis']=='H' else 3
        ax.add_patch(Rectangle((s['c']-.5,s['r']-.5),w,h,facecolor=bodycolors[key],edgecolor='none'))
        ax.text(s['c']+w/2-.5,s['r']+h/2-.5,key,ha='center',va='center',fontsize=9)
    n=state['N'];dot=(n['r']+n['L']-2,n['c']+1)
    ax.scatter([dot[1]],[dot[0]],color='#992222',s=18,zorder=4)
    ax.set_xlim(8,44);ax.set_ylim(43,30);ax.set_aspect('equal')
    ax.set_xticks([10,20,30,40]);ax.set_yticks([31,34,37,40])
    ax.set_title(title,fontsize=11,fontweight='bold',pad=12)
    ax.set_xlabel('Column');ax.set_ylabel('Row')
    ax.text(9,32,'Green body row '+str(n['r']),fontsize=9)
fig.text(.06,.12,'Body bounds shown schematically; all panels use the same coordinates. This is observed movement, not a supplied game rule.',fontsize=9,color='#58666d')
fig.text(.06,.08,'At definition time: 18 predictor calls, 13 successor comparisons, 3 contradictions. A verified donor model finds a 31-click route.',fontsize=9,color='#58666d')
fig.text(.06,.046,'Local detection is demonstrated. Faster model repair and improved competition score under feedback remain untested.',fontsize=9,color='#58666d')
save(fig,'02-historical-counterexample')

# A concrete proposed test packet already exists, without future states.
metrics={}
for label,rec in R.items():
    fs=[f for b in rec['blocks'] for f in b['fragments']]
    outs=[f for f in fs if f['type'].startswith('TOOL RESULT:')]
    metrics[label]={'minutes':rec['level_minutes'],'actions':len(rec['actions']),
                    'python_calls':sum(f['type'].startswith('TOOL CALL:') for f in fs),
                    'error_results':sum('Traceback' in f['text'] for f in outs),
                    'reported_b1_request_time_trim_counter':rec['mechanism_counters'].get('trims'),
                    'thinking_characters':sum(len(f['text']) for f in fs if f['type']=='THINKING')}
findings={'game':'s5i5 L2, one-based','same_entry_board':len({r['entry_board_sha256'] for r in R.values()})==1,
          'verified_frames':sum(r['verified_actions'] for r in E['runs'].values()),'metrics':metrics,
          'available_historical_check':{k:v for k,v in M['available_when_failed_predictor_was_written'].items() if k!='row'},
          'offline_recovery_routes':P,
          'source_evidence':[
              {'run':'b1-r1','line':8225,'claim':'Failed simulator BFS prints 538647 states and no path'},
              {'run':'b1-r1','line':8373,'claim':'Model then says its goal model must be wrong'},
              {'run':'control-pc','line':5631,'claim':'Successful run recognizes buggy BFS and repairs it'},
              {'run':'b1-r2','line':7583,'claim':'Hand-planned direct descent hits horizontal wall'},
              {'source':str(Path(E['engine_source'])),'line':1958,'claim':'HUD width is scaled fraction of remaining step budget, not action count'}],
          'feedback_packet':'feedback-packet.json','model_feedback_sent':False,
          'interpretation':'Model validation recurs as a failure mechanism, but not all failed attempts share the same defect. Incremental feedback benefit needs a matched continuation test.',
          'limits':['Selected cases do not estimate prevalence','Successful overlays differ from B1',
                    'B1 trim counters omit commit-time eviction; see analysis/failure-audit-20261013/request-context-audit.json',
                    'Donor-model offline route is not proof the failed model can repair itself',
                    'Action gaps include scheduling, reasoning and tool time','Historical agreement does not establish unseen-transition correctness']}
(OUT/'findings.json').write_text(json.dumps(findings,indent=2)+'\n')
print(json.dumps({'metrics':metrics,'verified_frames':findings['verified_frames'],'figures':2},indent=2))
