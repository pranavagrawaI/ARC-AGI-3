"""Mechanical worst-case timeline from the frozen constants and the code's own formulas."""
import importlib.util,json
from pathlib import Path
P=Path(__file__).resolve().parents[3]/'deployment/kaggle/submission-20260917-actor-critic-competition-v1'
def load(n,p):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
b=load('budget',P/'budget.py');outer=load('outer',P/'runtime/exact_request_outer_runtime.py')
started=0.0
deadline=started+b.PROVIDER-b.UNCERTAINTY-b.FINAL_OUTPUT-b.OUTER_CLEANUP          # entry.py ARC_NOTEBOOK_WORK_DEADLINE
rows=[]
for bootstrap in (60,900,b.BOOTSTRAP):
    t=started+bootstrap                                                            # after install.py (entry: timeout=BOOTSTRAP)
    remaining=min(b.GAMEPLAY,int(deadline-t-b.STARTUP_WITH_WARMUP-30))              # entry.py
    b.admission(remaining,b.GAMES)                                                  # refuse-before-startup check
    for startup in (300,b.STARTUP_WITH_WARMUP):
        t2=t+startup                                                                # outer: server+warmup bounded by --startup-timeout-s
        launcher=outer.bounded_launcher_timeout(remaining,deadline,t2)              # outer: min(remaining, deadline-now-30)
        play_timeout=launcher-60                                                    # outer passes --timeout-s launcher-60; subprocess timeout=launcher
        adm=b.admission(play_timeout,b.GAMES,b.REQUEST)                              # play.py first-game admission
        per_game=adm+2*b.REQUEST+b.TOOL_MARGIN+b.GAME_CLEANUP
        games_end=t2+b.GAMES*per_game+180                                           # == t2+play_timeout by construction
        outer_subprocess_kill=deadline+b.OUTER_CLEANUP                              # entry.py subprocess timeout
        rows.append(dict(bootstrap_s=bootstrap,startup_s=startup,entry_remaining=remaining,launcher_effective=launcher,play_timeout=play_timeout,first_game_admission_s=round(adm,2),per_game_bound_s=round(per_game,2),games_end_s=round(games_end,1),work_deadline_alarm_s=deadline,entry_kill_s=outer_subprocess_kill,provider_cap_s=b.PROVIDER,slack_to_alarm_s=round(deadline-games_end,1),slack_to_provider_s=round(b.PROVIDER-(outer_subprocess_kill+b.FINAL_OUTPUT),1)))
print(json.dumps(dict(constants={k:getattr(b,k) for k in ('PROVIDER','UNCERTAINTY','FINAL_OUTPUT','BOOTSTRAP','STARTUP_WITH_WARMUP','OUTER_CLEANUP','GAMEPLAY','GAMES','REQUEST','CRITIC_RESERVE','GAME_CLEANUP','TOOL_MARGIN','ACTOR_CALLS','CRITIC_CALLS')},validate=b.validate(),rows=rows,v15_remote_measured=dict(bootstrap_plus_startup_s='~360 (launcher-budget at_monotonic 511.9 vs cell start ~152)',outer_elapsed_minus_launcher_s=round(940.147-602.611,1),orderly_shutdown_s=12.139)),indent=1))
