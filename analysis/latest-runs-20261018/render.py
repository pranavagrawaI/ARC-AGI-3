"""Build a standalone comparison sheet from the collected snapshot."""
import datetime as dt, html, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = json.loads((HERE/'data.json').read_text())
esc = html.escape
stamp = dt.datetime.fromisoformat(D['as_of_utc'])
local = stamp.astimezone(dt.timezone(dt.timedelta(hours=5, minutes=30)))
def txt(x,y,s,**kw):
    attrs=' '.join(f'{k.replace("_","-")}="{v}"' for k,v in kw.items())
    return f'<text x="{x}" y="{y}" {attrs}>{esc(str(s))}</text>'

# A shared score axis preserves the control spread without suggesting a time trend.
W,H=700,322
x=lambda v: 145+v/40*440
plot=[f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Hidden-draw leaderboard scores. Control spans 23.87 to 31.31; F1 is pending.">']
lo,hi=D['control_summary']['minimum'],D['control_summary']['maximum']
plot.append(f'<rect x="{x(lo)}" y="12" width="{x(hi)-x(lo)}" height="278" fill="var(--muted,#eef0f2)" opacity=".8"/>')
for tick in range(0,41,10):
    plot.append(f'<line x1="{x(tick)}" x2="{x(tick)}" y1="12" y2="290" class="grid"/>')
    plot.append(txt(x(tick),309,tick,text_anchor='middle',class_='axis'))
rows=list(D['hidden'])
# Include missing D6 visibly; no zero-valued measurement is invented.
rows.insert(next(i for i,r in enumerate(rows) if r['draw']=='D7'),dict(draw='D6',arm='Unused',score=None,status='NO SUBMISSION',date='2026-10-15'))
for i,r in enumerate(rows):
    y=28+i*27
    label=('Oct 1 base' if r['draw']=='Base' else r['draw'])+' · '+r['arm']
    plot.append(txt(0,y+4,label,font_size='12'))
    color='var(--muted-foreground,#5c626c)' if r['arm']=='Control' else 'var(--chart-1,#2164ac)'
    if r['score'] is not None:
        plot.append(f'<circle cx="{x(r["score"])}" cy="{y}" r="4.5" fill="{color}"/>')
        plot.append(txt(620,y+4,f'{r["score"]:.2f}',font_size='12',font_weight='600'))
    else:
        plot.append(txt(155,y+4,'Pending score' if r['draw']=='D9' else 'Slot unused',font_size='12',fill='var(--muted-foreground,#5c626c)'))
plot.append('</svg>')
chart='<div class="hidden-chart">'+''.join(plot).replace('class-="','class="')+'</div>'
chart+='<table class="hidden-mobile"><thead><tr><th>Draw / arm</th><th class="num">Score</th></tr></thead><tbody>'+''.join('<tr><td>'+esc(('Oct 1 base' if r['draw']=='Base' else r['draw'])+' · '+r['arm'])+'</td><td class="num">'+(f"{r['score']:.2f}" if r['score'] is not None else ('Pending' if r['draw']=='D9' else 'Unused'))+'</td></tr>' for r in rows)+'</tbody></table>'

summaries=[]
for label in ['F1','F1x','Lexicon']:
    s=next(s for s in D['screens'] if s['label']==label)
    t,c=s['arms'][s['treatment']],s['arms'][s['control']]
    summaries.append(f'<tr><td><strong>{label}</strong><small>{"Screen 2" if label=="F1x" else "Screen 1"} · Oct 17</small></td><td class="num">{t["levels"]} / {c["levels"]}</td><td class="num">{s["delta"]:+d}<small>{s["relative_delta"]:+.1%}</small></td><td class="num">{s["games_ahead"]} / {s["games_tied"]} / {s["games_behind"]}</td><td>{"Dropped" if label=="Lexicon" else "No-harm pass"}</td></tr>')

f1,f1x=[next(s for s in D['screens'] if s['label']==v) for v in ['F1','F1x']]
mechanism=[]
for s in [f1,f1x]:
    t,c=[s['arms'][s[k]]['metrics'] for k in ['treatment','control']]
    mechanism.append(f'<tr><td>{s["label"]}</td><td class="num">{t["preload_name_errors"]} / {c["preload_name_errors"]}</td><td class="num">{t["length_cutoffs"]} / {c["length_cutoffs"]}</td><td class="num">{t["tool_calls_between_caps"]}</td></tr>')

pergame=[]
ordered=[f1,f1x,next(s for s in D['screens'] if s['label']=='Lexicon')]
for gid in sorted(f1['per_game_delta']):
    cells=''
    for s in ordered:
        t,c=[s['arms'][s[k]]['per_game'][gid]['levels'] for k in ['treatment','control']]
        delta=t-c
        cells+=f'<td class="num">{t} / {c} <span class="delta">({delta:+d})</span></td>'
    pergame.append(f'<tr><td>{gid.split("-")[0].upper()}</td>{cells}</tr>')
hidden_rows=[]
for r in reversed(D['hidden']):
    hidden_rows.append(f'<tr><td>{r["draw"]} · {esc(r["arm"])}</td><td>{r["date"][:16]} UTC</td><td class="num">{r["score"] if r["score"] is not None else "—"}</td><td>{r["status"]}</td><td>{r["ref"]}</td></tr>')
sources=''.join(f'<li><code>{esc(path)}</code><small>SHA256 {meta["sha256"]}</small></li>' for path,meta in D['source_manifest'].items())

page='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Latest ARC-AGI-3 runs · 18 October 2026</title><style>

.hidden-mobile{display:none}*{box-sizing:border-box}body{margin:0;color:var(--foreground,#20262d);font:14px/1.5 var(--font-sans,system-ui,sans-serif);background:transparent}main{width:100%;padding:0}.eyebrow{font-size:11px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted-foreground,#5c626c)}.lead{font-size:23px;line-height:1.25;font-weight:650;margin:8px 0 10px;max-width:660px}p{margin:8px 0}small,.note{font-size:12px;color:var(--muted-foreground,#5c626c)}small{display:block}.status{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:18px 0 24px}.panel{background:var(--card,#f7f8fa);border-radius:var(--radius,10px);padding:16px}.big{font-size:30px;font-weight:650;line-height:1.2;margin:5px 0}h2{font-size:16px;font-weight:650;margin:26px 0 5px}h3{font-size:14px;font-weight:600;margin:10px 0 5px}svg{display:block;width:100%;height:322px;overflow:visible;margin-top:6px}svg text{font-family:var(--font-sans,system-ui,sans-serif);fill:var(--foreground,#20262d)}.grid{stroke:var(--border,#d9dde3);stroke-width:1}.axis{fill:var(--muted-foreground,#5c626c)}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:12px;margin:10px 0}th{font-weight:500;color:var(--muted-foreground,#5c626c);text-align:left;font-size:11px}td,th{padding:9px 6px;border-bottom:1px solid var(--border,#d9dde3);vertical-align:top}td:first-child,th:first-child{padding-left:0}td:last-child,th:last-child{padding-right:0}.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}td small{font-size:10px}.delta{color:var(--muted-foreground,#5c626c)}.callout{border-left:3px solid var(--chart-1,#2164ac);padding:2px 0 2px 12px;margin:14px 0;font-size:13px}details{margin:18px 0;border-top:1px solid var(--border,#d9dde3);padding-top:12px}summary{cursor:pointer;font-size:13px;font-weight:600}code{font:11px var(--font-mono,monospace);overflow-wrap:anywhere}li{margin:8px 0}li small{font-size:10px;word-break:break-all}.footer{margin:20px 0 8px;font-size:11px;color:var(--muted-foreground,#5c626c)}@media(max-width:480px){.hidden-chart,.chart-legend{display:none}.hidden-mobile{display:table}.lead{font-size:20px}.status{gap:8px}.panel{padding:16px}.big{font-size:25px}td,th{padding:8px 4px}table{font-size:11px}svg{height:285px}h2{margin-top:20px}}
</style></head><body><main>
'''
page+=f'<div class="eyebrow">ARC-AGI-3 · Snapshot {local:%d %b %Y, %H:%M} IST / {stamp:%H:%M} UTC</div>'
page+='<p class="lead">F1 is now on the hidden set.<br>Its score is still pending.</p>'
page+='<div class="status"><div class="panel"><div class="eyebrow">D9 · F1 v1</div><div class="big">Pending</div><small>Submitted Oct 18, 12:00 IST<br>Reference 57342990</small></div><div class="panel"><div class="eyebrow">D8 · unchanged control</div><div class="big">31.31</div><small>Complete · latest scored draw<br>Reference 57307656</small></div></div>'
page+='<h2>Hidden draws: control variation remains large</h2><p class="note">Kaggle public leaderboard score on hidden games. <span class="chart-legend">Gray dots = unchanged control; blue = candidate. Shaded band = observed control range, not a confidence interval.</span></p>'+chart
c=D['control_summary']
page+=f'<p class="note">Five control draws: mean <strong>{c["mean"]:.2f}</strong>, range <strong>{c["minimum"]:.2f}–{c["maximum"]:.2f}</strong>. These are independent draws; a candidate inside or above this range alone does not establish an effect. Private scores and hidden per-game results are unavailable.</p>'
page+='<h2>Latest public screens: friction passes; lexicon loses</h2><p class="note">All six kernels COMPLETE. Same 25 game IDs, one pass per arm, nominal 60-minute play window. Each row compares its treatment with its own concurrent control.</p><div class="scroll"><table><thead><tr><th>Candidate</th><th class="num">Levels<br>treat / control</th><th class="num">Difference</th><th class="num">Games<br>ahead / tied / behind</th><th>Screen rule</th></tr></thead><tbody>'+''.join(summaries)+'</tbody></table></div>'
page+='<p class="note">F1 and F1x pass the permitted 7-level deficit rule. This establishes screen eligibility; neither has a demonstrated gameplay improvement. “Cancelled” games are expected at the screen deadline.</p>'
page+='<div class="callout"><strong>The frame rewrite changes board reading; its score effect is unresolved.</strong> F1’s −3 and F1x’s +4 are from separate pairs. The screen-2 <code>REWRITE_SLOWS</code> label was explicitly voided in the card addendum after control variation undermined the speed claim.</div>'
page+='<h3>What the friction bundle actually changed</h3><div class="scroll"><table><thead><tr><th>Variant</th><th class="num">Preload NameErrors<br>treat / control</th><th class="num">Length cutoffs<br>16K treat / 12K control</th><th class="num">Tool-call replies<br>above old 12K cap</th></tr></thead><tbody>'+''.join(mechanism)+'</tbody></table></div><p class="note">Counts from the saved request-log analysis. Fewer cutoffs at a higher cap are an operational observation; they do not measure recovered solves.</p>'
page+='<h2>Next recorded comparison</h2><p><strong>D10 · F1x v1 is queued for Oct 19 UTC.</strong> The fixed read rule selects F1 only if D9 beats D10 by at least 6 score points; otherwise F1x. An invalid draw yields no comparison. This selects a variant; benefit over control remains a separate question.</p>'
page+='<details><summary>Inspect the 25 matched games</summary><p class="note">Each cell is treatment / its control (level difference). Game IDs align across all six runs.</p><div class="scroll"><table><thead><tr><th>Game</th><th class="num">F1</th><th class="num">F1x</th><th class="num">Lexicon</th></tr></thead><tbody>'+''.join(pergame)+'</tbody></table></div></details>'
page+='<details><summary>Hidden-draw records and submission IDs</summary><div class="scroll"><table><thead><tr><th>Draw / arm</th><th>Submitted</th><th class="num">Score</th><th>Status</th><th>Reference</th></tr></thead><tbody>'+''.join(hidden_rows)+'</tbody></table></div><p class="note">D6 (Oct 15) had no submission; it is missing data, not a zero.</p></details>'
page+='<details><summary>Sources and verification</summary><p class="note">Live Kaggle submissions, latest-kernel listing and six kernel status checks are saved in <code>live/</code>. Recomputed all six level totals, matched 25 unique IDs per run, checked level bounds and terminal game states, and reconciled all 75 paired differences to totals. Mechanism counters are taken from saved RESULT files. No runs were launched by this collection.</p><ul>'+sources+'</ul></details>'
page+='<p class="footer">Scope: the six latest screening kernels (Oct 17), hidden draw series through D9, and the next queued draw. Snapshot files: data.json · per-game.csv · live/*.json. Rebuild with gather.py, then render.py.</p></main></body></html>'
(HERE/'latest-runs.html').write_text(page)
print(json.dumps(dict(html=str(HERE/'latest-runs.html'),bytes=len(page))))
