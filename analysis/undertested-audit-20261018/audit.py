"""Reconcile saved evidence and render the retrospective; no network or inference."""
import hashlib
import html
import json
import math
from pathlib import Path
import statistics as st

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read(path):
    return json.loads((ROOT / path).read_text())


def bench(path):
    games = read(path)['game_runs']
    assert len(games) == 25
    assert len({g['game_id'] for g in games}) == 25
    assert all(0 <= g['levels_completed'] <= g['number_of_levels'] for g in games)
    return dict(levels=sum(g['levels_completed'] for g in games),
                score=st.mean(g['final_score'] for g in games),
                games=sorted(g['game_id'] for g in games))


def main():
    ideas = read(str(HERE.relative_to(ROOT) / 'ideas.json'))['ideas']
    inventory = read(str(HERE.relative_to(ROOT) / 'thread-inventory.json'))
    assert len(inventory['threads']) == 76
    assert not any(t['hasMore'] for t in inventory['threads'])
    assert sum(t['itemsRead'] for t in inventory['threads']) == 2840
    assert len({i['id'] for i in ideas}) == len(ideas)
    source_paths = {p for i in ideas for p in i['sources']}
    full = {
        'b1-r1': 'solvers/m2_b1/experiment/pair-1/b1-r1/output/benchmark.json',
        'b1-r2': 'solvers/m2_b1/experiment/pair-1/b1-r2/output/benchmark.json',
        'evidence-control': 'solvers/m2_evidence/experiment/pair-1/control/output/benchmark.json',
        'evidence-treatment': 'solvers/m2_evidence/experiment/pair-1/treatment/output/benchmark.json',
        'c14': 'analysis/model-serving-20261011/wave-1/e2-c14/output/benchmark.json',
        'b1-c14': 'analysis/model-serving-20261011/wave-2/b1-c14/output/benchmark.json',
        'prune-control': 'analysis/pruning-transfer-20261011/control/output/benchmark.json',
        'prune-candidate2': 'analysis/pruning-transfer-20261011/candidate2/output/benchmark.json',
    }
    source_paths.update(full.values())
    full_results = {k: bench(v) for k, v in full.items()}
    game_sets = {tuple(v['games']) for v in full_results.values()}
    assert len(game_sets) == 1
    assert full_results['b1-r1']['levels'] == 102
    assert full_results['b1-r2']['levels'] == 117
    prune = read('analysis/pruning-transfer-20261011/RESULT.json')
    assert full_results['prune-control']['levels'] == prune['control']['levels'] == 103
    assert full_results['prune-candidate2']['levels'] == prune['candidate']['levels'] == 106
    for arm, key in [('control', 'prune-control'), ('candidate', 'prune-candidate2')]:
        assert math.isclose(full_results[key]['score'], prune[arm]['score'])
    snapshot = read('analysis/latest-runs-20261018/data.json')
    hidden = [d for d in snapshot['hidden'] if d['status'] == 'COMPLETE']
    controls = [d['score'] for d in hidden if d['arm'] == 'Control']
    assert controls == [23.87, 29.44, 24.97, 29.38, 31.31]
    mean, sd = st.mean(controls), st.stdev(controls)
    assert sum(s >= 30 for s in controls) == 1
    for d in hidden:
        d['delta_vs_control_mean'] = d['score'] - mean
    # Optimistic normal approximation: known variance, equal independent draws,
    # two-sided alpha=.05 and 80% power. Not a recommendation or formal power guarantee.
    power = {str(effect): math.ceil(2 * sd**2 * (1.96 + .8416212336)**2 / effect**2)
             for effect in [3, 6]}
    evidence = dict(
        scope=dict(prior_threads=76, messages_read=2840, idea_families=len(ideas),
                   historical_source_limit='Available T3 project threads plus archived ledgers; not every historical model token or missing archive.'),
        score_snapshot_utc=snapshot['as_of_utc'],
        hidden_controls=dict(scores=controls, n=len(controls), mean=mean, sample_sd=sd,
                             at_least_30=sum(s >= 30 for s in controls)),
        hidden_completed_draws=hidden,
        full_public_runs={k: {a: b for a, b in v.items() if a != 'games'} for k, v in full_results.items()},
        pruning=dict(level_delta=106-103, score_delta=prune['candidate']['score']-prune['control']['score'],
                     decode_relative=prune['candidate']['throughput']/prune['control']['throughput']-1),
        illustrative_power=dict(draws_per_arm=power, assumptions='Normal known-sigma approximation, 80% power, two-sided 5%; sigma estimated from only five controls; optimistic and regime-specific.'),
        corrections=[
            'Only one of five unchanged controls reaches 30, not two as the recent narrative stated.',
            'The eight full runs in the new ANOVA include different treatments; its residual is not a pure identical-candidate noise estimate.',
            'Failure to detect a run effect does not establish absent treatment effects or prove the high B1 replicate is selection noise.',
            'Public-25 levels, public-25 RHAE, and leaderboard scores on hidden games have different noise regimes.',
            'The old 7254-vs-6000-second controls change deadline-dependent scheduling and cannot be repaired by cropping.',
            'The screen-2 REWRITE_SLOWS label was superseded by its card addendum; use raw counts and the later interpretation.',
            'No-harm/retention gates are deployment decisions; neither passing nor failing one establishes the sign of a small effect.',
        ],
        actions_taken='CPU/source audit only. No model calls, GPU launch, training, submission, queue/solver edits or promotion.',
    )
    source_paths.update(['analysis/draw-variance-20261018/decompose.py',
                         'analysis/draw-variance-20261018/FINDINGS.md'])
    missing = [p for p in sorted(source_paths) if not (ROOT / p).is_file()]
    assert not missing, missing
    evidence['source_manifest'] = {p: dict(bytes=(ROOT/p).stat().st_size,
                                          sha256=hashlib.sha256((ROOT/p).read_bytes()).hexdigest())
                                   for p in sorted(source_paths)}
    (HERE/'RESULT.json').write_text(json.dumps(evidence, indent=2)+'\n')
    render(ideas, evidence, inventory)
    print(json.dumps(dict(verified_sources=len(source_paths), threads=76, messages=2840,
                          idea_families=len(ideas), hidden_control_mean=mean, hidden_control_sd=sd,
                          controls_at_least_30=1, pruning=evidence['pruning'],
                          optimistic_draws_per_arm=power, output=str(HERE/'audit.html'))))


def render(ideas, evidence, inventory):
    esc = html.escape
    mean = evidence['hidden_controls']['mean']
    def sx(v): return 168 + (v-20)/14 * 470
    # Four categorical rows; every score shares one horizontal coordinate scale.
    svg = ['<svg viewBox="0 0 728 226" role="img" aria-label="Five control draws span 23.87 to 31.31. Each candidate has one draw. The lines are a control mean reference, not confidence intervals.">']
    for tick in [20, 24, 28, 32, 34]:
        x=sx(tick)
        svg.append(f'<line x1="{x}" x2="{x}" y1="20" y2="190" class="grid"/><text x="{x}" y="211" text-anchor="middle" class="tick">{tick}</text>')
    svg.append(f'<line x1="{sx(mean)}" x2="{sx(mean)}" y1="18" y2="190" class="ref"/>')
    rows=[('Unchanged control','Control',42),('Evidence record','Evidence',86),('B1 + c14','B1 + c14',130),('B1 alone','B1',174)]
    for label,arm,y in rows:
        ds=[d for d in evidence['hidden_completed_draws'] if d['arm']==arm]
        svg.append(f'<text x="1" y="{y+5}" class="series">{esc(label)} <tspan class="small">n={len(ds)}</tspan></text>')
        for j,d in enumerate(ds):
            dy = (-7 if j==1 else 7 if j==3 else 0) if arm=='Control' else 0
            x=sx(d['score'])
            svg.append(f'<circle cx="{x}" cy="{y+dy}" r="4" class="dot"/><text x="{x}" y="{y+dy-11}" text-anchor="middle" class="value">{d["score"]:.2f}</text>')
    svg.append('</svg>')
    top=[]
    for i in ideas[:6]:
        top.append(f'<tr><td><b>{i["rank"]}. {esc(i["name"])}</b><span class="tag">{esc(i["group"])}</span></td><td>{esc(i["evidence"])}</td><td>{esc(i["gap"])}<details><summary>Next discriminator + limits</summary><p>{esc(i["next"])}</p><p class="muted">{esc(i["limit"])}</p>{source_links(i)}</details></td></tr>')
    ledger=[]
    for i in ideas:
        search=esc(' '.join([i['name'],i['group'],i['evidence'],i['gap'],i['next']]),quote=True)
        ledger.append(f'<details class="idea" data-group="{esc(i["group"],quote=True)}" data-search="{search}"><summary><span class="name">{esc(i["name"])}</span><span class="tag">{esc(i["group"])}</span></summary><p><b>Observed:</b> {esc(i["evidence"])}</p><p><b>What remains open:</b> {esc(i["gap"])}</p><p><b>Next:</b> {esc(i["next"])}</p><p class="muted">{esc(i["limit"])}</p>{source_links(i)}</details>')
    groups=sorted({i['group'] for i in ideas})
    options='<option value="">All decisions</option>'+''.join(f'<option>{esc(g)}</option>' for g in groups)
    threads=''.join(f'<li><a href="t3-thread://v1/{esc(t["threadId"],quote=True)}">{esc(t["title"])}</a> <span class="muted">({t["itemsRead"]} messages)</span></li>' for t in inventory['threads'])
    corrections=''.join(f'<li>{esc(c)}</li>' for c in evidence['corrections'])
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Undertested experiment audit</title><style>
:root{font-family:system-ui,sans-serif}*{box-sizing:border-box}html,body{margin:0;background:transparent;color:var(--foreground,#242932);font-family:var(--font-sans,system-ui,sans-serif);font-size:14px;line-height:1.5}body{padding:0}h2{font-size:20px;margin:24px 0 10px;line-height:1.3}p{margin:9px 0 13px}.lead{font-size:19px;line-height:1.4;font-weight:650;margin-top:0}.muted,.small{color:var(--muted-foreground,#626c7b)}.meta{font-size:12px}.note{background:var(--card,#f4f5f7);padding:16px;border-radius:var(--radius,8px);margin:16px 0}.note p:last-child{margin-bottom:0}svg{display:block;width:100%;height:226px;overflow:visible}svg text{font-family:var(--font-sans,system-ui,sans-serif);fill:var(--foreground,#242932)}.grid{stroke:var(--border,#d8dce2);stroke-width:1}.ref{stroke:var(--muted-foreground,#626c7b);stroke-dasharray:4 4}.dot{fill:var(--accent,#346f85)}.tick,.value{font-size:12px}.series{font-size:14px}.small{fill:var(--muted-foreground,#626c7b);font-size:11px}table{border-collapse:collapse;width:100%;table-layout:fixed;font-size:13px}th,td{text-align:left;vertical-align:top;border-bottom:1px solid var(--border,#d8dce2);padding:12px 10px 12px 0}th{font-size:12px;color:var(--muted-foreground,#626c7b)}th:first-child{width:23%}th:nth-child(2){width:35%}th:nth-child(3){width:42%}.tag{display:block;font-size:11px;color:var(--muted-foreground,#626c7b);font-weight:500;margin-top:4px}.idea{padding:12px 0;border-bottom:1px solid var(--border,#d8dce2)}summary{cursor:pointer;font-weight:600}td details{margin-top:9px}td summary{font-size:12px;color:var(--accent,#346f85)}details p{font-weight:400}.sources{font-size:11px;overflow-wrap:anywhere}.sources a,a{color:var(--accent,#346f85)}input,select{font:inherit;color:var(--foreground,#242932);background:transparent;border:1px solid var(--border,#d8dce2);border-radius:6px;padding:9px;min-width:0}.controls{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:15px 0 8px}.name{font-size:14px}.idea[hidden]{display:none}ul,ol{padding-left:22px}li{margin:7px 0}.coverage{font-size:12px}.mobile-scores{display:none}.comparison-note{font-size:12px;margin-top:0}button{font:inherit;cursor:pointer}code{font-family:var(--font-mono,monospace);font-size:12px} @media(max-width:520px){svg{display:none}.mobile-scores{display:block;font-size:14px;padding:8px 0}.mobile-scores div{margin:8px 0}.mobile-scores b{display:block}.lead{font-size:17px}table,tbody,tr,td{display:block}thead{display:none}td{padding:8px 0;border:0}tr{padding:12px 0;border-bottom:1px solid var(--border,#d8dce2)}td:first-child{font-size:15px}.controls{grid-template-columns:1fr}svg{height:190px}.note{padding:16px}h2{font-size:18px}}
</style><body>
<p class="lead">We have underpowered candidates and untested mechanisms. Neither means the pipeline is exhausted.</p>
<p class="meta muted">18 October 2026 · 76 prior project threads / 2,840 messages · 39 idea families · Primary result and receipt checks, plus older research ledgers. Score snapshot: 06:41 UTC; not a live submission-status page.</p>
<div class="note"><b>My allocation of attention</b><p><b>Best existing artifact to revisit:</b> REAP448 + c18. <b>Best unresolved stack:</b> B1 + c14. <b>Best new mechanism test:</b> automatic temporal evidence that leads to an executable rule correction.</p><p class="muted">No candidate here is a demonstrated score win on the current hidden set. Keep the incumbent until the specified comparison earns a change. The audit performed no inference, GPU launches, submissions, queue changes, training or solver edits.</p></div>
<h2>What the hidden draws actually resolve</h2>
''' + ''.join(svg) + f'''<div class="mobile-scores"><div><b>Unchanged control · n=5</b>23.87 · 24.97 · 29.38 · 29.44 · 31.31</div><div><b>Evidence record · n=1</b>26.24</div><div><b>B1 + c14 · n=1</b>23.56</div><div><b>B1 alone · n=1</b>26.96</div></div><p class="comparison-note muted">Points are individual public-leaderboard scores on hidden games; these are not public-25 scores. Dashed line: unchanged-control mean {mean:.2f}. Control SD {evidence['hidden_controls']['sample_sd']:.2f}, n=5; a descriptive estimate, not a confidence band. Different candidate packages must not be pooled as one treatment.</p>
<p>A single D3 score of 23.56 is <b>4.23 points below the control mean</b>. That is compatible with an inferior stack and with ordinary variation. One B1 draw of 26.96 likewise does not settle its effect. The identical public B1 replicas differ by 15 levels and 9.47 score points.</p>
<details><summary>Noise audit: corrections that affect the decision</summary><ul>''' + corrections + '''</ul><p>Using SD 3.20 as if known, a normal approximation needs about 18 independent draws per arm to detect+3 points with80% power, or5 per arm for+6. These are optimistic examples, not promises: SD itself comes from only 5 controls. One more repeat can triage an idea; it cannot certify a small gain.</p><p>The public/private correlation projection also depends on hidden per-game variability that we cannot observe. No run-effect detection in 8 mixed arms does not establish no shared run luck.</p></details>
<h2>Priorities, with the test each idea still needs</h2><p class="muted meta">Ranks 1–2 need score replication. Ranks 3–6 first need direct evidence that the intended mechanism changes a useful decision. The order is judgment about information value, not a calibrated success probability.</p>
<table><thead><tr><th>Idea</th><th>Observed evidence</th><th>Why it remains open</th></tr></thead><tbody>''' + ''.join(top) + '''</tbody></table>
<h2>What I would do next</h2><ol><li><b>Repair and qualify REAP competition packaging, then use one exploratory hidden draw.</b> A subsequent source audit found a public-only guard and a short-budget overwrite in the completed candidate. The gameplay package remains promising, but its exact notebook cannot be submitted unchanged. Preserve the repaired version identity; a lucky score is not a promotion rule.</li><li><b>Run one small temporal-evidence comparison before another broad prompt or helper screen.</b> Use several games and multiple prefixes, count actual correction and new predictive accuracy, then test score. Do not count animation frames or seeds as independent games.</li><li><b>Spend replication on fixed packages.</b> Compare native+c14 vs REAP+c18, or B1 vs B1+c14, with fresh controls and unchanged deadlines. Early high-concurrency lag is expected; keep the full-horizon readout.</li></ol>
<p class="muted">Do not immediately combine all six ideas. There is no evidence that their gains add, and more throughput can produce longer unproductive reasoning. Existing F1/F1x draws are already evaluation work; duplicate screens would answer less.</p>
<h2>Full disposition ledger</h2><details><summary>Inspect all 39 idea families</summary><p>Whole-package negatives remain negative. Their untested components stay visible without earning automatic reruns. Search to inspect any family.</p>
<div class="controls"><input id="search" type="search" placeholder="Search ideas, mechanisms, evidence…" aria-label="Search idea ledger"><select id="group" aria-label="Filter by disposition">''' + options + '''</select></div><p id="count" class="meta muted">39 of 39 families</p><section id="ledger">''' + ''.join(ledger) + '''</section></details>
<details class="coverage"><summary>Coverage and reproducibility</summary><p>Read every timeline page of all 76 prior top-level threads in this project's current T3 catalog, without truncated message text, plus the board, experiment cards/results, and older Questions and Ideas/hypothesis/archive ledgers. This is coverage of accessible families, not a claim to have recovered missing runs or every token in pre-T3 archives. Notebook mining and old family reviews provide additional historical coverage.</p><p><code>python3 -B analysis/undertested-audit-20261018/audit.py</code> reconciles 8 full public benchmarks, 5 unchanged hidden draws, candidate scores and the source manifest. It writes RESULT.json and this HTML only. ideas.json retains all judgments and next tests; thread-inventory.json records the navigation inventory and excerpts.</p><ul>''' + threads + '''</ul></details>
<script>const search=document.getElementById('search'),group=document.getElementById('group'),rows=[...document.querySelectorAll('.idea')];function filter(){let n=0;for(const row of rows){const ok=(!group.value||row.dataset.group===group.value)&&row.dataset.search.toLowerCase().includes(search.value.toLowerCase());row.hidden=!ok;if(ok)n++;}document.getElementById('count').textContent=n+' of '+rows.length+' families';return n;}search.addEventListener('input',filter);group.addEventListener('change',filter);filter();const checks=[];function check(label,expected){const actual=rows.filter(r=>!r.hidden).length;if(actual!==expected)throw new Error(label+': '+actual+' != '+expected);checks.push(label+'='+actual);}check('initial',39);group.value='Retest score';group.dispatchEvent(new Event('change'));check('score retests',2);search.value='REAP448 + 18 streams';search.dispatchEvent(new Event('input'));check('exact search',1);group.value='Test mechanism';group.dispatchEvent(new Event('change'));check('combined excludes',0);search.value='';group.value='';filter();check('reset',39);console.log('Interactive filter checks PASS',checks.join(', '));</script></body></html>'''
    (HERE/'audit.html').write_text(page)


def source_links(idea):
    return '<div class="sources">Sources: '+ '<br>'.join(
        f'<a href="{html.escape(str(ROOT/p), quote=True)}">{html.escape(p)}</a>'
        for p in idea['sources'])+'</div>'


if __name__ == '__main__':
    main()
