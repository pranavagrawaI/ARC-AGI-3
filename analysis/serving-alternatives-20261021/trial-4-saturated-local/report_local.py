"""Render the CPU design sensitivity directly from its checked-in JSON output."""
import collections
import html
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def main():
    data=json.loads((HERE/'sensitivity.json').read_text());q=json.loads((HERE/'QUALIFICATION.json').read_text())
    rows=data['rows'];counts=collections.Counter(r['result']['verdict'] for r in rows)
    labels={'FEASIBILITY_ONLY':('F','feas'),'CLOSE_TESTED_CONFIG':('C','close'),
            'QUALITY_INVESTIGATION_ONLY':('Q','quality')}
    body=[]
    for prefill in (.25,1.,4.):
        for cache in (0.,.5,1.):
            group=[r for r in rows if r['prefill_rate_multiplier']==prefill and r['recorded_cache_hit_multiplier']==cache]
            cells=[]
            for row in group:
                result=row['result'];label,cls=labels[result['verdict']]
                ratio='—' if result['ratio'] is None else f"{result['ratio']:.4f}×"
                issue='; '.join(dict.fromkeys(result['control']['issues']+result['candidate']['issues']+result['issues']))
                cells.append(f'<td class="{cls}" title="{html.escape(issue or result["verdict"])}"><b>{label}</b><small>{ratio}</small></td>')
            native=group[0]['result']['control']['native_mean_R']
            body.append(f'<tr><th>{prefill:g}× / {cache:g}</th><td>{native:.2f}</td>{"".join(cells)}</tr>')
    arms=[r['result'][arm] for r in rows for arm in ('control','candidate')]
    assert len(rows)==27 and all(a['load_valid'] and a['load']['idle_fraction']==0 for a in arms)
    content='''<!doctype html><html><head><meta charset="utf-8"><style>
body{font-family:var(--font-sans,system-ui);font-size:14px;color:var(--foreground);line-height:1.45}
h2{font-size:21px;margin:0 0 8px}p{margin:10px 0}.muted,small{color:var(--muted-foreground)}
.note{padding:16px;background:var(--warning-surface,#fef3c7);color:var(--warning-foreground,#92400e);border-radius:var(--radius,8px)}
.stats{display:flex;gap:18px;flex-wrap:wrap;margin:16px 0}.stats strong{font-size:23px;display:block}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:13px;margin:12px 0}
caption{text-align:left;font-weight:650;margin-bottom:6px}th,td{text-align:center;padding:8px 3px;border-bottom:1px solid var(--border,#ddd)}
th:first-child{text-align:left}thead th{font-weight:600}td small{display:block;font-size:11px}
.quality b{color:var(--info-foreground,#245e9f)}.feas{background:var(--muted,#eee)}
.checks{padding-left:18px}.checks li{margin:6px 0}code{font-size:12px;overflow-wrap:anywhere}
@media(max-width:400px){body{font-size:13px}th,td{padding:7px 2px}table{font-size:12px}.stats{gap:12px}}
</style></head><body><h2>S10 capacity design: local reducer qualified</h2>
<p class="note"><b>CPU fixtures only.</b> Actual corpus wire/image/render bytes; synthetic clocks and completions. No model or serving engine ran. No GPU budget, slot request or launch packet.</p>
<div class="stats"><div><strong>26 / 26</strong>contract tests</div><div><strong>54 / 54</strong>offered-load checks</div><div><strong>0</strong>GPU seconds</div></div>
<p>Zero artificial gaps in both arms. Ten HTTP requests stay pending; native decode residence is measured separately and never forced into a matching band.</p>
<table><caption>Prefill/cache sensitivity · not predicted engine speeds</caption>
<thead><tr><th rowspan="2">Prefill rate /<br>cache-hit factor</th><th rowspan="2">Control<br>native R</th><th colspan="3">Hypothetical decode multiplier</th></tr>
<tr><th>1.0×</th><th>1.2×</th><th>1.5×</th></tr></thead><tbody>'''+''.join(body)+'''</tbody></table>
<p><b>F</b> feasibility only; <b>C</b> close this tested configuration; <b>Q</b> quality investigation only. Values are qualified synthetic response-rate ratios. F cells have no speed decision.</p>
<p>All cells have ten pending HTTP requests and zero offered-slot loss. Low native R reflects the fixture's serial-prefill/cache model. Nine pairings fail sample or block-stability checks, not native occupancy. Decoder-only acceleration can leave the full response-rate ratio below the 1.20 threshold.</p>
<ul class="checks"><li>Window and initial-holder completion derive from raw phase/response events.</li>
<li>FIFO, handovers, full-image fidelity, response usage and true remaining-stream starvation reconstruct from the immutable corpus and raw events.</li>
<li>Capacity load guards: at most 0.5% missing offered-slot seconds and no eligible vacancy over 0.25 seconds. Native residence has no lower band or pair match.</li>
<li>Independent review passes. Overlapping native tails, FIFO turn bypass and repeated response IDs now reject.</li></ul>
<p class="muted">Trial-1/2 match committed bytes; trial-3's 56-file inventory remains unchanged and DESIGN BLOCKED. Open: the historical 11-token runtime offset, native adapters, actual dispatch/capture, processor behavior, memory fit and supervisor. Runtime/quantization/drafter confounds remain. Guarded production is unqualified.</p>
<p class="muted">This uncalibrated CPU model challenges the design; it establishes neither performance nor runtime pass probability. Local card: <code>serving-s10-capacity-local-4-20261022</code>.</p></body></html>'''
    (HERE/'local-review.html').write_text(content)
    print(json.dumps(dict(cells=len(rows),verdicts=dict(counts),tests=q['tests'],html='local-review.html')))


if __name__=='__main__':main()
