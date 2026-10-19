"""Render the descriptive comparisons in RESULT.json; no statistical model or inference."""
import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
r = json.loads((HERE / "RESULT.json").read_text())
scale = lambda v: 145 + (v - 20) / 14 * 460
marks = []
for tick in [20, 24, 28, 32, 34]:
    x = scale(tick)
    marks.append(f'<line x1="{x}" x2="{x}" y1="25" y2="315" stroke="var(--border,#ddd)"/>')
    marks.append(f'<text x="{x}" y="339" text-anchor="middle" class="small">{tick}</text>')
mean = scale(r["own_control"]["mean"])
marks.append(f'<line x1="{mean}" x2="{mean}" y1="22" y2="315" stroke="var(--muted-foreground,#777)" stroke-dasharray="4 4"/>')
names = {"D0": "Raw · D0", "D1": "Evidence · D1", "D2": "Raw · D2", "D3": "B1 + c14 · D3", "D4": "Raw · D4", "D5": "B1 · D5", "D7": "Raw · D7"}
for i, row in enumerate(r["observations"]):
    y = 42 + i * 43
    color = 'var(--chart-1,#6ca4e9)' if row['control'] else 'var(--chart-2,#e8ad68)'
    x = scale(row['score'])
    marks.append(f'<text x="0" y="{y+5}">{html.escape(names[row["draw"]])}</text>')
    marks.append(f'<circle cx="{x}" cy="{y}" r="{7 if row["draw"] == "D7" else 5}" fill="{color}"/>')
    marks.append(f'<text x="{x+13}" y="{y+5}" font-weight="{700 if row["draw"] == "D7" else 400}">{row["score"]:.2f}</text>')
svg = '<svg viewBox="0 0 728 355" role="img" aria-label="All seven hidden draws. Four unchanged controls range from 23.87 to 29.44, surrounding B1 at 26.96.">' + ''.join(marks) + '</svg>'
a, b = r["b1_replicates"]
page = f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>
body{{margin:0;color:var(--foreground,#232323);font:15px/1.5 var(--font-sans,system-ui)}}h2{{font-size:21px;line-height:1.25;margin:0 0 8px}}h3{{font-size:16px;margin:18px 0 4px}}p{{margin:5px 0 10px}}.small,.muted{{color:var(--muted-foreground,#777);font-size:13px}}svg{{display:block;width:100%;height:auto;max-height:355px}}svg text{{fill:var(--foreground,#232323);font:14px var(--font-sans,system-ui)}}svg .small{{fill:var(--muted-foreground,#777);font-size:12px}}.facts{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}.fact{{background:var(--secondary,#eee);padding:16px;border-radius:var(--radius,8px)}}.big{{font-size:24px;font-weight:650}}table{{width:100%;border-collapse:collapse;font-size:14px}}td,th{{text-align:left;padding:6px 8px;border-bottom:1px solid var(--border,#ddd)}}td:first-child,th:first-child{{padding-left:0}}th{{font-weight:500;color:var(--muted-foreground,#777)}}@media(max-width:450px){{.facts{{grid-template-columns:1fr}}svg text{{font-size:15px}}}}
</style></head><body><h2>D7 is a high baseline repeat</h2><p>The same version already scored 29.44 on D2. D7 scored 29.38.</p>
{svg}<p class="small">Public scores on hidden games. Dashed line: our unchanged-control mean, 26.915. Points show all four controls and each of the three candidate packages, once each. D0 was October 1; the other draws were October 10–16. Row spacing does not encode time.</p>
<div class="facts"><div class="fact"><div class="big">+0.045 points</div>B1 relative to the four-control mean.<br><span class="small">One B1 hidden draw cannot establish benefit, harm or equivalence.</span></div><div class="fact"><div class="big">9 of 26</div>Scores in the earlier plain-family ledger reached at least D7.<br><span class="small">Selected external notebooks and mixed version bindings; descriptive context only.</span></div></div>
<h3>Identical public B1 code also produces different outcomes</h3><table><tr><th>Measure</th><th>Repeat 1</th><th>Repeat 2</th></tr><tr><td>Levels completed</td><td>{a['levels']}</td><td>{b['levels']}</td></tr><tr><td>Mean score</td><td>{a['mean_score']:.2f}</td><td>{b['mean_score']:.2f}</td></tr><tr><td>Generated tokens</td><td>{a['generated_tokens']:,}</td><td>{b['generated_tokens']:,}</td></tr></table><p class="small">Same 25 public games and configured budgets. Only 1.41% more tokens, but 15 more levels. This demonstrates variability in B1, not the causes of D7's hidden outcome.</p>
<p class="muted">D7 hidden per-game attribution remains unavailable: episode lists were empty and submission downloads returned HTTP 403. Verified October 17, 2026. No new inference or submissions.</p></body></html>'''
(HERE / "sheet.html").write_text(page)
print(HERE / "sheet.html")
