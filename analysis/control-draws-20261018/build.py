"""Hidden-draw dot plot: unchanged controls vs single candidate draws vs medal lines.
Sources: draw-variance-20261018/FINDINGS.md (controls, medal lines, 10-18 board);
d7-attribution-20261017/sheet.html (candidate draws D1/D3/D5)."""
import json, statistics, pathlib
controls = [23.87, 29.44, 24.97, 29.38, 31.31]
cands = [("Evidence · D1", 26.24), ("B1 + c14 · D3", 23.56), ("B1 · D5", 26.96)]
lines = [("D9 ≥", 30.0), ("bronze", 30.64), ("silver", 31.91), ("gold", 38.2)]
mean, sd = statistics.mean(controls), statistics.stdev(controls)
n30 = sum(c >= 30 for c in controls)
assert abs(mean - 27.794) < 1e-9 and abs(sd - 3.20) < 0.01 and n30 == 1

W, L, R, lo, hi = 728, 150, 20, 22, 39
x = lambda v: L + (v - lo) / (hi - lo) * (W - L - R)
rows = [("Control", v, "c") for v in sorted(controls)] + [(n, v, "k") for n, v in cands]
top, step = 46, 30
H = top + step * len(rows) + 40
s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Five hidden control draws, one at or above 30; three single candidate draws, all below the control mean.">']
for t in range(lo, hi + 1, 2):
    s.append(f'<text x="{x(t):.1f}" y="{H-12}" text-anchor="middle" class="tick">{t}</text>')
for name, v in lines:
    col = "var(--muted-foreground,#888)" if name.startswith("D9") else "var(--warning,#d9a400)"
    s.append(f'<line x1="{x(v):.1f}" x2="{x(v):.1f}" y1="30" y2="{H-30}" stroke="{col}" stroke-dasharray="3 4"/>')
lab = {"D9 ≥": (30.0, "end", -4), "bronze": (30.64, "start", 4), "silver": (31.91, "start", 4), "gold": (38.2, "end", -4)}
for i, (name, v) in enumerate(lines):
    _, anchor, dx = lab[name]
    s.append(f'<text x="{x(v)+dx:.1f}" y="{18 + (i % 2) * 0}" text-anchor="{anchor}" class="tick">{name} {v:g}</text>' if name in ("D9 ≥", "gold") else
             f'<text x="{x(v)+dx:.1f}" y="{18 if name=="silver" else 32}" text-anchor="{anchor}" class="tick">{name} {v:g}</text>')
s.append(f'<line x1="{x(mean):.1f}" x2="{x(mean):.1f}" y1="38" y2="{top+step*5-12}" stroke="var(--chart-1,#6ca4e9)" stroke-width="2" stroke-opacity=".8"/>')
for i, (name, v, kind) in enumerate(rows):
    y = top + step * i + (14 if kind == "k" else 0)
    fill = "var(--chart-1,#6ca4e9)" if kind == "c" else "none"
    stroke = "var(--chart-1,#6ca4e9)" if kind == "c" else "var(--chart-2,#e8ad68)"
    bold = ' font-weight="700"' if v >= 30 and kind == "c" else ""
    s.append(f'<text x="0" y="{y+5}">{name}</text><circle cx="{x(v):.1f}" cy="{y}" r="6" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
             f'<text x="{x(v)+11:.1f}" y="{y+5}"{bold}>{v:.2f}</text>')
s.append("</svg>")
html = f"""<!doctype html><html><head><meta charset="utf-8"><style>
body{{margin:0;color:var(--foreground,#222);font:15px/1.5 var(--font-sans,system-ui)}}
h2{{font-size:20px;line-height:1.25;margin:0 0 4px}}p{{margin:4px 0 10px}}
.muted{{color:var(--muted-foreground,#777);font-size:13px}}
.chart{{overflow-x:auto}}svg{{display:block;width:100%;min-width:560px;height:auto}}svg text{{fill:var(--foreground,#222);font:14px var(--font-sans,system-ui)}}svg .tick{{fill:var(--muted-foreground,#777);font-size:12px}}
.facts{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:10px}}.fact{{background:var(--secondary,#eee);padding:16px;border-radius:var(--radius,8px)}}.big{{font-size:22px;font-weight:650}}
@media(max-width:450px){{.facts{{grid-template-columns:1fr}}}}
</style></head><body>
<h2>{n30} of 5 unchanged controls reached 30, not 2</h2>
<p>Hidden public-draw scores. Filled: identical control code. Hollow: one draw each of a candidate package.</p>
<div class="chart">{"".join(s)}</div>
<p class="muted">Blue line: control mean {mean:.3f} (sd {sd:.2f}, n=5). Medal lines are the public leaderboard on 10-18; the private board scores different games from the same run. Rows are sorted by score, not date.</p>
<div class="facts">
<div class="fact"><div class="big">{n30} / 5</div>controls ≥30. #755's FINDINGS line 50 says 2 / 5; #757's correction is right.</div>
<div class="fact"><div class="big">{min(controls):.2f}–{max(controls):.2f}</div>range of identical code: {max(controls)-min(controls):.2f} points, wider than any candidate gap.</div>
<div class="fact"><div class="big">0 / 3</div>candidate draws above the control mean. With one draw each, none is shown worse.</div>
</div></body></html>"""
out = pathlib.Path(__file__).with_name("sheet.html")
out.write_text(html)
print(json.dumps({"mean": mean, "sd": round(sd, 3), "n_ge_30": n30, "bytes": len(html)}))
