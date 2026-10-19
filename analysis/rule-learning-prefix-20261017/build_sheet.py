"""Build the standalone experimental readout from preserved receipts."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
r = json.loads((HERE / "RESULT.json").read_text())
tracks = json.loads((HERE / "inputs.json").read_text())
track = next(t for t in tracks if t["id"] == "tn36-b1-r1")
audit = json.loads((ROOT / "analysis/failure-audit-20261013/tn36/engine-replay.json").read_text())
command = next(c for c in audit["runs"]["b1-r1"]["commands"] if c["local_action"] == 41)
before = track["actions"][39]["board"]
mid = command["animation_frames"][1]
after = track["actions"][40]["board"]


def key_mask(board):
    return [[int(board[y][x] == 11) for x in range(45, 49)] for y in range(24, 28)]


assert key_mask(before) == key_mask(after)
assert key_mask(before) != key_mask(mid)


def pixels(board):
    parts = []
    for y, row in enumerate(key_mask(board)):
        for x, on in enumerate(row):
            color = "var(--warning,#f2cc45)" if on else "var(--muted,#303030)"
            parts.append(f'<rect x="{x*20}" y="{y*20}" width="19" height="19" fill="{color}"/>')
    return '<svg viewBox="0 0 80 80" width="100" height="100" role="img" aria-label="Observed yellow key pixel mask">' + ''.join(parts) + '</svg>'


rows = []
for game, label in [("cn04", "Joining puzzle"), ("s5i5", "Coupled growth"), ("tn36", "Programmed controls")]:
    q = r["per_game"][game]
    rows.append(f'<tr><td>{game} · {label}</td><td>{q["exact_on_raw_changed"]} / {q["raw_changed"]}</td><td>{q["explained_events"]} / {q["observed_events"]}</td></tr>')

page = f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>
body{{margin:0;color:var(--foreground,#242424);font:15px/1.5 var(--font-sans,system-ui)}}h2{{font-size:21px;line-height:1.3;margin:0 0 8px}}h3{{font-size:16px;margin:20px 0 8px}}p{{margin:6px 0 12px}}.muted{{color:var(--muted-foreground,#777);font-size:13px}}.masks{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;text-align:center}}.mask{{background:var(--secondary,#eee);border-radius:var(--radius,8px);padding:16px}}.mask svg{{display:block;margin:12px auto 4px}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{text-align:left;vertical-align:top;padding:9px 8px;border-bottom:1px solid var(--border,#ddd)}}th{{font-weight:500;color:var(--muted-foreground,#777)}}th:first-child,td:first-child{{padding-left:0}}.finding{{padding:16px;background:var(--secondary,#eee);border-radius:var(--radius,8px)}}code{{font-size:12px;overflow-wrap:anywhere}}details{{margin:12px 0}}@media(max-width:450px){{.masks{{gap:7px}}.mask{{padding:16px 8px;font-size:12px}}.mask svg{{width:75px;height:75px}}table{{font-size:12px}}}}
</style></head><body><h2>Rule-learning experiment: observation and prediction fail differently</h2><p class="muted">Six archived level trajectories · 256 future transitions scored · 38 CPU seconds · zero model calls</p>
<h3>A decisive effect disappears before the final frame</h3><div class="masks"><div class="mask">Before action{pixels(before)}</div><div class="mask">During animation{pixels(mid)}</div><div class="mask">Settled result{pixels(after)}</div></div>
<p class="muted">tn36, repeat 1, local action 41. Actual yellow-key pixel masks at screen rows 24–27, columns 45–48. The key changes orientation during the action and returns. These are observations from a previously verified replay; engine state was never supplied to the learner.</p>
<div class="finding">The unchanged lexicon encoder reports an orientation change on all four intermediate frames checked. On the settled frame, it reports only two interface changes. All five comparisons use identical frozen rules and past observations.</div>
<h3>Describing changes does not establish a predictive rule model</h3><table><thead><tr><th>Selected game</th><th>Complete effect set correct<br>on changed boards</th><th>Effects explained<br>by prior rules</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<p class="muted">Overall: 22 / 248 changed-board transitions fully predicted; 168 / 663 extracted effects explained. Counts refer to the donor's object representation, not complete native state or gameplay success. An empty-effect baseline fully predicts 0 / 248 changed boards. Eight unchanged boards are reported separately in RESULT.json.</p>
<h3>The coupling is visible, but its rule is wrong</h3><p>At the fixed s5i5 checkpoint, the observer reports both coupled objects moving upward by 3 pixels. The prior learned rule predicts one moving right by 3 instead. The information is present here; the generalization fails.</p>
<p class="muted">On the cn04 joining checkpoint, the output is only “object became another shape.” It does not establish which connectors joined or validate the model's geometry.</p>
<details><summary>What the experiment can establish</summary><p>Each prediction was frozen before its target frame was encoded. The scale and rules used only preceding frames. The original candidate and inputs passed their frozen hashes; 256 records scored without errors or resource limits. One level boundary and one RESET were excluded explicitly.</p><p>This is a diagnostic on selected known failures, with complete recorded past frames. The production sandbox may retain less history. It tests the learner's capability, not model adoption, rule repair, action selection, or score improvement. The animation comparison is a separately carded post-run observation ablation on one case.</p><p>Reproduce the summaries with <code>python3 -B analysis/rule-learning-prefix-20261017/readout.py</code>. Source, frozen cards, raw per-transition predictions and verification are preserved beside this page.</p></details></body></html>'''
(HERE / "sheet.html").write_text(page)
print(HERE / "sheet.html")
