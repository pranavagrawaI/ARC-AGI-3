"""Human-facing report and interactive view for epistemic-funnel results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from tools.epistemic_funnel_core import STAGES
except ModuleNotFoundError:  # pragma: no cover - direct script entry point
    from epistemic_funnel_core import STAGES


FAMILY_LABELS = {
    "goal_mechanism_failure": "Goal → probe",
    "handoff_failure": "Action handoff",
    "consolidation_failure": "Terminal consolidation",
}


def money(value: float | None) -> str:
    return "unknown" if value is None else f"${value:.4f}"


def render_report(result: dict[str, Any]) -> str:
    runs = {run["key"]: run for run in result["runs"]}
    horizon = result["parameters"]["goal_obligation_actions"]
    episodes = result["failure_episodes"]
    model_findings = result["model_fidelity_findings"]
    top = result["shadow_candidates"][0]

    lines = [
        "# Zero-cost epistemic-funnel census",
        "",
        "## Judgment",
        "",
        f"At a **{horizon}-action provisional-goal obligation**, all five development traces expose one supported first broken control boundary:",
        "",
        "- **2 goal-to-probe failures**: ar25 goal-gate and LS20 Gemma feasibility;",
        "- **2 terminal-to-consolidation failures**: both successful ar25 symmetry traces;",
        "- **1 selected-action-to-host failure**: LS20 Gemma reasoning capture.",
        "",
        "The only failure family that repeats **across games and model lineages** is candidate objective → named discriminator. That makes the architecture-wide goal-to-probe hypothesis stronger than before, but this five-trace development census is not a cross-game reliability estimate.",
        "",
        "The census also rejects a tempting simplification: model fidelity is not the first traceable failure in these five episodes, yet independent saved-model tests find defects in both games. Therefore, objective acquisition should be investigated first **without assuming** mechanics/planning will become sufficient afterward.",
        "",
        "## Census",
        "",
        "| Run | First supported boundary | Seq | Actor before failure | Builder before failure | Evidence class |",
        "|---|---|---:|---:|---:|---|",
    ]
    for episode in episodes:
        run = runs[episode["run_key"]]
        spend = episode.get("spend") or {}
        lines.append(
            f"| {run['label']} | {episode['boundary']} | {episode.get('sequence') if episode.get('sequence') is not None else 'censored'} "
            f"| {money(spend.get('actor_usd'))} | {money(spend.get('builder_usd'))} | {run['evidence_class']} |"
        )

    lines += [
        "",
        "**Unit:** one deduplicated first observable failure episode per preserved run-level trace. Later failures remain visible in the decision table but do not inflate the aggregate. Missing cost is reported as unknown, never zero.",
        "",
        "### Horizon sensitivity",
        "",
        "| Obligation (actions) | Goal → probe | Handoff | Consolidation | Censored/no supported failure |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in result["horizon_sensitivity"]:
        counts = row["family_counts"]
        lines.append(
            f"| {row['goal_obligation_actions']} | {counts.get('goal_mechanism_failure', 0)} "
            f"| {counts.get('handoff_failure', 0)} | {counts.get('consolidation_failure', 0)} "
            f"| {counts.get('censored_or_no_supported_failure', 0)} |"
        )
    lines += [
        "",
        "The exact mix is horizon-sensitive. At five or more actions the short Gemma feasibility trace becomes censored, so the cross-game recurrence becomes **unresolved**, not contradicted. At one or two actions, the obligation fires before later evidence can reveal the timeout handoff. Three actions is therefore an explicit working policy, not an empirically calibrated constant.",
        "",
        "## What repeats",
        "",
        "1. **Goal mechanism, across ar25 and LS20.** A candidate objective exists, but no objective-discriminating experiment appears before the bounded obligation expires.",
        "2. **Terminal consolidation, twice on ar25.** Successful action suffixes reach host-confirmed completion, but the live trace lacks a resolved outcome checkpoint afterward. This is a repeated lifecycle defect, not cross-game capability evidence.",
        "3. **Model-fidelity defects, independently in both games.** These are not dated as the earliest historical failure:",
    ]
    for finding in model_findings:
        lines.append(f"   - {finding['finding']} ({finding['status']}); {finding['claim_boundary']} Evidence: `{finding['evidence_ref']}`.")

    lines += [
        "",
        "## Best discriminating checkpoint",
        "",
        f"**Exact actor call {top['call_index']} at LS20 sequence {top['sequence']}** is the highest-value next paid shadow: `{top['evidence_ref']}`.",
        "",
        f"State: {top['causal_state']}. The current actor inspects a route while the terminal predicate is still unverified. The one factor is: **{top['one_factor']}**",
        "",
        "Measure the first response for: named hypothesis, predicted observation, selected discriminator, legality, and a promote/demote rule. Two repeats per arm are enough initially; expand only if within-arm variability blocks judgment.",
        "",
        "No single frozen shadow separates all four proposed causes. Call 36 directly separates **goal-mechanism failure** from **evidence-use failure**. **Handoff** needs the scripted only-legal-action/lifecycle control, and **model fidelity** needs the offline replay/contract control. Treating one response perturbation as a four-way diagnosis would be self-deception.",
        "",
        "## Positive controls and validity",
        "",
    ]
    for control in result["positive_controls"]:
        lines.append(f"- {'PASS' if control['pass'] else 'FAIL'} — {control['name']} (expected `{control['expected']}`, observed `{control['observed']}`).")
    lines += [
        "",
        f"The miner processed **{result['aggregate']['decision_sequences']} decision sequences**, **{result['aggregate']['exact_checkpoint_calls']} exact checkpoint calls**, and **{result['aggregate']['llm_trace_calls']} accounting/response rows**. It made **0 model calls and 0 ARC environment calls**.",
        "",
        "## Claim ceiling",
        "",
    ]
    lines.extend(f"- {claim}" for claim in result["claim_ceiling"])
    lines += [
        "",
        "## Artifacts",
        "",
        "- `funnel.json` — canonical evidence, hashes, detectors, episodes, and shadow ranking.",
        "- `decisions.csv` — one row per decision sequence with cumulative stages and spend.",
        "- `index.html` — filterable timeline, horizon sensitivity, evidence drill-down, and state export.",
        "",
    ]
    return "\n".join(lines)


def write_report(result: dict[str, Any], path: Path) -> None:
    path.write_text(render_report(result), encoding="utf-8")


def _dashboard_data(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": result["schema"],
        "parameters": result["parameters"],
        "execution": result["execution"],
        "positive_controls": result["positive_controls"],
        "aggregate": result["aggregate"],
        "failure_episodes": result["failure_episodes"],
        "horizon_sensitivity": result["horizon_sensitivity"],
        "model_fidelity_findings": result["model_fidelity_findings"],
        "shadow_candidates": result["shadow_candidates"],
        "claim_ceiling": result["claim_ceiling"],
        "stages": list(STAGES),
        "runs": [
            {
                key: run[key]
                for key in (
                    "key", "label", "path", "game_id", "seed", "evidence_class", "summary",
                    "spend_alignment", "signals", "decisions", "terminal_sequences",
                    "consolidated_sequences", "max_sequence",
                )
            }
            for run in result["runs"]
        ],
    }


def render_html(result: dict[str, Any]) -> str:
    payload = json.dumps(_dashboard_data(result), ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Epistemic Funnel — Tycho/NOOA</title>
<style>
:root{{--ink:#17201d;--muted:#62706a;--paper:#f5f1e8;--card:#fffdf7;--line:#d9d1c2;--green:#1f6b55;--red:#a34436;--amber:#aa6a15;--blue:#315d86;--shadow:0 12px 34px rgba(31,42,37,.08)}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}}
header{{padding:34px clamp(20px,5vw,72px) 26px;border-bottom:1px solid var(--line);background:linear-gradient(125deg,#e9f0e7,#f7f1e3 58%,#eee3d5)}}
.eyebrow{{text-transform:uppercase;letter-spacing:.13em;font-size:11px;font-weight:800;color:var(--green)}} h1{{font:700 clamp(30px,5vw,54px)/1.05 ui-serif,Georgia,serif;margin:8px 0 12px;max-width:900px}}
.lede{{max-width:940px;font-size:18px;color:#394640;margin:0}} main{{max-width:1480px;margin:auto;padding:24px clamp(16px,3vw,42px) 60px}}
.controls{{display:flex;gap:16px;align-items:end;flex-wrap:wrap;margin-bottom:20px}} label{{display:grid;gap:5px;font-size:12px;font-weight:750;color:var(--muted)}} select,input{{border:1px solid var(--line);border-radius:8px;background:var(--card);padding:9px;color:var(--ink)}} input[type=range]{{padding:0;width:170px}}
button{{border:1px solid var(--green);background:var(--green);color:white;border-radius:8px;padding:9px 13px;font-weight:750;cursor:pointer}} button.secondary{{background:transparent;color:var(--green)}}
.grid{{display:grid;grid-template-columns:repeat(12,1fr);gap:16px}} .card{{background:var(--card);border:1px solid var(--line);border-radius:13px;padding:18px;box-shadow:var(--shadow)}}
.span12{{grid-column:span 12}} .span8{{grid-column:span 8}} .span7{{grid-column:span 7}} .span5{{grid-column:span 5}} .span4{{grid-column:span 4}}
h2{{font:700 24px/1.2 ui-serif,Georgia,serif;margin:0 0 12px}} h3{{font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin:0 0 10px}}
.kpis{{display:grid;grid-template-columns:repeat(4,minmax(120px,1fr));gap:10px}} .kpi{{border-left:4px solid var(--green);padding:8px 12px;background:#f6f7f0;border-radius:6px}} .kpi strong{{display:block;font:700 28px/1 ui-serif,Georgia,serif}} .kpi small{{color:var(--muted)}}
.barrow{{display:grid;grid-template-columns:minmax(125px,1.2fr) 3fr 40px;gap:10px;align-items:center;margin:10px 0}} .bar{{height:16px;background:#ece6d9;border-radius:99px;overflow:hidden}} .fill{{height:100%;background:var(--red);border-radius:inherit}}
.matrix,.timeline{{overflow:auto}} table{{width:100%;border-collapse:collapse;font-size:13px}} th{{position:sticky;top:0;background:var(--card);z-index:1;text-align:left;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.06em}} th,td{{border-bottom:1px solid #e7e0d3;padding:9px 8px;vertical-align:top}}
.dot{{display:inline-block;width:10px;height:10px;border-radius:50%;background:#d7d0c2}} .dot.on{{background:var(--green)}} .dot.posthoc{{background:var(--amber)}}
.pill{{display:inline-block;border:1px solid var(--line);border-radius:99px;padding:2px 7px;margin:1px;font-size:11px;background:#faf7ef}} .pill.goal{{border-color:#d6a989;color:#7d321f}} .pill.host{{border-color:#9fbdb2;color:#185342}}
.mono{{font:12px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace}} details{{border-top:1px solid var(--line);padding:10px 0}} summary{{cursor:pointer;font-weight:750}} .evidence{{background:#f1ede3;padding:10px;border-radius:7px;white-space:pre-wrap;overflow-wrap:anywhere}}
.callout{{border-left:5px solid var(--amber);background:#fff5dc;padding:13px 15px;border-radius:7px}} .ok{{color:var(--green)}} .bad{{color:var(--red)}} .muted{{color:var(--muted)}}
@media(max-width:980px){{.span8,.span7,.span5,.span4{{grid-column:span 12}}.kpis{{grid-template-columns:repeat(2,1fr)}}}} @media(max-width:600px){{.kpis{{grid-template-columns:1fr}}}}
@media print{{.controls,button{{display:none!important}}body{{background:white}}.card{{box-shadow:none;break-inside:avoid}}}}
</style>
</head>
<body>
<header><div class="eyebrow">Zero-call control-system census</div><h1>Where does evidence stop changing behavior?</h1><p class="lede">A trace-grounded epistemic funnel over ar25 and LS20. Move the obligation horizon; inspect the first broken boundary, spend accumulated before it, and the exact evidence that licenses each claim.</p></header>
<main>
<div class="controls">
  <label>Run <select id="runFilter"><option value="all">All runs</option></select></label>
  <label>Failure family <select id="familyFilter"><option value="all">All families</option><option value="goal_mechanism_failure">Goal → probe</option><option value="handoff_failure">Action handoff</option><option value="consolidation_failure">Terminal consolidation</option></select></label>
  <label>Goal obligation <span><b id="horizonValue">3</b> actions</span><input id="horizon" type="range" min="1" max="6" value="3"></label>
  <button id="export">Export view state</button><button class="secondary" onclick="window.print()">Print</button>
</div>
<div class="grid">
  <section class="card span12"><h2>Judgment at this horizon</h2><div class="kpis" id="kpis"></div><p id="judgment"></p></section>
  <section class="card span5"><h2>First-failure episodes</h2><div id="bars"></div><p class="muted">One episode per trace; downstream absences are not re-counted.</p></section>
  <section class="card span7"><h2>Spend before the first failure</h2><div class="matrix"><table><thead><tr><th>Run</th><th>Boundary</th><th>Seq</th><th>Actor</th><th>Builder</th><th>Alignment</th></tr></thead><tbody id="spendRows"></tbody></table></div></section>
  <section class="card span12"><h2>Run × stage reachability</h2><div class="matrix"><table id="matrix"></table></div><p class="muted">A dot means the trace contains evidence for the stage. It does not certify correctness. Amber terminal-derived validation is post hoc.</p></section>
  <section class="card span8"><h2>Decision timeline</h2><div class="timeline"><table><thead><tr><th>Run</th><th>Seq</th><th>Action</th><th>Host?</th><th>New evidence crossing boundary</th><th>Cumulative spend</th></tr></thead><tbody id="timelineRows"></tbody></table></div></section>
  <aside class="card span4"><h2>Highest-value next shadow</h2><div id="shadow"></div><h3 style="margin-top:20px">Why not one four-way shadow?</h3><p class="callout">The exact packet separates goal mechanism from evidence use. Handoff requires a lifecycle positive control; model fidelity requires replay/contract validation. One response perturbation cannot identify all four.</p></aside>
  <section class="card span7"><h2>Evidence drill-down</h2><div id="evidence"></div></section>
  <section class="card span5"><h2>Validity and claim ceiling</h2><div id="controls"></div><div id="ceiling"></div></section>
</div>
</main>
<script id="data" type="application/json">{payload}</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent); const $=id=>document.getElementById(id);
const labels={{goal_mechanism_failure:'Goal → probe',handoff_failure:'Action handoff',consolidation_failure:'Terminal consolidation'}};
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
const money=v=>v==null?'unknown':'$'+Number(v).toFixed(4);
function classify(run,h){{
 const first=stage=>{{const s=run.signals.find(x=>x.stage===stage);return s?s.sequence:null}}; const choices=[];
 const c=first('candidate_objective'), d=first('discriminator_named');
 if(c===null && run.max_sequence>=h) choices.push({{sequence:h,family:'goal_mechanism_failure',boundary:'observation → candidate objective'}});
 else if(c!==null && (d===null||d>c+h) && run.max_sequence>=c+h) choices.push({{sequence:c+h,family:'goal_mechanism_failure',boundary:'candidate objective → named discriminator'}});
 const missed=run.decisions.find(x=>x.selected_action&&!x.host_executed); if(missed) choices.push({{sequence:missed.sequence,family:'handoff_failure',boundary:'selected action → host execution'}});
 const t=run.terminal_sequences[0]; if(t!==undefined&&!run.consolidated_sequences.some(x=>x>=t)) choices.push({{sequence:t,family:'consolidation_failure',boundary:'terminal consequence → durable consolidation'}});
 choices.sort((a,b)=>a.sequence-b.sequence); return choices[0]||{{sequence:null,family:null,boundary:'censored / no supported failure'}};
}}
const runSelect=$('runFilter'); D.runs.forEach(r=>runSelect.insertAdjacentHTML('beforeend',`<option value="${{esc(r.key)}}">${{esc(r.label)}}</option>`));
function filteredRuns(){{return D.runs.filter(r=>$('runFilter').value==='all'||r.key===$('runFilter').value)}}
function render(){{
 const h=+$('horizon').value; $('horizonValue').textContent=h; const family=$('familyFilter').value;
 let rows=filteredRuns().map(run=>({{run,episode:classify(run,h)}})); if(family!=='all') rows=rows.filter(x=>x.episode.family===family);
 const counts={{}}; rows.forEach(x=>{{if(x.episode.family)counts[x.episode.family]=(counts[x.episode.family]||0)+1}}); const supported=Object.values(counts).reduce((a,b)=>a+b,0);
 $('kpis').innerHTML=`<div class="kpi"><strong>${{rows.length}}</strong><small>visible traces</small></div><div class="kpi"><strong>${{supported}}</strong><small>supported first failures</small></div><div class="kpi"><strong>${{D.aggregate.exact_checkpoint_calls}}</strong><small>exact checkpoint calls</small></div><div class="kpi"><strong>0 / 0</strong><small>model / ARC calls</small></div>`;
 const cross=counts.goal_mechanism_failure>=2?'Goal → probe remains the only cross-game recurring first failure.':'The selected view does not establish a cross-game recurring family.'; $('judgment').innerHTML=`<b>${{esc(cross)}}</b> Horizon changes reclassify bounded obligations; host handoff and terminal consolidation do not depend on it.`;
 const max=Math.max(1,...Object.values(counts)); $('bars').innerHTML=Object.keys(labels).map(k=>`<div class="barrow"><span>${{labels[k]}}</span><div class="bar"><div class="fill" style="width:${{100*(counts[k]||0)/max}}%"></div></div><b>${{counts[k]||0}}</b></div>`).join('');
 $('spendRows').innerHTML=rows.map(({{run,episode}})=>{{const d=episode.sequence==null?null:run.decisions[Math.min(episode.sequence,run.decisions.length-1)];return `<tr><td><b>${{esc(run.label)}}</b></td><td>${{esc(episode.boundary)}}</td><td>${{episode.sequence??'—'}}</td><td>${{money(d?.actor_spend_usd_cumulative)}}</td><td>${{money(d?.builder_spend_usd_cumulative)}}</td><td>${{esc(run.spend_alignment)}}</td></tr>`}}).join('');
 const visible=new Set(rows.map(x=>x.run.key)); const short=s=>s.replaceAll('_',' '); $('matrix').innerHTML='<thead><tr><th>Run</th>'+D.stages.map(s=>`<th title="${{esc(s)}}">${{esc(short(s).split(' ')[0])}}</th>`).join('')+'</tr></thead><tbody>'+D.runs.filter(r=>visible.has(r.key)).map(r=>{{const by={{}};r.signals.forEach(s=>by[s.stage]=(by[s.stage]||[]).concat(s));return `<tr><td><b>${{esc(r.label)}}</b></td>`+D.stages.map(s=>`<td title="${{esc((by[s]||[]).map(x=>x.evidence_ref).join('\\n'))}}"><span class="dot ${{by[s]?'on':''}}"></span></td>`).join('')+'</tr>'}}).join('')+'</tbody>';
 $('timelineRows').innerHTML=D.runs.filter(r=>visible.has(r.key)).flatMap(r=>r.decisions.filter(d=>d.recorded_turn||d.stages_new.length||d.selected_action).map(d=>`<tr><td>${{esc(r.label)}}</td><td>${{d.sequence}}</td><td class="mono">${{esc(d.selected_action||'—')}}</td><td>${{d.host_executed?'yes':'<b class="bad">no</b>'}}</td><td>${{d.stages_new.map(s=>`<span class="pill ${{s.includes('objective')||s.includes('discriminator')?'goal':''}}">${{esc(short(s))}}</span>`).join('')||'—'}}</td><td>${{money(d.actor_spend_usd_cumulative)}} / ${{money(d.builder_spend_usd_cumulative)}}</td></tr>`)).join('');
 const top=D.shadow_candidates[0]; $('shadow').innerHTML=`<h3>Exact call ${{top.call_index}} · seq ${{top.sequence}}</h3><p>${{esc(top.causal_state)}}</p><p><b>One factor:</b> ${{esc(top.one_factor)}}</p><p><b>Measure:</b> ${{esc(top.measure)}}</p><p class="mono">${{esc(top.evidence_ref)}}</p>`;
 $('evidence').innerHTML=D.runs.filter(r=>visible.has(r.key)).map(r=>`<details><summary>${{esc(r.label)}} · ${{r.signals.length}} signals</summary>`+r.signals.map(s=>`<div class="evidence"><b>${{esc(short(s.stage))}}</b> · seq ${{s.sequence}} · ${{esc(s.owner)}} · ${{esc(s.detector)}}<br><span class="mono">${{esc(s.evidence_ref)}}</span><br>${{esc(s.snippet)}}</div>`).join('')+'</details>').join('');
 $('controls').innerHTML='<h3>Scripted positive controls</h3>'+D.positive_controls.map(c=>`<p class="${{c.pass?'ok':'bad'}}"><b>${{c.pass?'PASS':'FAIL'}}</b> · ${{esc(c.name)}}</p>`).join(''); $('ceiling').innerHTML='<h3>Claim ceiling</h3><ul>'+D.claim_ceiling.map(x=>`<li>${{esc(x)}}</li>`).join('')+'</ul>';
}}
['runFilter','familyFilter','horizon'].forEach(id=>$(id).addEventListener('input',render)); $('export').addEventListener('click',()=>{{const state={{run:$('runFilter').value,family:$('familyFilter').value,goal_obligation_actions:+$('horizon').value,schema:D.schema}};const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(state,null,2)],{{type:'application/json'}}));a.download='epistemic-funnel-view.json';a.click();URL.revokeObjectURL(a.href)}}); render();
</script>
</body></html>"""


def write_html(result: dict[str, Any], path: Path) -> None:
    path.write_text(render_html(result), encoding="utf-8")
