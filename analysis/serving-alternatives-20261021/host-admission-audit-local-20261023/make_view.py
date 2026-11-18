"""Generate a compact source-bound HTML analysis sheet; no external resources."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
r=json.loads((HERE/'derived-original/AUDIT.json').read_text());base=HERE.parent
samples=json.loads((base/'trial-12-bootstrap-local/remote-readback/derived-original/SAMPLE-SERIES.json').read_text())
term=json.loads((base/'trial-12-bootstrap-local/remote-readback/derived-original/RAW-REDUCTION.json').read_text())['signals'][0]['start']
points=[[s['end']-term,s['host_bytes']/1024**3] for s in samples]
html='''<!doctype html><html><head><meta charset="utf-8"><style>
body{margin:0;color:var(--foreground,#222);font:14px/1.5 var(--font-sans,system-ui)}h2{font-size:20px;line-height:1.25;margin:0 0 8px}h3{font-size:15px;margin:20px 0 7px}p{margin:8px 0;color:var(--muted-foreground,#555)}strong{color:var(--foreground,#222)}svg{width:100%;height:230px;display:block}table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;vertical-align:top;border-bottom:1px solid var(--border,#ddd);padding:8px 6px 8px 0}th{color:var(--muted-foreground,#555)}td:nth-child(2){font-variant-numeric:tabular-nums;white-space:nowrap}small{font-size:12px;color:var(--muted-foreground,#555)}ol{padding-left:22px;margin:6px 0}li{margin:9px 0}code{font-size:12px;word-break:break-word}details{margin-top:14px}summary{cursor:pointer}.tag{color:var(--warning-foreground,#986700);font-weight:650}.box{padding:16px;border:1px solid var(--border,#ddd);border-radius:var(--radius,8px);background:var(--card,transparent);margin-top:12px}@media(max-width:420px){body{font-size:13px}table{font-size:12px}th,td{padding-right:5px}h2{font-size:18px}}
</style></head><body>
<h2>Host admission: the peak we have not measured</h2>
<p><span class="tag">Matched capacity remains blocked.</span> The original 140GiB gate failed; no retained derivation ties it to the next loader’s peak. No threshold has been changed.</p>
<h3>Release 12: measured headroom under the original formula</h3>
<svg id="plot" role="img" aria-label="85 samples remain below the140GiB host gate, settling near111.48GiB"></svg>
<small>85 samples; 82 timely joins pass all nonhost predicates. Only frontend TERM recorded. Lines connect observations, not guaranteed continuous bounds.</small>
<h3>These are different quantities</h3>
<table><thead><tr><th>Quantity</th><th>GiB</th><th>Meaning / limitation</th></tr></thead><tbody>
<tr><td>Trial 7 total cgroup capacity</td><td>175</td><td>Total cap. Startup’s 170GiB guard checks capacity, not remaining headroom.</td></tr>
<tr><td>Release 12 last headroom</td><td>111.48</td><td>Original max−current+inactive_file, capped by host MemAvailable.</td></tr>
<tr><td>Candidate pinned CPU tables</td><td>96.55</td><td>Source-derived tensor storage. <strong>Not a loading peak or sufficient budget.</strong></td></tr>
<tr><td>Checkpoint files</td><td>167.74</td><td>Disk bytes. Lazy iteration; actual trial7 auto-prefetch skipped.</td></tr>
<tr><td>Model-loading GPU log</td><td>71.91</td><td>GPU accounting. Do not use as CPU demand.</td></tr>
<tr><td>Additional host peak after control</td><td>Unknown</td><td>No continuous trial7 host series / peak trace. Entry margin cannot be established.</td></tr>
</tbody></table>
<p>Config says FP8 PLE, but the pinned <code>compressed-tensors</code> branch chooses BF16 host storage. Trial 7 confirms <code>cpu, pinned=True</code>. PLE 95.37GiB + input embedding 1.18GiB = 96.55GiB of named storage; other demand remains unknown.</p>
<div class="box"><strong>File state is not reclaimability proof.</strong><p>Release 12’s last <code>active_file</code> is 62.81GiB. It is descriptive, not permission to credit those bytes. No active-file addition, cache clearing or memory-reclaim operation was performed.</p></div>
<h3>Admission alternatives — judgment still required</h3>
<ol><li><strong>Keep 140GiB and stop.</strong> Preserves the current contract; this observed pair stays blocked.</li>
<li><strong>Measure the missing demand.</strong> Smallest future discriminator: one exact post-control candidate load → actual health 200, normal cache behavior, phase-bound host/current/peak/OOM evidence. No replay or rate claim. Separate bounded risk decision and authority required; no packet built here.</li>
<li><strong>Derive a new rule only from a defensible bound.</strong> Incremental peak plus an independent margin, in the same scope. 96.55GiB is a lower bound on storage, not that bound. A different PLE format, loader or host envelope is a new configuration.</li></ol>
<details><summary>Evidence and uncertainty</summary><p>Five vLLM source files match trial7’s installed hashes; config matches its exact asset digest. Pure pinned vocabulary layout and independent prime calculation agree. Trial 7’s 10.78/3.42GiB AvailableRAM entries occur after construction began; they are not entry peaks. Read-only memory.peak can retain the control’s earlier high-water mark; process RSS can double-count shared pages; sampled maxima can miss transients. One fit observation would not yield a universal admission threshold.</p><p><code>host-admission-audit-local-20261023/derived-original/AUDIT.json</code> binds 24 source/receipt files. Release 12 archive and all 229 freeze bindings verified; monitor/follow-up disabled. Local CPU/source audit only.</p></details>
<script>
const data=__DATA__;const svg=document.querySelector('#plot');function draw(){const w=Math.max(280,svg.clientWidth),h=230,L=46,R=10,T=12,B=34,ns='http://www.w3.org/2000/svg';svg.setAttribute('viewBox',`0 0 ${w} ${h}`);svg.replaceChildren();const X=x=>L+x/180*(w-L-R),Y=y=>h-B-y/160*(h-T-B);function add(tag,a,text){let e=document.createElementNS(ns,tag);for(const[k,v]of Object.entries(a))e.setAttribute(k,v);if(text)e.textContent=text;svg.append(e)}for(const v of[0,40,80,120,160]){add('line',{x1:L,x2:w-R,y1:Y(v),y2:Y(v),stroke:'var(--border,#ddd)'});add('text',{x:L-6,y:Y(v)+4,'text-anchor':'end','font-size':11,fill:'var(--muted-foreground,#555)'},String(v))}for(const t of[0,60,120,180])add('text',{x:X(t),y:h-B+17,'text-anchor':t===180?'end':'middle','font-size':11,fill:'var(--muted-foreground,#555)'},String(t));add('text',{x:L,y:10,'font-size':11,fill:'var(--muted-foreground,#555)'},'GiB');add('line',{x1:L,x2:w-R,y1:Y(140),y2:Y(140),stroke:'var(--warning,#a67815)','stroke-dasharray':'5 4'});add('text',{x:w-R-2,y:Y(140)-6,'text-anchor':'end','font-size':11,fill:'var(--warning-foreground,#986700)'},'140GiB gate');add('path',{d:data.map((p,i)=>(i?'L':'M')+X(p[0])+','+Y(p[1])).join(' '),fill:'none',stroke:'var(--chart-1,#397fc6)','stroke-width':2});add('text',{x:w-R-2,y:Y(111.48)+17,'text-anchor':'end','font-size':11,fill:'var(--foreground,#222)'},'last 111.48GiB');add('text',{x:(L+w-R)/2,y:h-2,'text-anchor':'middle','font-size':11,fill:'var(--muted-foreground,#555)'},'seconds after frontend TERM');console.log('host series',data.length,'last',data.at(-1)[1]);}draw();window.addEventListener('resize',draw);
</script></body></html>'''.replace('__DATA__',json.dumps(points))
(HERE/'host-admission.html').write_text(html)
