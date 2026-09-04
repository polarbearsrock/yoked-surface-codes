"""Web report: LER per patch round (paper unit) against a coarse Micro Blossom cycle count, mean and maximum as separate panels."""
import json, sys
from pathlib import Path
import sinter

cell = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("out/cluster_size_study_d7/d7_p0.003_100k")
out_html = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("out/cluster_size_study_d7/l2_cycles.html")
fr = json.load(open(cell / "frontier_interior.json")); prov = json.load(open(cell / "provenance.json")); cy = json.load(open(cell / "microblossom_cycles.json"))
c = prov["cell"]; N, O = fr["shots"], fr["num_observables"]; PIECES = c["patches"] * c["rounds"]
conv = lambda p: sinter.shot_error_rate_to_piece_error_rate(p, pieces=PIECES, values=O)
gbase = fr["global_failures"] / N; BASE = conv(gbase)
F = {f"{r['family']}:{r['value']}": r for r in fr["rows"]}
rows = []
for r in cy["rows"]:
    k = f"{r['family']}:{r['value']}"
    if r["family"] not in ("none", "size", "margin", "all"): continue
    if k in F:
        f = F[k]; p = f["failures"] / N; paper = conv(p); lo, hi = conv(gbase + f["ci_pp"][0] / 100), conv(gbase + f["ci_pp"][1] / 100); rd = f["risk_difference_pp"]
    else:
        p = prov["summary"]["treatment_failures"] / N; paper = conv(p); lo = hi = paper; rd = 100 * (p - gbase)
    rows.append(dict(key=k, family=r["family"], value=r["value"], name=r["rule"], coverage=r["coverage_pct"], paper=1e3 * paper, paper_lo=1e3 * lo, paper_hi=1e3 * hi, mult=paper / BASE, rd=rd,
                     defects_mean=r["noniso_mean"], defects_max=r["noniso_max"], mean=r["cyc_shot_mean"], p99=r["cyc_shot_p99"], max=r["cyc_shot_max"],
                     rel_mean=100 * r["rel_mean"], rel_p99=100 * r["rel_p99"], rel_max=100 * r["rel_max"], us_round_mean=r["us_round_mean"], us_round_p99=r["us_round_p99"], us_round_max=r["us_round_max"]))
by = {r["key"]: r for r in rows}; none = by["none:None"]; allc = by["all:None"]
data = dict(shots=N, cell=c, pieces=PIECES, observables=O, base=1e3 * BASE, none=none, all=allc, model=cy["model"],
            size=sorted([r for r in rows if r["family"] == "size"], key=lambda r: r["coverage"]), margin=sorted([r for r in rows if r["family"] == "margin"], key=lambda r: r["coverage"]))

html = r"""<title>L2 Cycle Ladders</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{color-scheme:light;--page:#f7f7f4;--surface:#fcfcfb;--ink:#0b0b0b;--ink-2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);
  --s1:#2a78d6;--s2:#eb6834;--accent:#2a78d6;--tip:#ffffff;--goal:#16a34a;--font:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;--mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#ffffff;--ink-2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--accent:#3987e5;--tip:#242423;--goal:#4ade80}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#ffffff;--ink-2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--accent:#3987e5;--tip:#242423;--goal:#4ade80}
body{margin:0;background:var(--page);color:var(--ink);font-family:var(--font);font-size:15px;line-height:1.5}
main{max-width:1040px;margin:0 auto;padding:40px 24px 64px;display:flex;flex-direction:column;gap:36px}
h1{font-size:30px;font-weight:600;letter-spacing:-.01em;line-height:1.15;margin:0 0 8px;text-wrap:balance}
h2{font-size:19px;font-weight:600;margin:0 0 4px}
p{max-width:70ch;margin:0} .lede{color:var(--ink-2);font-size:16px}
.eyebrow{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin-bottom:12px;font-weight:500}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px}
.tile{background:var(--surface);border:1px solid var(--ring);border-radius:6px;padding:16px 18px}
.tile .label{font-size:13px;color:var(--ink-2)} .tile .value{font-size:28px;font-weight:600;margin-top:4px;line-height:1.1;font-variant-numeric:tabular-nums} .tile .sub{font-size:13px;color:var(--muted);margin-top:6px}
.controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap;font-size:13px;color:var(--ink-2)}
.controls button{font:inherit;background:var(--surface);color:var(--ink);border:1px solid var(--ring);border-radius:4px;padding:4px 10px;cursor:pointer}
.controls button[aria-pressed="true"]{background:var(--ink);color:var(--surface);border-color:var(--ink)}
.controls button:focus-visible,.hit:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
section{display:flex;flex-direction:column;gap:12px}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:6px;padding:20px 20px 12px;position:relative}
svg.chart{display:block;width:100%;height:auto;font-family:var(--font)}
.grid line{stroke:var(--grid);stroke-width:1} .axis line{stroke:var(--axis);stroke-width:1}
.tick{fill:var(--muted);font-size:12px;font-variant-numeric:tabular-nums} .axlabel{fill:var(--ink-2);font-size:13px} .ptitle{fill:var(--ink);font-size:14px;font-weight:500}
.front{fill:none;stroke-width:1.6;stroke-linejoin:round} .pt{stroke:var(--surface);stroke-width:1.5} .hit{fill:transparent;cursor:pointer}
.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:13px;color:var(--ink-2);margin-top:8px} .legend span{display:inline-flex;align-items:center;gap:6px} .key{width:10px;height:10px;border-radius:50%;display:inline-block}
.tooltip{position:absolute;display:none;background:var(--tip);border:1px solid var(--ring);border-radius:6px;padding:10px 12px;font-size:13px;pointer-events:none;box-shadow:0 6px 24px rgba(0,0,0,.12);min-width:240px;z-index:2}
.tooltip .v{font-weight:600;margin-bottom:6px} .tooltip .r{display:flex;justify-content:space-between;gap:16px} .tooltip .r span{color:var(--muted)} .tooltip .r b{font-weight:500;font-variant-numeric:tabular-nums}
table{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}
th{text-align:left;font-weight:500;color:var(--ink-2);font-size:12px;letter-spacing:.04em;text-transform:uppercase;padding:8px 10px;border-bottom:1px solid var(--axis)}
td{padding:7px 10px;border-bottom:1px solid var(--grid)} td.n,th.n{text-align:right} td.rule{font-family:var(--mono);font-size:13px;white-space:nowrap}
.scroll{overflow-x:auto} .note{font-size:13px;color:var(--muted);max-width:none}
.read{display:grid;grid-template-columns:repeat(auto-fit,minmax(225px,1fr));gap:16px}
.read div{background:var(--surface);border:1px solid var(--ring);border-radius:6px;padding:16px 18px} .read h3{margin:0 0 6px;font-size:15px;font-weight:600} .read p{font-size:14px;color:var(--ink-2);max-width:none}
</style>
<main>
<header>
  <div class="eyebrow">Port-wall Patch-UF frontend · d = 7, SI1000 p = 0.003 · 100,000 paired shots · interior clusters only · coarse hardware model · non-claim-bearing</div>
  <h1>Logical error rate against L2 cycle count</h1>
  <p class="lede">The size and margin ladders against an estimated cycle count for a Micro Blossom-style hardware MWPM decoder on each rule's residual, mean and maximum over all shots in separate panels. The vertical axis is the paper's unit, logical error per patch per round, on a log scale; the right edge reads it as a multiple of Global MWPM alone. The original syndrome, with no frontend, is the black point.</p>
</header>

<div class="tiles" id="tiles"></div>
<p class="note" id="model-note"></p>

<section>
  <div class="controls" role="group" aria-label="Right panel statistic" id="stats"><span>right panel</span><button data-stat="max" aria-pressed="true">maximum</button><button data-stat="p99" aria-pressed="false">99th percentile</button></div>
  <div class="card" id="card"><div id="chart"></div>
    <div class="legend"><span><i class="key" style="background:var(--s1)"></i>size cap, interior clusters</span><span><i class="key" style="background:var(--s2)"></i>margin threshold, interior clusters</span><span><i class="key" style="background:var(--ink)"></i>original syndrome</span><span><svg width="15" height="15" viewBox="-10 -10 20 20" aria-hidden="true"><path d="M0,-9.5 L2.7,-3.6 L9,-2.9 L4.4,1.4 L5.6,7.7 L0,4.6 L-5.6,7.7 L-4.4,1.4 L-9,-2.9 L-2.7,-3.6 Z" fill="var(--goal)"/></svg>goal: MWPM's accuracy at the fewest cycles</span><span>hover a point for its numbers</span></div>
    <div class="tooltip" id="tip"></div></div>
</section>


<section>
  <h2>Every rule</h2>
  <div class="scroll"><table id="table"><thead><tr><th>syndrome given to L2</th><th class="n">coverage</th><th class="n">per patch-round, ×10⁻³</th><th class="n">× baseline</th><th class="n">CPU defects / shot</th><th class="n">max</th><th class="n">cycles / shot mean</th><th class="n">p99</th><th class="n">max</th><th class="n">rel. mean</th><th class="n">rel. max</th><th class="n">busiest round, µs mean</th><th class="n">max</th></tr></thead><tbody></tbody></table></div>
  <p class="note" id="method"></p>
</section>
</main>
<script>
const DATA = {DATA_JSON};
const fmt=(x,d=2)=>x.toLocaleString(undefined,{minimumFractionDigits:d,maximumFractionDigits:d});
const svgNS="http://www.w3.org/2000/svg";
function el(tag,attrs={},parent=null){const e=document.createElementNS(svgNS,tag);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);if(parent)parent.appendChild(e);return e;}
function txt(e,s){e.textContent=s;return e;}
function star(cx,cy,r){let d="";for(let i=0;i<10;i++){const a=-Math.PI/2+i*Math.PI/5, rr=i%2?r*0.48:r; d+=(i?"L":"M")+(cx+rr*Math.cos(a)).toFixed(1)+" "+(cy+rr*Math.sin(a)).toFixed(1);} return d+"Z";}
let stat="max";
const Y_LO=2.42, Y_HI=3.0;
function draw(){
  const host=document.getElementById("chart"); host.innerHTML="";
  const W=960,H=470,m={t:34,b:54,l:96,r:74}, gap=34; const pw=(W-m.l-m.r-gap)/2;
  const svg=el("svg",{class:"chart",viewBox:`0 0 ${W} ${H}`,role:"img","aria-label":"logical error rate per patch round against mean and "+(stat==="max"?"maximum":"99th percentile")+" L2 cycles"},host);
  const y=v=>m.t+(1-(Math.log10(v)-Math.log10(Y_LO))/(Math.log10(Y_HI)-Math.log10(Y_LO)))*(H-m.t-m.b);
  const panels=[["mean","mean cycles per shot",m.l],[stat,stat==="max"?"maximum cycles per shot":"99th-percentile cycles per shot",m.l+pw+gap]];
  const tip=document.getElementById("tip"), card=document.getElementById("card");
  for(const [st,title,x0] of panels){
    const vals=DATA.size.concat(DATA.margin).map(p=>p[st]).concat([DATA.none[st]]);
    const X_LO=Math.floor(Math.min(...vals)*0.9/500)*500, X_HI=Math.ceil(Math.max(...vals)*1.04/500)*500;
    const x=v=>x0+((v-X_LO)/(X_HI-X_LO))*pw;
    const g=el("g",{class:"grid"},svg);
    for(let t=2.5;t<=3.0+1e-9;t+=0.1){ el("line",{x1:x0,x2:x0+pw,y1:y(t),y2:y(t)},g); if(x0===m.l) txt(el("text",{x:x0-10,y:y(t)+4,"text-anchor":"end",class:"tick"},svg),fmt(t,1)+"×10⁻³"); }
    const step=(X_HI-X_LO)>4000?1000:500;
    for(let t=Math.ceil(X_LO/step)*step;t<=X_HI;t+=step){ el("line",{x1:x(t),x2:x(t),y1:m.t,y2:H-m.b},g); txt(el("text",{x:x(t),y:H-m.b+20,"text-anchor":"middle",class:"tick"},svg),(t/1000).toLocaleString(undefined,{maximumFractionDigits:1})+"k"); }
    const ax=el("g",{class:"axis"},svg); el("line",{x1:x0,x2:x0+pw,y1:H-m.b,y2:H-m.b},ax); el("line",{x1:x0,x2:x0,y1:m.t,y2:H-m.b},ax);
    txt(el("text",{x:x0+pw/2,y:m.t-12,"text-anchor":"middle",class:"ptitle"},svg),title);
    el("line",{x1:x0,x2:x0+pw,y1:y(DATA.base),y2:y(DATA.base),stroke:"var(--ink)","stroke-width":1},svg);
    if(x0===m.l) txt(el("text",{x:x0+6,y:y(DATA.base)+15,class:"tick"},svg),"Global MWPM alone  "+fmt(DATA.base)+"×10⁻³");
    const pts=[];
    for(const [rows,cls] of [[DATA.size,"s1"],[DATA.margin,"s2"]]){
      const rr=rows.slice().sort((a,b)=>a.coverage-b.coverage);
      el("path",{class:"front",style:`stroke:var(--${cls})`,d:rr.map((p,i)=>(i?"L":"M")+x(p[st])+" "+y(p.paper)).join(" ")},svg);
      for(const p of rr) pts.push([p,x(p[st]),y(p.paper),`var(--${cls})`]);
    }
    pts.push([DATA.none,x(DATA.none[st]),y(DATA.base),"var(--ink)"]);
    const gx=Math.min(...DATA.size.concat(DATA.margin).map(p=>p[st])); pts.push([{goal:true,name:"goal",x:gx},x(gx),y(DATA.base),"var(--goal)"]);
    for(const [p,cx,cy,fill] of pts){ if(p.goal) el("path",{class:"pt",d:star(cx,cy,8),fill},svg); else el("circle",{class:"pt",cx,cy,r:4.5,fill},svg);
      const h=el("circle",{class:"hit",cx,cy,r:11,tabindex:0,role:"button","aria-label":p.name},svg);
      const show=()=>{ tip.innerHTML=""; const v=document.createElement("div"); v.className="v"; v.textContent=p.name; tip.appendChild(v);
        const rows=p.goal?[["what","Global MWPM's own accuracy at the fewest cycles on this page"],[st+" cycles per shot",fmt(p.x,0)],["error rate",fmt(DATA.base)+"×10⁻³ · ×1.000"]]:[["coverage",fmt(p.coverage,1)+"%"],["per patch-round",fmt(p.paper,3)+"×10⁻³"],["multiple of MWPM alone","×"+fmt(p.mult,3)],["CPU-handled defects per shot",fmt(p.defects_mean,0)+" mean · "+p.defects_max+" max"],["cycles per shot, mean",fmt(p.mean,0)+" · "+fmt(p.rel_mean,0)+"%"],["cycles per shot, p99",fmt(p.p99,0)+" · "+fmt(p.rel_p99,0)+"%"],["cycles per shot, maximum",fmt(p.max,0)+" · "+fmt(p.rel_max,0)+"%"],["busiest round",fmt(p.us_round_mean,2)+" µs mean · "+fmt(p.us_round_max,1)+" µs max"]];
        for(const [k,val] of rows){ const r=document.createElement("div"); r.className="r"; const a=document.createElement("span"); a.textContent=k; const b=document.createElement("b"); b.textContent=val; r.append(a,b); tip.appendChild(r); }
        tip.style.display="block"; const rect=card.getBoundingClientRect(), sr=svg.getBoundingClientRect(); const px=sr.left-rect.left+(cx/W)*sr.width, py=sr.top-rect.top+(cy/H)*sr.height; tip.style.left=Math.min(px+14,rect.width-260)+"px"; tip.style.top=Math.max(8,py-10)+"px"; };
      h.addEventListener("pointerenter",show); h.addEventListener("focus",show); h.addEventListener("pointerleave",()=>tip.style.display="none"); h.addEventListener("blur",()=>tip.style.display="none"); }
  }
  const xr=m.l+pw+gap+pw; const ax2=el("g",{class:"axis"},svg); el("line",{x1:xr,x2:xr,y1:m.t,y2:H-m.b},ax2);
  for(const r of [1.0,1.02,1.05,1.1,1.15,1.2]){ const v=DATA.base*r; if(v<Y_LO||v>Y_HI) continue; el("line",{x1:xr,x2:xr+5,y1:y(v),y2:y(v)},ax2); txt(el("text",{x:xr+9,y:y(v)+4,class:"tick"},svg),"×"+fmt(r,2)); }
  txt(el("text",{x:18,y:(m.t+H-m.b)/2,"text-anchor":"middle",class:"axlabel",transform:`rotate(-90 18 ${(m.t+H-m.b)/2})`},svg),"logical error per patch per round");
  txt(el("text",{x:W-16,y:(m.t+H-m.b)/2,"text-anchor":"middle",class:"axlabel",transform:`rotate(90 ${W-16} ${(m.t+H-m.b)/2})`},svg),"multiple of Global MWPM alone");
  txt(el("text",{x:m.l+(W-m.l-m.r)/2,y:H-12,"text-anchor":"middle",class:"axlabel"},svg),"coarse Micro Blossom accelerator cycles per shot ("+DATA.cell.rounds+" rounds, "+DATA.cell.patches+" patches)");
}
for(const b of document.querySelectorAll("#stats button")) b.addEventListener("click",()=>{ stat=b.dataset.stat; for(const o of document.querySelectorAll("#stats button")) o.setAttribute("aria-pressed",o===b?"true":"false"); draw(); });
draw();

(function(){
  const M=Object.fromEntries(DATA.margin.map(p=>[p.value,p])), S=Object.fromEntries(DATA.size.map(p=>[p.value===null?"any":p.value,p]));
  const m15=M[1.5], m10=M[1], m05=M[0.5], s2=S[2], s4=S[4], sA=S["any"], n=DATA.none, all=DATA.all, md=DATA.model;
  const tiles=[
    ["Original syndrome, cycles per shot", fmt(n.mean,0)+" mean", "p99 "+fmt(n.p99,0)+" · maximum "+fmt(n.max,0)+" · "+fmt(n.defects_mean,0)+" CPU-handled defects per shot"],
    ["Resolved inside the accelerator for free", fmt(100*md.isolated_defect_share,0)+"% of defects", "isolated pairs and lone wall defects · the same set the size ≤ 2 rule commits"],
    ["Size ≤ 4, ×"+fmt(s4.mult,3), fmt(s4.rel_mean,0)+"% mean", "maximum "+fmt(s4.rel_max,0)+"% · "+fmt(s4.coverage,0)+"% coverage"],
    ["Interior, any size, ×"+fmt(sA.mult,3), fmt(sA.rel_mean,0)+"% mean", "maximum "+fmt(sA.rel_max,0)+"% · "+fmt(sA.coverage,0)+"% coverage"]];
  const root=document.getElementById("tiles");
  for(const [l,v,sub] of tiles){ const d=document.createElement("div"); d.className="tile"; const a=document.createElement("div"); a.className="label"; a.textContent=l; const b=document.createElement("div"); b.className="value"; b.textContent=v; const c=document.createElement("div"); c.className="sub"; c.textContent=sub; d.append(a,b,c); root.appendChild(d); }
  document.getElementById("model-note").textContent=`The cycle model, from Micro Blossom (Wu, Liyanage, Zhong, ASPLOS 2025): a defect whose growth region meets its partner, or the real boundary, before touching anything else is an isolated conflict and is resolved inside the accelerator in constant time; every other defect costs one CPU interaction of ${md.interaction_ns} ns plus a ${md.round_cycles}-cycle grow round, at a ${md.accelerator_mhz} MHz accelerator clock, so ${fmt(md.cycles_per_cpu_defect,1)} cycles each, on top of a ${md.io_floor_us} µs I/O floor. Which defects are isolated comes from the frontend's own cluster records: interior size-2 clusters and real-wall singletons with positive margin. Blossom formation, clusters that need several interactions, and the yoke hub vertices are ignored, so large clusters are undercounted. Coarse by construction.`;
  const tb=document.querySelector("#table tbody");
  for(const p of [n].concat(DATA.size,DATA.margin,[all])){ const tr=document.createElement("tr");
    for(const [v,cls] of [[p.name,"rule"],[fmt(p.coverage,1)+"%","n"],[fmt(p.paper,3),"n"],[fmt(p.mult,3),"n"],[fmt(p.defects_mean,1),"n"],[String(p.defects_max),"n"],[fmt(p.mean,0),"n"],[fmt(p.p99,0),"n"],[fmt(p.max,0),"n"],[fmt(p.rel_mean,0)+"%","n"],[fmt(p.rel_max,0)+"%","n"],[fmt(p.us_round_mean,2),"n"],[fmt(p.us_round_max,1),"n"]]){ const td=document.createElement("td"); td.className=cls; td.textContent=v; tr.appendChild(td);} tb.appendChild(tr); }
  document.getElementById("method").textContent=`Method: the union-find run does not depend on the commit rule, so each rule's residual is the set of clusters it leaves, and the cycle count is computed from the retained cluster records of ${DATA.shots.toLocaleString()} shots without re-decoding. "Busiest round" attributes each CPU-handled defect to its measurement round and takes the round with the most in each shot, the quantity a round-wise streaming decoder would have to finish within its budget. Error rates are the paper's unit: sinter's shot-to-piece conversion of the per-shot block failure with ${DATA.pieces} patch-rounds and ${DATA.observables} observables. Source: ${DATA.model.source}.`;
})();
</script>
"""
out_html.write_text(html.replace("{DATA_JSON}", json.dumps(data))); print("wrote", out_html)
