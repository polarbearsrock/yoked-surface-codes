"""Interactive presentation page: coverage ladder, three L2 cost views (two panels each), and the cross-model summary."""
import json, sys
from pathlib import Path
import numpy as np, sinter

cell = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("out/cluster_size_study_d7/d7_p0.003_100k")
out_html = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("out/cluster_size_study_d7/talk_figures.html")
fr = json.load(open(cell / "frontier_interior.json")); prov = json.load(open(cell / "provenance.json"))
T = {f"{r['family']}:{r['value']}": r for r in json.load(open(cell / "l2_timing.json"))["rows"]}
MB = json.load(open(cell / "microblossom_cycles.json")); M = {f"{r['family']}:{r['value']}": r for r in MB["rows"]}
ZG = json.load(open(cell / "zerog_cycles.json")); Z = {f"{r['family']}:{r['value']}": r for r in ZG["rows"]}
c = prov["cell"]; N, O = fr["shots"], fr["num_observables"]; PIECES = c["patches"] * c["rounds"]
conv = lambda p: sinter.shot_error_rate_to_piece_error_rate(p, pieces=PIECES, values=O)
gbase = fr["global_failures"] / N; BASE = conv(gbase)
z = np.load(cell / "cell.npz"); ceiling = 100 * float(z["component_size"][z["component_committed"] & ~z["component_boundary"]].sum()) / float(z["component_size"].sum())
def name(f):
    if f["family"] == "size": return "every interior cluster" if f["value"] is None else f"size ≤ {f['value']}"
    return "every interior cluster" if f["value"] == 0 else f"margin > {f['value']:g}"
rows = {"size": [], "margin": []}
for f in fr["rows"]:
    fam = f["family"]
    if fam not in rows or f["coverage_pct"] <= 0 or (fam == "size" and f["value"] is not None and f["value"] % 2): continue
    k = f"{fam}:{f['value']}"; p = f["failures"] / N; paper = conv(p)
    r = dict(key=k, family=fam, value=f["value"], name=name(f), coverage=f["coverage_pct"], paper=1e3 * paper, mult=paper / BASE, rd=f["risk_difference_pp"], lo=f["ci_pp"][0], hi=f["ci_pp"][1],
             t=None, mb=None, zg=None)
    if k in T: t = T[k]; r["t"] = dict(mean=100 * t["rel_mean"], p99=100 * t["rel_p99"], max=100 * t["rel_max"], mean_ms=t["mean_ms"], p99_ms=t["p99_ms"], max_ms=t["max_ms"], events=t["mean_events"])
    if k in M: m = M[k]; r["mb"] = dict(mean=m["cyc_shot_mean"], p99=m["cyc_shot_p99"], max=m["cyc_shot_max"], rel_mean=100 * m["rel_mean"], rel_p99=100 * m["rel_p99"], rel_max=100 * m["rel_max"], defects=m["noniso_mean"], defects_max=m["noniso_max"], us_mean=m["us_round_mean"], us_max=m["us_round_max"])
    if k in Z: g = Z[k]; r["zg"] = dict(mean=g["cyc_mean"], p99=g["cyc_p99"], max=g["cyc_max"], rel_mean=100 * g["rel_mean"], rel_p99=100 * g["rel_p99"], rel_max=100 * g["rel_max"], n=g["n_mean"], n_max=g["n_max"], cap=g["within_capacity_pct"], us_mean=g["us_mean"], us_max=g["us_max"])
    rows[fam].append(r)
for fam in rows: rows[fam].sort(key=lambda r: r["coverage"])
tn, mn, zn = T["none:None"], M["none:None"], Z["none:None"]
none = dict(name="untouched syndrome", coverage=0.0, paper=1e3 * BASE, mult=1.0, t=dict(mean=100, p99=100, max=100, mean_ms=tn["mean_ms"], p99_ms=tn["p99_ms"], max_ms=tn["max_ms"], events=tn["mean_events"]),
            mb=dict(mean=mn["cyc_shot_mean"], p99=mn["cyc_shot_p99"], max=mn["cyc_shot_max"], rel_mean=100, rel_p99=100, rel_max=100, defects=mn["noniso_mean"], defects_max=mn["noniso_max"], us_mean=mn["us_round_mean"], us_max=mn["us_round_max"]),
            zg=dict(mean=zn["cyc_mean"], p99=zn["cyc_p99"], max=zn["cyc_max"], rel_mean=100, rel_p99=100, rel_max=100, n=zn["n_mean"], n_max=zn["n_max"], cap=zn["within_capacity_pct"], us_mean=zn["us_mean"], us_max=zn["us_max"]))
summary = [dict(rule="margin > 1.5", mult=next(r["mult"] for r in rows["margin"] if r["value"] == 1.5), models=[("software sparse blossom (measured time)", T["margin:1.5"]), ("Micro Blossom style (coarse cycles)", M["margin:1.5"]), ("Zero-G style (coarse cycles)", Z["margin:1.5"])]),
           dict(rule="margin > 1.0", mult=next(r["mult"] for r in rows["margin"] if r["value"] == 1.0), models=[("software sparse blossom (measured time)", T["margin:1.0"]), ("Micro Blossom style (coarse cycles)", M["margin:1.0"]), ("Zero-G style (coarse cycles)", Z["margin:1.0"])])]
for s in summary: s["models"] = [dict(name=n, mean=100 * r["rel_mean"], p99=100 * r["rel_p99"], max=100 * r["rel_max"]) for n, r in s["models"]]
cap = None
if (cell / "cap_ladder.json").exists():
    CL = json.load(open(cell / "cap_ladder.json")); HB = json.load(open(cell / "helios_budget_cycles.json"))
    cyc = {("inf" if r["cap"] is None else str(r["cap"])): dict(mean=r["mean"], p50=r["p50"], p99=r["p99"], max=r["max"]) for r in HB["rows"] if r["hop"] == 1}
    cap = dict(quantum=CL["quantum"], cycles=cyc, fixed=HB["model"]["fixed_cycles_per_iteration"], full=cyc["inf"], series={})
    for r in CL["rows"]:
        p = r["failures"] / N; paper = conv(p)
        cap["series"].setdefault(r["series"], dict(name=r["series_name"], rows=[]))["rows"].append(dict(cap=r["cap"], cycles=r["cycles_est"], coverage=r["coverage_pct"], paper=1e3 * paper, mult=paper / BASE, rd=r["risk_difference_pp"], lo=r["ci_pp"][0], hi=r["ci_pp"][1],
            residual=r["residual_events_per_shot"], closed=r["closed_share_pct"]))
allp = [r["paper"] for fam in rows for r in rows[fam]] + ([r["paper"] for sr in cap["series"].values() for r in sr["rows"]] if cap else [])
y_lo = 1e3 * BASE * 0.975; y_hi = max(1e3 * BASE * 1.21, max(allp) * 1.02)
y_step = next(st for st in (0.02, 0.05, 0.1, 0.2, 0.5, 1.0) if (y_hi - y_lo) / st <= 10)
title = "09/03/26 Meeting DI" if c["d"] == 7 else f"Union-Find + Joint MWPM, d = {c['d']}"
data = dict(shots=N, cell=c, base=1e3 * BASE, y_lo=y_lo, y_hi=y_hi, y_step=y_step, ceiling=ceiling, rows=rows, none=none, summary=summary, mb_model=MB["model"], zg_model=ZG["model"], zg_windows=ZG["window_rounds"], windows_per_shot=ZG["windows_per_shot"], cap=cap, residual_none=T["none:None"]["mean_events"])
foot = f"d = {c['d']}, {c['patches']} patches + {c['yokes']} yokes, {c['rounds']} rounds, SI1000 p = {c['p']}, {N:,} paired shots · vertical axis: logical error per patch round (paper unit), log scale · star: MWPM's accuracy at the best x"

html = r"""<title>{TITLE}</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{color-scheme:light;--page:#f7f7f4;--surface:#fcfcfb;--ink:#0b0b0b;--ink-2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);
  --s1:#2a78d6;--s2:#eb6834;--goal:#16a34a;--bar-mean:#9ed9cf;--bar-max:#0f766e;--accent:#2a78d6;--tip:#ffffff;--font:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#ffffff;--ink-2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--goal:#4ade80;--bar-mean:#2f6b64;--bar-max:#2dd4bf;--accent:#3987e5;--tip:#242423}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#ffffff;--ink-2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--goal:#4ade80;--bar-mean:#2f6b64;--bar-max:#2dd4bf;--accent:#3987e5;--tip:#242423}
body{margin:0;background:var(--page);color:var(--ink);font-family:var(--font);font-size:15px;line-height:1.5}
main{max-width:1120px;margin:0 auto;padding:40px 24px 64px;display:flex;flex-direction:column;gap:40px}
h1{font-size:30px;font-weight:600;letter-spacing:-.01em;margin:0 0 8px} h2{font-size:19px;font-weight:600;margin:0}
.eyebrow{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin-bottom:12px;font-weight:500}
p{max-width:72ch;margin:0;color:var(--ink-2)}
section{display:flex;flex-direction:column;gap:12px}
.head{display:flex;justify-content:space-between;align-items:baseline;gap:16px;flex-wrap:wrap}
.controls{display:flex;gap:8px;align-items:center;font-size:13px;color:var(--ink-2)}
.controls button{font:inherit;background:var(--surface);color:var(--ink);border:1px solid var(--ring);border-radius:4px;padding:3px 10px;cursor:pointer}
.controls button[aria-pressed="true"]{background:var(--ink);color:var(--surface);border-color:var(--ink)}
.controls button:focus-visible,.hit:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:6px;padding:18px 18px 10px;position:relative}
svg.chart{display:block;width:100%;height:auto;font-family:var(--font)}
.grid line{stroke:var(--grid);stroke-width:1} .axis line{stroke:var(--axis);stroke-width:1}
.tick{fill:var(--muted);font-size:12.5px;font-variant-numeric:tabular-nums} .axlabel{fill:var(--ink-2);font-size:13.5px} .ptitle{fill:var(--ink);font-size:14.5px;font-weight:500} .dlabel{font-size:13px}
.front{fill:none;stroke-width:1.7;stroke-linejoin:round} .pt{stroke:var(--surface);stroke-width:1.5} .hit{fill:transparent;cursor:pointer}
.legend{display:flex;gap:20px;flex-wrap:wrap;font-size:13.5px;color:var(--ink-2);margin:10px 0 4px;justify-content:center} .legend span{display:inline-flex;align-items:center;gap:7px} .key{width:11px;height:11px;border-radius:50%;display:inline-block} .band{width:22px;height:10px;background:var(--goal);opacity:.18;display:inline-block;border-radius:2px}
.foot{font-size:12.5px;color:var(--muted);text-align:center}
.tooltip{position:absolute;display:none;background:var(--tip);border:1px solid var(--ring);border-radius:6px;padding:10px 12px;font-size:13px;pointer-events:none;box-shadow:0 6px 24px rgba(0,0,0,.12);min-width:240px;z-index:2}
.tooltip .v{font-weight:600;margin-bottom:6px} .tooltip .r{display:flex;justify-content:space-between;gap:16px} .tooltip .r span{color:var(--muted)} .tooltip .r b{font-weight:500;font-variant-numeric:tabular-nums}
</style>
<main>
<header><div class="eyebrow">{EYEBROW}</div>
<h1>Union-Find + Joint MWPM</h1></header>

<section><div class="head"><h2>1 · Accuracy cost of the L1 frontend against coverage</h2></div>
  <div class="card"><div id="c1"></div><div class="legend" id="l1"></div><div class="tooltip" id="t1"></div></div></section>
<section><div class="head"><h2>2 · Software sparse blossom (PyMatching): measured decode time of the residual</h2><div class="controls" id="k2"><span>right panel</span><button data-stat="p99" aria-pressed="true">99th percentile</button><button data-stat="max" aria-pressed="false">slowest shot</button></div></div>
  <div class="card"><div id="c2"></div><div class="legend" id="l2"></div><div class="tooltip" id="t2"></div></div></section>
<section><div class="head"><h2>3 · Micro Blossom-style accelerator: coarse cycles per shot</h2><div class="controls" id="k3"><span>right panel</span><button data-stat="p99" aria-pressed="true">99th percentile</button><button data-stat="max" aria-pressed="false">slowest shot</button></div></div>
  <div class="card"><div id="c3"></div><div class="legend" id="l3"></div><div class="tooltip" id="t3"></div></div></section>
<section><div class="head"><h2>4 · Zero-G-style decoder: coarse cycles per decoding window</h2><div class="controls" id="k4"><span>right panel</span><button data-stat="p99" aria-pressed="true">99th percentile</button><button data-stat="max" aria-pressed="false">slowest window</button></div></div>
  <div class="card"><div id="c4"></div><div class="legend" id="l4"></div><div class="tooltip" id="t4"></div></div></section>
<section><div class="head"><h2>5 · L2 work reduction within an exploratory 1% accuracy budget</h2></div>
  <div class="card"><div id="c5"></div><div class="legend" id="l5"></div><div class="tooltip" id="t5"></div></div></section>
<section id="s6" hidden><div class="head"><h2>6 · Accuracy versus capped Helios-style UF growth-depth proxy</h2><div class="controls" id="k6"><span>right panel</span><button data-stat="p99" aria-pressed="true">99th percentile</button><button data-stat="max" aria-pressed="false">slowest shot</button></div></div>
  <div class="card"><div id="c6"></div><div class="legend" id="l6"></div><div class="tooltip" id="t6"></div></div></section>
<p class="foot">{FOOT} · figure 2 is measured; figures 3 and 4 are coarse cycle models with constants borrowed from the Micro Blossom and Zero-G papers · files: out/cluster_size_study_d7/figures/presentation/</p>
</main>
<script>
const DATA = {DATA_JSON};
const fmt=(x,d=2)=>x.toLocaleString(undefined,{minimumFractionDigits:d,maximumFractionDigits:d});
const svgNS="http://www.w3.org/2000/svg";
function el(tag,attrs={},parent=null){const e=document.createElementNS(svgNS,tag);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);if(parent)parent.appendChild(e);return e;}
function txt(e,s){e.textContent=s;return e;}
function star(cx,cy,r){let d="";for(let i=0;i<10;i++){const a=-Math.PI/2+i*Math.PI/5, rr=i%2?r*0.48:r; d+=(i?"L":"M")+(cx+rr*Math.cos(a)).toFixed(1)+" "+(cy+rr*Math.sin(a)).toFixed(1);} return d+"Z";}
function tri(cx,cy,r){return `M${cx-r} ${cy-r*0.7}L${cx+r} ${cy-r*0.7}L${cx} ${cy+r}Z`;}
const Y_LO=DATA.y_lo, Y_HI=DATA.y_hi, Y_STEP=DATA.y_step, YDEC=Y_STEP>=0.1?1:2, BASE=DATA.base;
const MULTS=[1.0,1.02,1.05,1.1,1.15,1.2,1.3,1.4,1.5,1.75,2.0];
function yTicks(){ const out=[]; for(let t=Math.ceil(Y_LO/Y_STEP-1e-9)*Y_STEP;t<=Y_HI+1e-9;t+=Y_STEP) out.push(+t.toFixed(3)); return out; }
function linTicks(lo,hi){ const span=hi-lo; const step=[100,200,500,1000,2000,5000,10000,20000,50000].find(s=>span/s<=8)||100000; const out=[]; for(let t=Math.ceil(lo/step)*step;t<=hi+1e-9;t+=step) out.push(t); return out; }
function logTicks(lo,hi){ const out=[]; for(let e=Math.floor(Math.log10(lo));e<=Math.ceil(Math.log10(hi));e++) for(const m of [1,2,5]){ const t=m*10**e; if(t>=lo&&t<=hi) out.push(t);} return out; }
function pow2Ticks(lo,hi){ const out=[]; for(let t=4;t<=hi;t*=2) if(t>=lo) out.push(t); return out; }
const SHOTS=DATA.shots.toLocaleString(), WINDOWS=(DATA.shots*DATA.windows_per_shot).toLocaleString(), R=DATA.cell.rounds;
function attachTip(h,tip,card,svg,W,H,cx,cy,title,rows){
  const show=()=>{ tip.innerHTML=""; const v=document.createElement("div"); v.className="v"; v.textContent=title; tip.appendChild(v);
    for(const [k,val] of rows){ const r=document.createElement("div"); r.className="r"; const a=document.createElement("span"); a.textContent=k; const b=document.createElement("b"); b.textContent=val; r.append(a,b); tip.appendChild(r); }
    tip.style.display="block"; const rect=card.getBoundingClientRect(), sr=svg.getBoundingClientRect(); const px=sr.left-rect.left+(cx/W)*sr.width, py=sr.top-rect.top+(cy/H)*sr.height; tip.style.left=Math.min(px+14,rect.width-260)+"px"; tip.style.top=Math.max(8,py-10)+"px"; };
  h.addEventListener("pointerenter",show); h.addEventListener("focus",show); h.addEventListener("pointerleave",()=>tip.style.display="none"); h.addEventListener("blur",()=>tip.style.display="none");
}
function legend(id, extra){
  const L=document.getElementById(id); L.innerHTML="";
  const items=[['<i class="key" style="background:var(--s1)"></i>','size cap, interior clusters'],['<i class="key" style="background:var(--s2)"></i>','margin threshold, interior clusters'],['<i class="key" style="background:var(--ink)"></i>','untouched syndrome'],
    ['<svg width="15" height="15" viewBox="-10 -10 20 20"><path d="M0,-9.5 L2.7,-3.6 L9,-2.9 L4.4,1.4 L5.6,7.7 L0,4.6 L-5.6,7.7 L-4.4,1.4 L-9,-2.9 L-2.7,-3.6 Z" fill="var(--goal)"/></svg>','goal'],['<i class="band"></i>','within 1% of MWPM alone (free region)']].concat(extra||[]);
  for(const [h,t] of items){ const s=document.createElement("span"); s.innerHTML=h+t; L.appendChild(s); }
}
/* shared: a panel with the paper-unit y-axis, band, baseline, ladders, untouched point, goal star, two labels, tooltips */
function panel(svg,x0,pw,m,H,W,cfg,tip,card,first,last){
  const y=v=>m.t+(1-(Math.log10(v)-Math.log10(Y_LO))/(Math.log10(Y_HI)-Math.log10(Y_LO)))*(H-m.t-m.b);
  const x=cfg.xscale==="log"?(v=>x0+((Math.log10(v)-Math.log10(cfg.xlim[0]))/(Math.log10(cfg.xlim[1])-Math.log10(cfg.xlim[0])))*pw):(v=>x0+((v-cfg.xlim[0])/(cfg.xlim[1]-cfg.xlim[0]))*pw);
  const g=el("g",{class:"grid"},svg);
  for(const t of yTicks()){ el("line",{x1:x0,x2:x0+pw,y1:y(t),y2:y(t)},g); if(first) txt(el("text",{x:x0-10,y:y(t)+4,"text-anchor":"end",class:"tick"},svg),fmt(t,YDEC)+"×10⁻³"); }
  for(const t of cfg.xticks){ el("line",{x1:x(t),x2:x(t),y1:m.t,y2:H-m.b},g); txt(el("text",{x:x(t),y:H-m.b+20,"text-anchor":"middle",class:"tick"},svg),cfg.xfmt(t)); }
  el("rect",{x:x0,y:y(BASE*1.01),width:pw,height:y(BASE)-y(BASE*1.01),fill:"var(--goal)",opacity:0.14},svg);
  const ax=el("g",{class:"axis"},svg); el("line",{x1:x0,x2:x0+pw,y1:H-m.b,y2:H-m.b},ax); el("line",{x1:x0,x2:x0,y1:m.t,y2:H-m.b},ax);
  el("line",{x1:x0,x2:x0+pw,y1:y(BASE),y2:y(BASE),stroke:"var(--ink)","stroke-width":1.1},svg);
  if(cfg.title) txt(el("text",{x:x0+pw/2,y:m.t-12,"text-anchor":"middle",class:"ptitle"},svg),cfg.title);
  const pts=[];
  for(const fam of ["size","margin"]){ const rr=DATA.rows[fam].filter(cfg.has); const col=fam==="size"?"var(--s1)":"var(--s2)";
    el("path",{class:"front",style:`stroke:${col}`,d:rr.map((p,i)=>(i?"L":"M")+x(cfg.xv(p))+" "+y(p.paper)).join(" ")},svg);
    for(const p of rr){ const cx=x(cfg.xv(p)), cy=y(p.paper); if(fam==="size") el("circle",{class:"pt",cx,cy,r:5,fill:col},svg); else el("circle",{class:"pt",cx,cy,r:5,fill:col},svg); pts.push([p,cx,cy]); } }
  const nx=x(cfg.xv(DATA.none)); el("circle",{class:"pt",cx:nx,cy:y(BASE),r:5.5,fill:"var(--ink)"},svg); pts.push([DATA.none,nx,y(BASE)]);
  const gx=cfg.goal!=null?cfg.goal:Math.min(...DATA.rows.size.concat(DATA.rows.margin).filter(cfg.has).map(cfg.xv)); el("path",{class:"pt",d:star(x(gx),y(BASE),9),fill:"var(--goal)"},svg);
  if(cfg.ceiling!=null){ el("line",{x1:x(cfg.ceiling),x2:x(cfg.ceiling),y1:m.t,y2:H-m.b,stroke:"var(--axis)","stroke-width":1,"stroke-dasharray":"4 4"},svg); txt(el("text",{x:x(cfg.ceiling)+7,y:m.t+14,class:"tick"},svg),"interior-only ceiling "+fmt(cfg.ceiling,0)+"%"); }
  for(const [p,cx,cy] of pts){ const h=el("circle",{class:"hit",cx,cy,r:11,tabindex:0,role:"button","aria-label":p.name},svg); attachTip(h,tip,card,svg,W,H,cx,cy,p.name,cfg.rows(p)); }
  const hg=el("circle",{class:"hit",cx:x(gx),cy:y(BASE),r:12,tabindex:0,role:"button","aria-label":"goal"},svg); attachTip(hg,tip,card,svg,W,H,x(gx),y(BASE),"goal",[["what","MWPM's own accuracy at the best x on this chart"],[cfg.xname,cfg.xfmt(gx)],["error rate",fmt(BASE)+"×10⁻³ · ×1.000"]]);
  if(last){ const xr=x0+pw; const a2=el("g",{class:"axis"},svg); el("line",{x1:xr,x2:xr,y1:m.t,y2:H-m.b},a2);
    for(const r of MULTS){ const v=BASE*r; if(v<Y_LO||v>Y_HI) continue; el("line",{x1:xr,x2:xr+5,y1:y(v),y2:y(v)},a2); txt(el("text",{x:xr+9,y:y(v)+4,class:"tick"},svg),"×"+fmt(r,2)); } }
}
function two(id,tipId,cfgs,xlabel){
  const host=document.getElementById(id); host.innerHTML=""; const W=1040,H=470,m={t:34,b:54,l:98,r:78},gap=34; const pw=(W-m.l-m.r-gap)/2;
  const svg=el("svg",{class:"chart",viewBox:`0 0 ${W} ${H}`,role:"img"},host); const tip=document.getElementById(tipId), card=host.parentElement;
  panel(svg,m.l,pw,m,H,W,cfgs[0],tip,card,true,false); panel(svg,m.l+pw+gap,pw,m,H,W,cfgs[1],tip,card,false,true);
  txt(el("text",{x:18,y:(m.t+H-m.b)/2,"text-anchor":"middle",class:"axlabel",transform:`rotate(-90 18 ${(m.t+H-m.b)/2})`},svg),"logical error rate per patch round");
  txt(el("text",{x:W-14,y:(m.t+H-m.b)/2,"text-anchor":"middle",class:"axlabel",transform:`rotate(90 ${W-14} ${(m.t+H-m.b)/2})`},svg),"multiple of Global MWPM alone");
  txt(el("text",{x:m.l+(W-m.l-m.r)/2,y:H-12,"text-anchor":"middle",class:"axlabel"},svg),xlabel);
}
function one(id,tipId,cfg,xlabel){
  const host=document.getElementById(id); host.innerHTML=""; const W=1040,H=470,m={t:34,b:54,l:98,r:78}; const pw=W-m.l-m.r;
  const svg=el("svg",{class:"chart",viewBox:`0 0 ${W} ${H}`,role:"img"},host); const tip=document.getElementById(tipId), card=host.parentElement;
  panel(svg,m.l,pw,m,H,W,cfg,tip,card,true,true);
  txt(el("text",{x:18,y:(m.t+H-m.b)/2,"text-anchor":"middle",class:"axlabel",transform:`rotate(-90 18 ${(m.t+H-m.b)/2})`},svg),"logical error rate per patch round");
  txt(el("text",{x:W-14,y:(m.t+H-m.b)/2,"text-anchor":"middle",class:"axlabel",transform:`rotate(90 ${W-14} ${(m.t+H-m.b)/2})`},svg),"multiple of Global MWPM alone");
  txt(el("text",{x:m.l+pw/2,y:H-12,"text-anchor":"middle",class:"axlabel"},svg),xlabel);
}
const acc=p=>[["coverage",fmt(p.coverage,1)+"%"],["per patch-round",fmt(p.paper,3)+"×10⁻³"],["multiple of MWPM alone","×"+fmt(p.mult,3)]];
const pct=v=>fmt(v,0)+"%", kfmt=v=>(v/1000).toLocaleString(undefined,{maximumFractionDigits:1})+"k";
/* 1 coverage */
one("c1","t1",{has:p=>true,xv:p=>p.coverage,xscale:"linear",xlim:[0,100],xticks:[0,10,20,30,40,50,60,70,80,90,100],xfmt:pct,xname:"coverage",goal:DATA.ceiling,ceiling:DATA.ceiling,
  rows:p=>acc(p).concat(p.rd!=null?[["regression vs MWPM alone",(p.rd>=0?"+":"")+fmt(p.rd)+" pp"],["paired 95% CI",fmt(p.lo)+" to "+fmt(p.hi)+" pp"]]:[])},"coverage: share of detector events removed by the L1 frontend before Global MWPM");
legend("l1");
/* 2-4 two-panel views */
const stats={c2:"p99",c3:"p99",c4:"p99"};
function draw2(){ const st=stats.c2, tr=p=>acc(p).concat([["mean time",fmt(p.t.mean_ms,3)+" ms · "+pct(p.t.mean)],["p99",fmt(p.t.p99_ms,3)+" ms · "+pct(p.t.p99)],["slowest shot",fmt(p.t.max_ms,2)+" ms · "+pct(p.t.max)],["events per shot",fmt(p.t.events,0)]]);
  const base={has:p=>!!p.t,xscale:"linear",xlim:[35,105],xticks:[40,50,60,70,80,90,100],xfmt:pct,rows:tr};
  two("c2","t2",[{...base,title:"mean over "+SHOTS+" shots",xv:p=>p.t.mean,xname:"mean time"},{...base,title:st==="max"?"slowest shot":"99th-percentile shot",xv:p=>p.t[st],xname:(st==="max"?"slowest-shot":"p99")+" time"}],"Global MWPM decode time on the residual, % of the untouched syndrome"); legend("l2"); }
function draw3(){ const st=stats.c3, tr=p=>acc(p).concat([["CPU-handled defects per shot",fmt(p.mb.defects,0)+" mean · "+p.mb.defects_max+" max"],["cycles, mean",fmt(p.mb.mean,0)+" · "+pct(p.mb.rel_mean)],["cycles, p99",fmt(p.mb.p99,0)+" · "+pct(p.mb.rel_p99)],["cycles, slowest shot",fmt(p.mb.max,0)+" · "+pct(p.mb.rel_max)],["busiest round",fmt(p.mb.us_mean,1)+" µs mean · "+fmt(p.mb.us_max,0)+" µs max"]]);
  const mbv=DATA.rows.size.concat(DATA.rows.margin).filter(p=>p.mb).map(p=>p.mb.mean).concat([DATA.none.mb.mean,DATA.none.mb[st]]); const mlo=Math.floor(Math.min(...mbv)*0.85/500)*500, mhi=Math.ceil(Math.max(...mbv)*1.05/500)*500;
  const base={has:p=>!!p.mb,xscale:"linear",xlim:[mlo,mhi],xticks:linTicks(mlo,mhi),xfmt:kfmt,rows:tr};
  two("c3","t3",[{...base,title:"mean over "+SHOTS+" shots",xv:p=>p.mb.mean,xname:"mean cycles"},{...base,title:st==="max"?"slowest shot":"99th-percentile shot",xv:p=>p.mb[st],xname:(st==="max"?"slowest-shot":"p99")+" cycles"}],"estimated accelerator cycles per "+R+"-round shot, 62 MHz · isolated pairs resolved in hardware"); legend("l3"); }
function draw4(){ const st=stats.c4, tr=p=>acc(p).concat([["active detectors per window",fmt(p.zg.n,0)+" mean · "+p.zg.n_max+" max"],["windows within 128 capacity",fmt(p.zg.cap,1)+"%"],["cycles, mean",fmt(p.zg.mean,0)+" · "+pct(p.zg.rel_mean)+" · "+fmt(p.zg.us_mean,1)+" µs"],["cycles, p99",fmt(p.zg.p99,0)+" · "+pct(p.zg.rel_p99)],["cycles, slowest window",fmt(p.zg.max,0)+" · "+pct(p.zg.rel_max)+" · "+fmt(p.zg.us_max,0)+" µs"]]);
  const zgv=DATA.rows.size.concat(DATA.rows.margin).filter(p=>p.zg).map(p=>p.zg.mean).concat([DATA.none.zg.max]); const zlo=10**Math.floor(Math.log10(Math.min(...zgv)*0.8)), zhi=10**Math.ceil(Math.log10(Math.max(...zgv)*1.1));
  const base={has:p=>!!p.zg,xscale:"log",xlim:[zlo,zhi],xticks:logTicks(zlo,zhi),xfmt:kfmt,rows:tr};
  two("c4","t4",[{...base,title:"mean over "+WINDOWS+" windows",xv:p=>p.zg.mean,xname:"mean cycles"},{...base,title:st==="max"?"slowest window":"99th-percentile window",xv:p=>p.zg[st],xname:(st==="max"?"slowest-window":"p99")+" cycles"}],"estimated cycles per "+DATA.zg_windows+"-round window of the whole block, 250 MHz, log scale · cost counts active detectors"); legend("l4"); }
draw2(); draw3(); draw4();
for(const [kid,cid,fn] of [["k2","c2",draw2],["k3","c3",draw3],["k4","c4",draw4]]) for(const b of document.querySelectorAll("#"+kid+" button")) b.addEventListener("click",()=>{ stats[cid]=b.dataset.stat; for(const o of document.querySelectorAll("#"+kid+" button")) o.setAttribute("aria-pressed",o===b?"true":"false"); fn(); });
/* 6 growth-time cap ladder: x = truncated Helios proxy cycles at cap K, mean | slowest shot */
const stat6={v:"p99"};
function draw6(){ const C=DATA.cap; if(!C) return; document.getElementById("s6").hidden=false;
  const host=document.getElementById("c6"); host.innerHTML=""; const W=1040,H=470,m={t:34,b:54,l:98,r:78},gap=34; const pw=(W-m.l-m.r-gap)/2; const svg=el("svg",{class:"chart",viewBox:`0 0 ${W} ${H}`,role:"img"},host); const tip=document.getElementById("t6"), card=host.parentElement;
  const y=v=>m.t+(1-(Math.log10(v)-Math.log10(Y_LO))/(Math.log10(Y_HI)-Math.log10(Y_LO)))*(H-m.t-m.b);
  const cols={cap:"#7c3aed",cap_m05:"#f59e0b",cap_m10:"var(--s2)"}; const order=["cap","cap_m05","cap_m10"]; const cy=k=>C.cycles[k==null?"inf":String(k)];
  const cmax=Math.max(...Object.values(C.cycles).map(v=>v.max)); const X_LO=3, X_HI=10**Math.ceil(Math.log10(cmax*1.1));
  const panels=[["mean","mean over "+SHOTS+" shots",m.l],[stat6.v,stat6.v==="max"?"slowest shot":"99th-percentile shot",m.l+pw+gap]];
  for(const [st,title,x0] of panels){ const x=v=>x0+((Math.log10(v)-Math.log10(X_LO))/(Math.log10(X_HI)-Math.log10(X_LO)))*pw;
    const g=el("g",{class:"grid"},svg);
    for(const t of yTicks()){ el("line",{x1:x0,x2:x0+pw,y1:y(t),y2:y(t)},g); if(x0===m.l) txt(el("text",{x:x0-10,y:y(t)+4,"text-anchor":"end",class:"tick"},svg),fmt(t,YDEC)+"×10⁻³"); }
    for(const t of pow2Ticks(X_LO,X_HI)){ el("line",{x1:x(t),x2:x(t),y1:m.t,y2:H-m.b},g); txt(el("text",{x:x(t),y:H-m.b+20,"text-anchor":"middle",class:"tick"},svg),String(t)); }
    el("rect",{x:x0,y:y(BASE*1.01),width:pw,height:y(BASE)-y(BASE*1.01),fill:"var(--goal)",opacity:0.14},svg);
    const ax=el("g",{class:"axis"},svg); el("line",{x1:x0,x2:x0+pw,y1:H-m.b,y2:H-m.b},ax); el("line",{x1:x0,x2:x0,y1:m.t,y2:H-m.b},ax);
    el("line",{x1:x0,x2:x0+pw,y1:y(BASE),y2:y(BASE),stroke:"var(--ink)","stroke-width":1.1},svg); txt(el("text",{x:x0+pw/2,y:m.t-12,"text-anchor":"middle",class:"ptitle"},svg),title);
    for(const sid of order){ const S=C.series[sid]; if(!S) continue; const rr=S.rows.slice().sort((a,b)=>(a.cap==null?1e9:a.cap)-(b.cap==null?1e9:b.cap));
      el("path",{class:"front",style:`stroke:${cols[sid]}`,d:rr.map((p,i)=>(i?"L":"M")+x(cy(p.cap)[st])+" "+y(p.paper)).join(" ")},svg);
      for(const p of rr){ const cx=x(cy(p.cap)[st]), cyy=y(p.paper); el("circle",{class:"pt",cx,cy:cyy,r:5,fill:cols[sid]},svg);
        const h=el("circle",{class:"hit",cx,cy:cyy,r:11,tabindex:0,role:"button","aria-label":S.name+" "+p.cap},svg); const c=cy(p.cap);
        attachTip(h,tip,card,svg,W,H,cx,cyy,S.name+(p.cap==null?", no cap":", budget "+p.cap+" iteration"+(p.cap>1?"s":"")),[["L1 cycles, mean · p99 · slowest",fmt(c.mean,0)+" · "+fmt(c.p99,0)+" · "+fmt(c.max,0)],["coverage",fmt(p.coverage,1)+"%"],["share of interior commits kept",fmt(p.closed,1)+"%"],["per patch-round",fmt(p.paper,3)+"×10⁻³"],["multiple of MWPM alone","×"+fmt(p.mult,3)],["regression vs MWPM alone",(p.rd>=0?"+":"")+fmt(p.rd)+" pp"],["residual events per shot",fmt(p.residual,0)+" of "+fmt(DATA.residual_none,0)]]); } }
    for(const [k,lab,dy] of [[5,"K = 5",10],[6,"K = 6",26],[null,"no cap",10]]){ const gx=x(cy(k)[st]); el("line",{x1:gx,x2:gx,y1:m.t+dy+6,y2:H-m.b,stroke:"var(--axis)","stroke-width":1,"stroke-dasharray":"3 4"},svg); txt(el("text",{x:gx,y:m.t+dy,"text-anchor":"middle",class:"tick",fill:"var(--ink-2)"},svg),lab); }
    if(x0!==m.l){ const xr=x0+pw; const a2=el("g",{class:"axis"},svg); el("line",{x1:xr,x2:xr,y1:m.t,y2:H-m.b},a2);
      for(const r of MULTS){ const v=BASE*r; if(v<Y_LO||v>Y_HI) continue; el("line",{x1:xr,x2:xr+5,y1:y(v),y2:y(v)},a2); txt(el("text",{x:xr+9,y:y(v)+4,class:"tick"},svg),"×"+fmt(r,2)); } } }
  txt(el("text",{x:18,y:(m.t+H-m.b)/2,"text-anchor":"middle",class:"axlabel",transform:`rotate(-90 18 ${(m.t+H-m.b)/2})`},svg),"logical error rate per patch round");
  txt(el("text",{x:W-14,y:(m.t+H-m.b)/2,"text-anchor":"middle",class:"axlabel",transform:`rotate(90 ${W-14} ${(m.t+H-m.b)/2})`},svg),"multiple of Global MWPM alone");
  txt(el("text",{x:m.l+(W-m.l-m.r)/2,y:H-12,"text-anchor":"middle",class:"axlabel"},svg),"Helios-style UF growth-depth proxy cycles at budget K, per "+R+"-round shot, slowest lane, log scale");
  const L=document.getElementById("l6"); L.innerHTML='<span><i class="key" style="background:#7c3aed"></i>cap alone</span><span><i class="key" style="background:#f59e0b"></i>cap with margin > 0.5</span><span><i class="key" style="background:var(--s2)"></i>cap with margin > 1.0</span><span><i class="band"></i>within 1% of MWPM alone (free region)</span><span>points from left to right: budgets of 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32 iterations, then no cap · 4 fixed cycles per iteration plus merge flooding</span>';
}
draw6(); for(const b of document.querySelectorAll("#k6 button")) b.addEventListener("click",()=>{ stat6.v=b.dataset.stat; for(const o of document.querySelectorAll("#k6 button")) o.setAttribute("aria-pressed",o===b?"true":"false"); draw6(); });
/* 5 summary bars */
(function(){ const host=document.getElementById("c5"); const W=1040,H=420,m={t:40,b:78,l:70,r:20},gap=40; const pw=(W-m.l-m.r-gap)/2; const svg=el("svg",{class:"chart",viewBox:`0 0 ${W} ${H}`,role:"img"},host); const tip=document.getElementById("t5"), card=host.parentElement;
  const y=v=>m.t+(1-v/125)*(H-m.t-m.b);
  DATA.summary.forEach((S,si)=>{ const x0=m.l+si*(pw+gap); const g=el("g",{class:"grid"},svg);
    for(const t of [0,20,40,60,80,100,120]){ el("line",{x1:x0,x2:x0+pw,y1:y(t),y2:y(t)},g); if(si===0) txt(el("text",{x:x0-10,y:y(t)+4,"text-anchor":"end",class:"tick"},svg),t+"%"); }
    el("line",{x1:x0,x2:x0+pw,y1:y(100),y2:y(100),stroke:"var(--ink)","stroke-width":1.1},svg); txt(el("text",{x:x0+pw-4,y:y(100)-6,"text-anchor":"end",class:"tick"},svg),"100% = no saving");
    txt(el("text",{x:x0+pw/2,y:m.t-14,"text-anchor":"middle",class:"ptitle"},svg),S.rule+"  (×"+fmt(S.mult,3)+")");
    const slot=pw/S.models.length, bw=slot*0.3;
    S.models.forEach((md,i)=>{ const cx=x0+slot*(i+0.5);
      for(const [k,off,col,lab] of [["mean",-bw,"var(--bar-mean)","mean"],["p99",0,"var(--bar-max)","p99"]]){ const v=md[k]; const rx=cx+off, ry=y(v); el("rect",{x:rx,y:ry,width:bw,height:y(0)-ry,fill:col},svg); txt(el("text",{x:rx+bw/2,y:ry-6,"text-anchor":"middle",class:"tick",fill:"var(--ink)"},svg),fmt(v,0)+"%");
        const h=el("rect",{class:"hit",x:rx,y:m.t,width:bw,height:y(0)-m.t,tabindex:0,role:"button","aria-label":md.name+" "+lab},svg); attachTip(h,tip,card,svg,W,H,rx+bw/2,ry,md.name,[["rule",S.rule],["mean",pct(md.mean)],["p99",pct(md.p99)],["maximum",pct(md.max)]]); }
      const lines=md.name.replace(" (", "\n(").split("\n"); lines.forEach((ln,li)=>txt(el("text",{x:cx,y:y(0)+20+li*16,"text-anchor":"middle",class:"tick"},svg),ln)); });
    const ax=el("g",{class:"axis"},svg); el("line",{x1:x0,x2:x0+pw,y1:y(0),y2:y(0)},ax); el("line",{x1:x0,x2:x0,y1:m.t,y2:y(0)},ax); });
  txt(el("text",{x:18,y:(m.t+H-m.b)/2,"text-anchor":"middle",class:"axlabel",transform:`rotate(-90 18 ${(m.t+H-m.b)/2})`},svg),"L2 cost with the frontend, % of untouched");
  const L=document.getElementById("l5"); L.innerHTML='<span><i class="key" style="background:var(--bar-mean);border-radius:2px"></i>mean</span><span><i class="key" style="background:var(--bar-max);border-radius:2px"></i>99th percentile</span><span>margin > 1.5 and margin > 1.0 · lower is better · hover a bar for mean, p99 and maximum</span>';
})();
</script>
"""
eyebrow = f"Port-wall Patch-UF frontend · d = {c['d']}, SI1000 p = {c['p']} · {N:,} paired shots · presentation set · non-claim-bearing"
out_html.write_text(html.replace("{DATA_JSON}", json.dumps(data)).replace("{FOOT}", foot).replace("{TITLE}", title).replace("{EYEBROW}", eyebrow)); print("wrote", out_html)
