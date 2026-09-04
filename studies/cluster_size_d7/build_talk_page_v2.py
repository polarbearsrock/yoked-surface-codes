"""Reviewed presentation page: one measured kernel-time metric, three architecture-inspired workload/depth proxies,
one feasibility screen.  Same data as build_talk_page.py plus workload_counts.json and the Helios sharing schedules.
Usage: build_talk_page_v2.py CELL OUT_HTML [TITLE]"""
import json, sys
from pathlib import Path
import numpy as np, sinter

cell = Path(sys.argv[1]); out_html = Path(sys.argv[2]); title_arg = sys.argv[3] if len(sys.argv) > 3 else None
fr = json.load(open(cell / "frontier_interior.json")); prov = json.load(open(cell / "provenance.json"))
T = {f"{r['family']}:{r['value']}": r for r in json.load(open(cell / "l2_timing.json"))["rows"]}
M = {f"{r['family']}:{r['value']}": r for r in json.load(open(cell / "microblossom_cycles.json"))["rows"]}
WL = json.load(open(cell / "workload_counts.json")); Wr = {f"{r['family']}:{r['value']}": r for r in WL["rows"]}
CL = json.load(open(cell / "cap_ladder.json")); HB = json.load(open(cell / "helios_budget_cycles.json"))
c = prov["cell"]; N, O = fr["shots"], fr["num_observables"]; PIECES = c["patches"] * c["rounds"]
conv = lambda p: sinter.shot_error_rate_to_piece_error_rate(p, pieces=PIECES, values=O)
gbase = fr["global_failures"] / N; BASE = conv(gbase)
z = np.load(cell / "cell.npz"); ceiling = 100 * float(z["component_size"][z["component_committed"] & ~z["component_boundary"]].sum()) / float(z["component_size"].sum())
def name(f):
    if f["family"] == "size": return "every interior cluster" if f["value"] is None else f"size ≤ {f['value']}"
    return "every interior cluster" if f["value"] == 0 else f"margin > {f['value']:g}"
m0 = M["none:None"]; w0 = Wr["none:None"]; t0 = T["none:None"]
def pack(k, f=None, is_none=False):
    r = {}
    if k in T: t = T[k]; r["t"] = dict(mean=100 * t["rel_mean"], p99=100 * t["rel_p99"], max=100 * t["rel_max"], mean_ms=t["mean_ms"], p99_ms=t["p99_ms"], max_ms=t["max_ms"], events=t["mean_events"])
    if k in M: m = M[k]; r["mb"] = dict(mean=m["noniso_mean"], p99=m["noniso_p99"], max=m["noniso_max"], rel_mean=100 * m["noniso_mean"] / m0["noniso_mean"], rel_p99=100 * m["noniso_p99"] / m0["noniso_p99"], rel_max=100 * m["noniso_max"] / m0["noniso_max"])
    if k in Wr: w = Wr[k]; b, u = w["block"], w["unit"]; r["wl"] = dict(mean=b["mean"], p99=b["p99"], max=b["max"], edges=b["edges_mean"], w10=b["within10_pct"], w128=b["within128_pct"], rel_mean=100 * b["mean"] / w0["block"]["mean"], rel_p99=100 * b["p99"] / w0["block"]["p99"],
                                                            u_mean=u["mean"], u_p99=u["p99"], u_max=u["max"], u_le10=u["le10_pct"], u_shot_max_mean=u["shot_max_mean"], shots_all=u["shots_all_le10_pct"])
    return r
rows = {"size": [], "margin": []}
for f in fr["rows"]:
    fam = f["family"]
    if fam not in rows or f["coverage_pct"] <= 0 or (fam == "size" and f["value"] is not None and f["value"] % 2): continue
    k = f"{fam}:{f['value']}"; p = f["failures"] / N; paper = conv(p)
    rows[fam].append(dict(key=k, family=fam, value=f["value"], name=name(f), coverage=f["coverage_pct"], paper=1e3 * paper, paper_lo=1e3 * conv(gbase + f["ci_pp"][0] / 100), paper_hi=1e3 * conv(gbase + f["ci_pp"][1] / 100),
                          mult=paper / BASE, mult_lo=conv(gbase + f["ci_pp"][0] / 100) / BASE, mult_hi=conv(gbase + f["ci_pp"][1] / 100) / BASE, rd=f["risk_difference_pp"], lo=f["ci_pp"][0], hi=f["ci_pp"][1], **pack(k)))
for fam in rows: rows[fam].sort(key=lambda r: r["coverage"])
none = dict(name="untouched syndrome", coverage=0.0, paper=1e3 * BASE, paper_lo=1e3 * BASE, paper_hi=1e3 * BASE, mult=1.0, mult_lo=1.0, mult_hi=1.0, rd=0.0, lo=0.0, hi=0.0, **pack("none:None"))
def by(fam, v): return next(r for r in rows[fam] if r["value"] == v)
summary = []
for v in (1.5, 1.0):
    r = by("margin", v)
    summary.append(dict(rule=f"margin > {v:g}", mult=r["mult"], mult_lo=r["mult_lo"], mult_hi=r["mult_hi"], coverage=r["coverage"],
                        metrics=[dict(name="measured PyMatching\nkernel time", mean=r["t"]["mean"], p99=r["t"]["p99"]), dict(name="CPU-interaction demand\n(Micro Blossom-inspired)", mean=r["mb"]["rel_mean"], p99=r["mb"]["rel_p99"]),
                                 dict(name="active detectors per window\n(Zero-G-inspired)", mean=r["wl"]["rel_mean"], p99=r["wl"]["rel_p99"])]))
astrea = [dict(rule=r["name"], coverage=r["coverage"], mult=r["mult"], u_le10=r["wl"]["u_le10"], u_mean=r["wl"]["u_mean"], u_max=r["wl"]["u_max"], shots_all=r["wl"]["shots_all"]) for r in [none, by("margin", 1.5), by("margin", 1.0), by("size", 2), by("size", None)]]
cyc = {("inf" if r["cap"] is None else str(r["cap"])): dict(e12=r["engines12"], e6=r["engines6"], e1=r["engine1"]) for r in HB["rows"] if r["hop"] == 1}
cap = dict(quantum=HB["model"]["quantum"], fixed=HB["model"]["fixed_cycles_per_iteration"], cycles=cyc, series={})
for r in CL["rows"]:
    p = r["failures"] / N; paper = conv(p)
    cap["series"].setdefault(r["series"], dict(name=r["series_name"], rows=[]))["rows"].append(dict(cap=r["cap"], coverage=r["coverage_pct"], paper=1e3 * paper, mult=paper / BASE, rd=r["risk_difference_pp"], residual=r["residual_events_per_shot"], closed=r["closed_share_pct"]))
allp = [r["paper_hi"] for fam in rows for r in rows[fam]] + [r["paper"] for sr in cap["series"].values() for r in sr["rows"]]
y_lo = 1e3 * BASE * 0.975; y_hi = max(1e3 * BASE * 1.21, max(allp) * 1.02); y_step = next(st for st in (0.02, 0.05, 0.1, 0.2, 0.5, 1.0) if (y_hi - y_lo) / st <= 10)
title = title_arg or f"Union-Find + Joint MWPM, d = {c['d']} (reviewed)"
data = dict(shots=N, cell=c, base=1e3 * BASE, y_lo=y_lo, y_hi=y_hi, y_step=y_step, ceiling=ceiling, rows=rows, none=none, summary=summary, astrea=astrea, cap=cap,
            window_rounds=WL["window_rounds"], windows_per_shot=WL["windows_per_shot"], units_per_shot=WL["units_per_shot"], capacity=WL["capacity"], fast_path=WL["fast_path"], timing_processes=json.load(open(cell / "l2_timing.json")).get("processes"), timing_reps=json.load(open(cell / "l2_timing.json")).get("reps"))
foot = f"d = {c['d']}, {c['patches']} patches + {c['yokes']} yokes, {c['rounds']} rounds, SI1000 p = {c['p']}, {N:,} paired shots · vertical axis: logical error per patch round (paper unit), log scale · whiskers on figure 1: paired 95% intervals · the free region is an exploratory point-estimate band; a formal 1% claim needs an independently confirmed frozen rule"
eyebrow = f"Port-wall Patch-UF frontend · d = {c['d']}, SI1000 p = {c['p']} · {N:,} paired shots · one measured kernel-time metric, three architecture-inspired proxies, one feasibility screen · non-claim-bearing"

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
.tick{fill:var(--muted);font-size:12.5px;font-variant-numeric:tabular-nums} .axlabel{fill:var(--ink-2);font-size:13.5px} .ptitle{fill:var(--ink);font-size:14.5px;font-weight:500}
.front{fill:none;stroke-width:1.7;stroke-linejoin:round} .pt{stroke:var(--surface);stroke-width:1.5} .hit{fill:transparent;cursor:pointer} .ci{stroke-width:1.2;opacity:.7}
.legend{display:flex;gap:20px;flex-wrap:wrap;font-size:13.5px;color:var(--ink-2);margin:10px 0 4px;justify-content:center} .legend span{display:inline-flex;align-items:center;gap:7px} .key{width:11px;height:11px;border-radius:50%;display:inline-block} .band{width:22px;height:10px;background:var(--goal);opacity:.18;display:inline-block;border-radius:2px}
.note{font-size:13px;color:var(--muted);max-width:none} .foot{font-size:12.5px;color:var(--muted);text-align:center}
.tooltip{position:absolute;display:none;background:var(--tip);border:1px solid var(--ring);border-radius:6px;padding:10px 12px;font-size:13px;pointer-events:none;box-shadow:0 6px 24px rgba(0,0,0,.12);min-width:250px;z-index:2}
.tooltip .v{font-weight:600;margin-bottom:6px} .tooltip .r{display:flex;justify-content:space-between;gap:16px} .tooltip .r span{color:var(--muted)} .tooltip .r b{font-weight:500;font-variant-numeric:tabular-nums}
table{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}
th{text-align:left;font-weight:500;color:var(--ink-2);font-size:12px;letter-spacing:.04em;text-transform:uppercase;padding:8px 10px;border-bottom:1px solid var(--axis)}
td{padding:7px 10px;border-bottom:1px solid var(--grid)} td.n,th.n{text-align:right} td.rule{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:13px;white-space:nowrap} .scroll{overflow-x:auto}
</style>
<main>
<header><div class="eyebrow">{EYEBROW}</div><h1>{H1}</h1></header>

<section><div class="head"><h2>1 · Accuracy cost of the L1 frontend against coverage</h2></div>
  <div class="card"><div id="c1"></div><div class="legend" id="l1"></div><div class="tooltip" id="t1"></div></div>
  <p class="note">Whiskers are paired 95% intervals against Global MWPM alone, converted to the paper's unit. The shaded band, within 1% of Global MWPM alone, is an exploratory point-estimate region: margin > 1.5 sits inside it with its whole interval; margin > 1.0's interval reaches just past it.</p></section>

<section><div class="head"><h2>2 · Measured PyMatching decode-call time, batch of one, best of three</h2><div class="controls" id="k2"><span>right panel</span><button data-stat="p99" aria-pressed="true">99th percentile</button><button data-stat="max" aria-pressed="false">slowest shot</button></div></div>
  <div class="card"><div id="c2"></div><div class="legend" id="l2"></div><div class="tooltip" id="t2"></div></div>
  <p class="note" id="n2"></p></section>

<section><div class="head"><h2>3 · UF-derived proxy for CPU-interaction demand (Micro Blossom-inspired)</h2><div class="controls" id="k3"><span>right panel</span><button data-stat="p99" aria-pressed="true">99th percentile</button><button data-stat="max" aria-pressed="false">slowest shot</button></div></div>
  <div class="card"><div id="c3"></div><div class="legend" id="l3"></div><div class="tooltip" id="t3"></div></div>
  <p class="note">Counts the residual defects that are not isolated pairs or lone wall defects with positive margin, the frontend's surrogate for the isolated conflicts a Micro Blossom-style accelerator resolves without its CPU. The real decoder detects isolated conflicts by a local tight-edge test, so the two classifications are not identical. Demand only; no cycles or latency are claimed.</p></section>

<section><div class="head"><h2>4 · Active-detector workload and instance capacity (Zero-G-inspired)</h2><div class="controls" id="k4"><span>right panel</span><button data-stat="p99" aria-pressed="true">99th percentile</button><button data-stat="max" aria-pressed="false">slowest window</button></div></div>
  <div class="card"><div id="c4"></div><div class="legend" id="l4"></div><div class="tooltip" id="t4"></div></div>
  <p class="note" id="n4"></p></section>

<section><div class="head"><h2>5 · L2 demand reduction within an exploratory 1% accuracy budget</h2></div>
  <div class="card"><div id="c5"></div><div class="legend" id="l5"></div><div class="tooltip" id="t5"></div></div></section>

<section><div class="head"><h2>6 · Astrea compatibility screen, yoke syndrome excluded</h2></div>
  <div class="scroll"><table id="astrea"><thead><tr><th>syndrome given to L2</th><th class="n">coverage</th><th class="n">× MWPM alone</th><th class="n">patch-basis windows at or below weight 10</th><th class="n">weight per window, mean / max</th><th class="n">shots with all windows at or below 10</th></tr></thead><tbody></tbody></table></div>
  <p class="note" id="n6"></p></section>

<section><div class="head"><h2>7 · Helios-inspired growth/merge critical-path proxy against the frontend budget</h2><div class="controls" id="k7"><span>engines</span><button data-eng="e12" aria-pressed="true">12 independent</button><button data-eng="e6" aria-pressed="false">6 patch-shared</button><button data-eng="e1" aria-pressed="false">1 shared</button><span style="margin-left:12px">right panel</span><button data-stat="p99" aria-pressed="true">99th percentile</button><button data-stat="max" aria-pressed="false">slowest shot</button></div></div>
  <div class="card"><div id="c7"></div><div class="legend" id="l7"></div><div class="tooltip" id="t7"></div></div>
  <p class="note" id="n7"></p></section>

<p class="foot">{FOOT}</p>
</main>
<script>
const DATA = {DATA_JSON};
const fmt=(x,d=2)=>x.toLocaleString(undefined,{minimumFractionDigits:d,maximumFractionDigits:d});
const svgNS="http://www.w3.org/2000/svg";
function el(tag,attrs={},parent=null){const e=document.createElementNS(svgNS,tag);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);if(parent)parent.appendChild(e);return e;}
function txt(e,s){e.textContent=s;return e;}
const Y_LO=DATA.y_lo, Y_HI=DATA.y_hi, Y_STEP=DATA.y_step, YDEC=Y_STEP>=0.1?1:2, BASE=DATA.base;
const MULTS=[1.0,1.02,1.05,1.1,1.15,1.2,1.3,1.4,1.5,1.75,2.0];
function yTicks(){ const out=[]; for(let t=Math.ceil(Y_LO/Y_STEP-1e-9)*Y_STEP;t<=Y_HI+1e-9;t+=Y_STEP) out.push(+t.toFixed(3)); return out; }
function linTicks(lo,hi){ const span=hi-lo; const step=[10,20,50,100,200,500,1000,2000,5000,10000,20000,50000].find(s=>span/s<=8)||100000; const out=[]; for(let t=Math.ceil(lo/step)*step;t<=hi+1e-9;t+=step) out.push(t); return out; }
function logTicks(lo,hi){ const out=[]; for(let e=Math.floor(Math.log10(lo));e<=Math.ceil(Math.log10(hi));e++) for(const m of [1,2,5]){ const t=m*10**e; if(t>=lo&&t<=hi) out.push(t);} return out; }
function pow2Ticks(lo,hi){ const out=[]; for(let t=4;t<=hi;t*=2) if(t>=lo) out.push(t); return out; }
const SHOTS=DATA.shots.toLocaleString(), WINDOWS=(DATA.shots*DATA.windows_per_shot).toLocaleString(), R=DATA.cell.rounds;
const pct=v=>fmt(v,0)+"%", kfmt=v=>v>=1000?(v/1000).toLocaleString(undefined,{maximumFractionDigits:1})+"k":String(v);
function attachTip(h,tip,card,svg,W,H,cx,cy,title,rows){
  const show=()=>{ tip.innerHTML=""; const v=document.createElement("div"); v.className="v"; v.textContent=title; tip.appendChild(v);
    for(const [k,val] of rows){ const r=document.createElement("div"); r.className="r"; const a=document.createElement("span"); a.textContent=k; const b=document.createElement("b"); b.textContent=val; r.append(a,b); tip.appendChild(r); }
    tip.style.display="block"; const rect=card.getBoundingClientRect(), sr=svg.getBoundingClientRect(); const px=sr.left-rect.left+(cx/W)*sr.width, py=sr.top-rect.top+(cy/H)*sr.height; tip.style.left=Math.min(px+14,rect.width-270)+"px"; tip.style.top=Math.max(8,py-10)+"px"; };
  h.addEventListener("pointerenter",show); h.addEventListener("focus",show); h.addEventListener("pointerleave",()=>tip.style.display="none"); h.addEventListener("blur",()=>tip.style.display="none");
}
const BANDSVG='<i class="band"></i>';
function legend(id, extra){ const L=document.getElementById(id); L.innerHTML="";
  const items=[['<i class="key" style="background:var(--s1)"></i>','size cap, interior clusters'],['<i class="key" style="background:var(--s2)"></i>','margin threshold, interior clusters'],['<i class="key" style="background:var(--ink)"></i>','untouched syndrome'],[BANDSVG,'within 1% of MWPM alone, exploratory']].concat(extra||[]);
  for(const [h,t] of items){ const s=document.createElement("span"); s.innerHTML=h+t; L.appendChild(s); } }
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
  for(const v of (cfg.vlines||[])){ if(v.x<cfg.xlim[0]||v.x>cfg.xlim[1]) continue; el("line",{x1:x(v.x),x2:x(v.x),y1:m.t+16,y2:H-m.b,stroke:"var(--axis)","stroke-width":1,"stroke-dasharray":"3 4"},svg); txt(el("text",{x:x(v.x),y:m.t+10,"text-anchor":"middle",class:"tick",fill:"var(--ink-2)"},svg),v.label); }
  const pts=[];
  for(const fam of ["size","margin"]){ const rr=DATA.rows[fam].filter(cfg.has); const col=fam==="size"?"var(--s1)":"var(--s2)";
    el("path",{class:"front",style:`stroke:${col}`,d:rr.map((p,i)=>(i?"L":"M")+x(cfg.xv(p))+" "+y(p.paper)).join(" ")},svg);
    for(const p of rr){ const cx=x(cfg.xv(p)), cy=y(p.paper); if(cfg.ci) el("line",{class:"ci",x1:cx,x2:cx,y1:y(p.paper_lo),y2:y(p.paper_hi),stroke:col},svg); el("circle",{class:"pt",cx,cy,r:5,fill:col},svg); pts.push([p,cx,cy]); } }
  const nx=x(cfg.xv(DATA.none)); el("circle",{class:"pt",cx:nx,cy:y(BASE),r:5.5,fill:"var(--ink)"},svg); pts.push([DATA.none,nx,y(BASE)]);
  if(cfg.ceiling!=null){ el("line",{x1:x(cfg.ceiling),x2:x(cfg.ceiling),y1:m.t,y2:H-m.b,stroke:"var(--axis)","stroke-width":1,"stroke-dasharray":"4 4"},svg); txt(el("text",{x:x(cfg.ceiling)+7,y:m.t+14,class:"tick"},svg),"interior-only ceiling "+fmt(cfg.ceiling,0)+"%"); }
  for(const [p,cx,cy] of pts){ const h=el("circle",{class:"hit",cx,cy,r:11,tabindex:0,role:"button","aria-label":p.name},svg); attachTip(h,tip,card,svg,W,H,cx,cy,p.name,cfg.rows(p)); }
  if(last){ const xr=x0+pw; const a2=el("g",{class:"axis"},svg); el("line",{x1:xr,x2:xr,y1:m.t,y2:H-m.b},a2);
    for(const r of MULTS){ const v=BASE*r; if(v<Y_LO||v>Y_HI) continue; el("line",{x1:xr,x2:xr+5,y1:y(v),y2:y(v)},a2); txt(el("text",{x:xr+9,y:y(v)+4,class:"tick"},svg),"×"+fmt(r,2)); } }
}
function frame(id,tipId,cfgs,xlabel){ const host=document.getElementById(id); host.innerHTML=""; const W=1040,H=470,m={t:34,b:54,l:98,r:78},gap=34; const one=cfgs.length===1; const pw=one?(W-m.l-m.r):(W-m.l-m.r-gap)/2;
  const svg=el("svg",{class:"chart",viewBox:`0 0 ${W} ${H}`,role:"img"},host); const tip=document.getElementById(tipId), card=host.parentElement;
  panel(svg,m.l,pw,m,H,W,cfgs[0],tip,card,true,one); if(!one) panel(svg,m.l+pw+gap,pw,m,H,W,cfgs[1],tip,card,false,true);
  txt(el("text",{x:18,y:(m.t+H-m.b)/2,"text-anchor":"middle",class:"axlabel",transform:`rotate(-90 18 ${(m.t+H-m.b)/2})`},svg),"logical error rate per patch round");
  txt(el("text",{x:W-14,y:(m.t+H-m.b)/2,"text-anchor":"middle",class:"axlabel",transform:`rotate(90 ${W-14} ${(m.t+H-m.b)/2})`},svg),"multiple of Global MWPM alone");
  txt(el("text",{x:m.l+(W-m.l-m.r)/2,y:H-12,"text-anchor":"middle",class:"axlabel"},svg),xlabel); }
const acc=p=>[["coverage",fmt(p.coverage,1)+"%"],["per patch-round",fmt(p.paper,3)+"×10⁻³ ["+fmt(p.paper_lo,3)+", "+fmt(p.paper_hi,3)+"]"],["multiple of MWPM alone","×"+fmt(p.mult,4)+" ["+fmt(p.mult_lo,4)+", "+fmt(p.mult_hi,4)+"]"]];
/* 1 */
frame("c1","t1",[{has:p=>true,xv:p=>p.coverage,xscale:"linear",xlim:[0,100],xticks:[0,10,20,30,40,50,60,70,80,90,100],xfmt:pct,ci:true,ceiling:DATA.ceiling,rows:p=>acc(p).concat([["regression vs MWPM alone",(p.rd>=0?"+":"")+fmt(p.rd)+" pp ["+fmt(p.lo)+", "+fmt(p.hi)+"]"]])}],"coverage: share of detector events removed by the L1 frontend before Global MWPM");
legend("l1",[['<span style="display:inline-block;width:2px;height:14px;background:var(--ink-2)"></span>','paired 95% interval']]);
/* 2-4 */
const stats={c2:"p99",c3:"p99",c4:"p99"};
function draw2(){ const st=stats.c2; const tr=p=>acc(p).concat([["mean time",fmt(p.t.mean_ms,3)+" ms · "+pct(p.t.mean)],["p99",fmt(p.t.p99_ms,3)+" ms · "+pct(p.t.p99)],["slowest shot",fmt(p.t.max_ms,2)+" ms · "+pct(p.t.max)],["residual events per shot",fmt(p.t.events,0)]]);
  const base={has:p=>!!p.t,xscale:"linear",xlim:[35,105],xticks:[40,50,60,70,80,90,100],xfmt:pct,rows:tr};
  frame("c2","t2",[{...base,title:"mean over "+SHOTS+" shots",xv:p=>p.t.mean},{...base,title:st==="max"?"slowest shot":"99th-percentile shot",xv:p=>p.t[st]}],"PyMatching decode-call time on the residual, % of the untouched syndrome"); legend("l2");
  document.getElementById("n2").textContent=`Kernel time of the PyMatching decode call only, batch size one, best of ${DATA.timing_reps} calls per shot, with ${DATA.timing_processes} worker processes running concurrently; the p99 is the p99 of those best-of-${DATA.timing_reps} kernel times, not an operational tail latency. Residual construction, L1, transport and queueing are excluded. Untouched syndrome: mean ${fmt(DATA.none.t.mean_ms,2)} ms, p99 ${fmt(DATA.none.t.p99_ms,2)} ms, slowest ${fmt(DATA.none.t.max_ms,2)} ms.`; }
function draw3(){ const st=stats.c3; const tr=p=>acc(p).concat([["CPU-handled defects per shot, mean",fmt(p.mb.mean,0)+" · "+pct(p.mb.rel_mean)],["p99",fmt(p.mb.p99,0)+" · "+pct(p.mb.rel_p99)],["slowest shot",fmt(p.mb.max,0)+" · "+pct(p.mb.rel_max)]]);
  const v=DATA.rows.size.concat(DATA.rows.margin).filter(p=>p.mb).map(p=>p.mb.mean).concat([DATA.none.mb[st]]); const lo=0, hi=Math.ceil(Math.max(...v)*1.05/50)*50;
  const base={has:p=>!!p.mb,xscale:"linear",xlim:[lo,hi],xticks:linTicks(lo,hi),xfmt:v=>String(v),rows:tr};
  frame("c3","t3",[{...base,title:"mean over "+SHOTS+" shots",xv:p=>p.mb.mean},{...base,title:st==="max"?"slowest shot":"99th-percentile shot",xv:p=>p.mb[st]}],"residual defects outside UF-isolated pairs and lone wall defects, per "+R+"-round shot"); legend("l3"); }
function draw4(){ const st=stats.c4; const tr=p=>acc(p).concat([["active detectors per window, mean",fmt(p.wl.mean,0)+" · "+pct(p.wl.rel_mean)],["p99",fmt(p.wl.p99,0)+" · "+pct(p.wl.rel_p99)],["slowest window",String(p.wl.max)],["candidate edges n(n−1)/2, mean",fmt(p.wl.edges,0)],["windows within fast path (≤ "+DATA.fast_path+")",fmt(p.wl.w10,1)+"%"],["windows within one instance (≤ "+DATA.capacity+")",fmt(p.wl.w128,1)+"%"],["per patch-basis window, mean / max",fmt(p.wl.u_mean,1)+" / "+p.wl.u_max]]);
  const v=DATA.rows.size.concat(DATA.rows.margin).filter(p=>p.wl).map(p=>p.wl.mean).concat([DATA.none.wl[st]]); const lo=5, hi=10**Math.ceil(Math.log10(Math.max(...v)*1.1));
  const base={has:p=>!!p.wl,xscale:"log",xlim:[lo,hi],xticks:logTicks(lo,hi),xfmt:kfmt,rows:tr,vlines:[{x:DATA.fast_path,label:"fast path ≤ "+DATA.fast_path},{x:DATA.capacity,label:"one instance ≤ "+DATA.capacity}]};
  frame("c4","t4",[{...base,title:"mean over "+WINDOWS+" windows",xv:p=>p.wl.mean},{...base,title:st==="max"?"slowest window":"99th-percentile window",xv:p=>p.wl[st]}],"active detectors per "+DATA.window_rounds+"-round window of the whole block (yoke detectors included), log scale"); legend("l4");
  const m10=DATA.rows.margin.find(p=>p.value===1.0), m15=DATA.rows.margin.find(p=>p.value===1.5), s4=DATA.rows.size.find(p=>p.value===4);
  document.getElementById("n4").textContent=`Workload and capacity only; no cycle count or latency is derived, since the published FPGA numbers are for residuals of five to ten detectors and this cell sits far outside that regime. Dashed lines mark the published fast-path threshold and the configured single-instance capacity. Windows within one instance: untouched ${fmt(DATA.none.wl.w128,1)}%, margin > 1.5 ${fmt(m15.wl.w128,0)}%, margin > 1.0 ${fmt(m10.wl.w128,0)}%, size ≤ 4 ${fmt(s4.wl.w128,0)}%. A per-patch deployment would see the per patch-basis counts in the tooltip, plus a separate yoke stage.`; }
draw2(); draw3(); draw4();
for(const [kid,cid,fn] of [["k2","c2",draw2],["k3","c3",draw3],["k4","c4",draw4]]) for(const b of document.querySelectorAll("#"+kid+" button")) b.addEventListener("click",()=>{ stats[cid]=b.dataset.stat; for(const o of document.querySelectorAll("#"+kid+" button")) o.setAttribute("aria-pressed",o===b?"true":"false"); fn(); });
/* 5 summary bars */
(function(){ const host=document.getElementById("c5"); const W=1040,H=420,m={t:40,b:78,l:70,r:20},gap=40; const pw=(W-m.l-m.r-gap)/2; const svg=el("svg",{class:"chart",viewBox:`0 0 ${W} ${H}`,role:"img"},host); const tip=document.getElementById("t5"), card=host.parentElement;
  const y=v=>m.t+(1-v/125)*(H-m.t-m.b);
  DATA.summary.forEach((S,si)=>{ const x0=m.l+si*(pw+gap); const g=el("g",{class:"grid"},svg);
    for(const t of [0,20,40,60,80,100,120]){ el("line",{x1:x0,x2:x0+pw,y1:y(t),y2:y(t)},g); if(si===0) txt(el("text",{x:x0-10,y:y(t)+4,"text-anchor":"end",class:"tick"},svg),t+"%"); }
    el("line",{x1:x0,x2:x0+pw,y1:y(100),y2:y(100),stroke:"var(--ink)","stroke-width":1.1},svg); txt(el("text",{x:x0+pw-4,y:y(100)-6,"text-anchor":"end",class:"tick"},svg),"100% = no reduction");
    txt(el("text",{x:x0+pw/2,y:m.t-14,"text-anchor":"middle",class:"ptitle"},svg),S.rule+"  (×"+fmt(S.mult,4)+" ["+fmt(S.mult_lo,4)+", "+fmt(S.mult_hi,4)+"], "+fmt(S.coverage,0)+"% coverage)");
    const slot=pw/S.metrics.length, bw=slot*0.3;
    S.metrics.forEach((md,i)=>{ const cx=x0+slot*(i+0.5);
      for(const [k,off,col,lab] of [["mean",-bw,"var(--bar-mean)","mean"],["p99",0,"var(--bar-max)","p99"]]){ const v=md[k]; const rx=cx+off, ry=y(v); el("rect",{x:rx,y:ry,width:bw,height:y(0)-ry,fill:col},svg); txt(el("text",{x:rx+bw/2,y:ry-6,"text-anchor":"middle",class:"tick",fill:"var(--ink)"},svg),fmt(v,0)+"%");
        const h=el("rect",{class:"hit",x:rx,y:m.t,width:bw,height:y(0)-m.t,tabindex:0,role:"button","aria-label":md.name+" "+lab},svg); attachTip(h,tip,card,svg,W,H,rx+bw/2,ry,md.name.replace("\n"," "),[["rule",S.rule],["mean",pct(md.mean)],["p99",pct(md.p99)]]); }
      const lines=md.name.split("\n"); lines.forEach((ln,li)=>txt(el("text",{x:cx,y:y(0)+20+li*16,"text-anchor":"middle",class:"tick"},svg),ln)); });
    const ax=el("g",{class:"axis"},svg); el("line",{x1:x0,x2:x0+pw,y1:y(0),y2:y(0)},ax); el("line",{x1:x0,x2:x0,y1:m.t,y2:y(0)},ax); });
  txt(el("text",{x:18,y:(m.t+H-m.b)/2,"text-anchor":"middle",class:"axlabel",transform:`rotate(-90 18 ${(m.t+H-m.b)/2})`},svg),"with the frontend, % of untouched");
  document.getElementById("l5").innerHTML='<span><i class="key" style="background:var(--bar-mean);border-radius:2px"></i>mean</span><span><i class="key" style="background:var(--bar-max);border-radius:2px"></i>99th percentile</span><span>one measured time, two structural demand proxies · lower is better</span>';
})();
/* 6 astrea table */
(function(){ const tb=document.querySelector("#astrea tbody");
  for(const a of DATA.astrea){ const tr=document.createElement("tr"); for(const [v,cls] of [[a.rule,"rule"],[fmt(a.coverage,1)+"%","n"],["×"+fmt(a.mult,3),"n"],[fmt(a.u_le10,1)+"%","n"],[fmt(a.u_mean,1)+" / "+a.u_max,"n"],[fmt(a.shots_all,2)+"%","n"]]){ const td=document.createElement("td"); td.className=cls; td.textContent=v; tr.appendChild(td);} tb.appendChild(tr); }
  document.getElementById("n6").textContent=`Astrea's brute-force search covers a patch-basis window of at most weight 10; this screen excludes the yoke detectors and so is optimistic. No safe rule makes an entire shot Astrea-compatible: the share of shots whose ${DATA.units_per_shot} patch-basis windows all sit at or below 10 is ${fmt(DATA.astrea[1].shots_all,2)}% at margin > 1.5 and ${fmt(DATA.astrea[2].shots_all,2)}% at margin > 1.0. This is a statement about p = ${DATA.cell.p}, thirty times Astrea's evaluation point, not about Astrea; Astrea-G's greedy reach was not modelled.`; })();
/* 7 helios */
const s7={eng:"e12",stat:"p99"};
function draw7(){ const C=DATA.cap; const host=document.getElementById("c7"); host.innerHTML=""; const W=1040,H=470,m={t:34,b:54,l:98,r:78},gap=34; const pw=(W-m.l-m.r-gap)/2; const svg=el("svg",{class:"chart",viewBox:`0 0 ${W} ${H}`,role:"img"},host); const tip=document.getElementById("t7"), card=host.parentElement;
  const y=v=>m.t+(1-(Math.log10(v)-Math.log10(Y_LO))/(Math.log10(Y_HI)-Math.log10(Y_LO)))*(H-m.t-m.b);
  const cols={cap:"#7c3aed",cap_m05:"#f59e0b",cap_m10:"var(--s2)"}; const order=["cap","cap_m05","cap_m10"]; const cy=k=>C.cycles[k==null?"inf":String(k)][s7.eng];
  const cmax=Math.max(...Object.values(C.cycles).map(v=>v[s7.eng].max)); const X_LO=3, X_HI=10**Math.ceil(Math.log10(cmax*1.1));
  const panels=[["mean","mean over "+SHOTS+" shots",m.l],[s7.stat,s7.stat==="max"?"slowest shot":"99th-percentile shot",m.l+pw+gap]];
  for(const [st,title,x0] of panels){ const x=v=>x0+((Math.log10(v)-Math.log10(X_LO))/(Math.log10(X_HI)-Math.log10(X_LO)))*pw; const g=el("g",{class:"grid"},svg);
    for(const t of yTicks()){ el("line",{x1:x0,x2:x0+pw,y1:y(t),y2:y(t)},g); if(x0===m.l) txt(el("text",{x:x0-10,y:y(t)+4,"text-anchor":"end",class:"tick"},svg),fmt(t,YDEC)+"×10⁻³"); }
    for(const t of pow2Ticks(X_LO,X_HI)){ el("line",{x1:x(t),x2:x(t),y1:m.t,y2:H-m.b},g); txt(el("text",{x:x(t),y:H-m.b+20,"text-anchor":"middle",class:"tick"},svg),String(t)); }
    el("rect",{x:x0,y:y(BASE*1.01),width:pw,height:y(BASE)-y(BASE*1.01),fill:"var(--goal)",opacity:0.14},svg);
    const ax=el("g",{class:"axis"},svg); el("line",{x1:x0,x2:x0+pw,y1:H-m.b,y2:H-m.b},ax); el("line",{x1:x0,x2:x0,y1:m.t,y2:H-m.b},ax); el("line",{x1:x0,x2:x0+pw,y1:y(BASE),y2:y(BASE),stroke:"var(--ink)","stroke-width":1.1},svg); txt(el("text",{x:x0+pw/2,y:m.t-12,"text-anchor":"middle",class:"ptitle"},svg),title);
    for(const [k,lab,dy] of [[5,"K = 5",10],[6,"K = 6",26],[null,"no cap",10]]){ const gx=x(cy(k)[st]); el("line",{x1:gx,x2:gx,y1:m.t+dy+6,y2:H-m.b,stroke:"var(--axis)","stroke-width":1,"stroke-dasharray":"3 4"},svg); txt(el("text",{x:gx,y:m.t+dy,"text-anchor":"middle",class:"tick",fill:"var(--ink-2)"},svg),lab); }
    for(const sid of order){ const S=C.series[sid]; if(!S) continue; const rr=S.rows.slice().sort((a,b)=>(a.cap==null?1e9:a.cap)-(b.cap==null?1e9:b.cap));
      el("path",{class:"front",style:`stroke:${cols[sid]}`,d:rr.map((p,i)=>(i?"L":"M")+x(cy(p.cap)[st])+" "+y(p.paper)).join(" ")},svg);
      for(const p of rr){ const cx=x(cy(p.cap)[st]), cyy=y(p.paper); el("circle",{class:"pt",cx,cy:cyy,r:5,fill:cols[sid]},svg); const h=el("circle",{class:"hit",cx,cy:cyy,r:11,tabindex:0,role:"button","aria-label":S.name+" "+p.cap},svg);
        const c12=C.cycles[p.cap==null?"inf":String(p.cap)];
        attachTip(h,tip,card,svg,W,H,cx,cyy,S.name+(p.cap==null?", no cap":", budget "+p.cap+" iteration"+(p.cap>1?"s":"")),[["12 engines: mean · p99 · slowest",fmt(c12.e12.mean,0)+" · "+fmt(c12.e12.p99,0)+" · "+fmt(c12.e12.max,0)],["6 patch-shared engines",fmt(c12.e6.mean,0)+" · "+fmt(c12.e6.p99,0)+" · "+fmt(c12.e6.max,0)],["1 shared engine (aggregate work)",fmt(c12.e1.mean,0)+" · "+fmt(c12.e1.p99,0)+" · "+fmt(c12.e1.max,0)],["coverage",fmt(p.coverage,1)+"%"],["per patch-round",fmt(p.paper,3)+"×10⁻³ · ×"+fmt(p.mult,4)],["residual events per shot",fmt(p.residual,0)]]); } }
    if(x0!==m.l){ const xr=x0+pw; const a2=el("g",{class:"axis"},svg); el("line",{x1:xr,x2:xr,y1:m.t,y2:H-m.b},a2); for(const r of MULTS){ const v=BASE*r; if(v<Y_LO||v>Y_HI) continue; el("line",{x1:xr,x2:xr+5,y1:y(v),y2:y(v)},a2); txt(el("text",{x:xr+9,y:y(v)+4,class:"tick"},svg),"×"+fmt(r,2)); } } }
  txt(el("text",{x:18,y:(m.t+H-m.b)/2,"text-anchor":"middle",class:"axlabel",transform:`rotate(-90 18 ${(m.t+H-m.b)/2})`},svg),"logical error rate per patch round");
  txt(el("text",{x:W-14,y:(m.t+H-m.b)/2,"text-anchor":"middle",class:"axlabel",transform:`rotate(90 ${W-14} ${(m.t+H-m.b)/2})`},svg),"multiple of Global MWPM alone");
  const engName={e12:"12 independent lane engines (critical path)",e6:"6 patch-shared engines",e1:"1 shared engine (aggregate work)"}[s7.eng];
  txt(el("text",{x:m.l+(W-m.l-m.r)/2,y:H-12,"text-anchor":"middle",class:"axlabel"},svg),"growth/merge proxy cycles at budget K, per "+R+"-round shot · "+engName+" · log scale");
  document.getElementById("l7").innerHTML='<span><i class="key" style="background:#7c3aed"></i>cap alone</span><span><i class="key" style="background:#f59e0b"></i>cap with margin > 0.5</span><span><i class="key" style="background:var(--s2)"></i>cap with margin > 1.0</span><span>'+BANDSVG+'within 1% of MWPM alone, exploratory</span><span>points left to right: budgets of 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32 iterations, then no cap</span>';
  document.getElementById("n7").textContent=`Depth, not work: ${C.fixed} fixed cycles per growth iteration (one growing, two controller, one merge-settle) plus merge flooding priced at one cycle per hop of the largest cluster with an event in that iteration, using its final forest diameter as an upper bound; the iteration is the closing time over a quantum of ${fmt(C.quantum,3)} weight units (largest lane edge weight / 16). This is an architectural interpretation of the Helios stage model, not a calibrated equation from the paper. The engine toggle changes the sharing assumption: the slowest of 12 lanes, the slowest patch running its two lanes serially, or all lanes on one engine, which equals the aggregate work. Not priced: peeling, the margin evaluation pass, patch transactions and residual updates.`; }
draw7(); for(const b of document.querySelectorAll("#k7 button")) b.addEventListener("click",()=>{ if(b.dataset.eng){ s7.eng=b.dataset.eng; for(const o of document.querySelectorAll("#k7 button[data-eng]")) o.setAttribute("aria-pressed",o===b?"true":"false"); } else { s7.stat=b.dataset.stat; for(const o of document.querySelectorAll("#k7 button[data-stat]")) o.setAttribute("aria-pressed",o===b?"true":"false"); } draw7(); });
</script>
"""
out_html.write_text(html.replace("{DATA_JSON}", json.dumps(data)).replace("{FOOT}", foot).replace("{TITLE}", title).replace("{H1}", title).replace("{EYEBROW}", eyebrow)); print("wrote", out_html)
