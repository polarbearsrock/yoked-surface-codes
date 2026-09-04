"""Build the self-contained Pareto-frontier page from frontier.json and analysis.json."""
import json, math, sys
from pathlib import Path

cell = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("out/cluster_size_study_d7/d7_p0.003_100k")
out_html = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("out/cluster_size_study_d7/frontier.html")
fr = json.load(open(cell / "frontier.json")); an = json.load(open(cell / "analysis.json"))
N = fr["shots"]; gfail = fr["global_failures"]; O = fr.get("num_observables", 12); gfail_obs = fr.get("global_observable_failures")
import sinter
prov = json.load(open(cell / "provenance.json")); PATCHES, ROUNDS = prov["cell"]["patches"], prov["cell"]["rounds"]; PIECES = PATCHES * ROUNDS
def paper(p_shot): return sinter.shot_error_rate_to_piece_error_rate(p_shot, pieces=PIECES, values=O)
PAPER_NONE = paper(gfail / N)

def rule_name(r):
    parts = []
    if r["size_cap"] == 0: return "nothing committed"
    if r["size_cap"] is not None: parts.append(f"size ≤ {r['size_cap']}")
    if r["interior"]: parts.append("interior only")
    if r["margin_gt"] > 0: parts.append(f"margin > {r['margin_gt']:g}")
    return ", ".join(parts) if parts else "everything committed"

points = []
for r in fr["rows"]:
    points.append(dict(name=rule_name(r), size_cap=r["size_cap"], interior=bool(r["interior"]), margin=r["margin_gt"],
                       coverage=r["coverage_pct"], rd=r["risk_difference_pp"], lo=r["ci_pp"][0], hi=r["ci_pp"][1],
                       failures=r["failures"], regressions=r["regressions"], recoveries=r["recoveries"],
                       ler=100 * r["failures"] / N,
                       paper=1e3 * paper(r["failures"] / N), paper_lo=1e3 * paper(gfail / N + r["ci_pp"][0] / 100), paper_hi=1e3 * paper(gfail / N + r["ci_pp"][1] / 100), mult=paper(r["failures"] / N) / PAPER_NONE,
                       ler_obs=(100 * r["observable_failures"] / (N * O)) if "observable_failures" in r else None,
                       rd_obs=r.get("observable_risk_difference_pp"),
                       residual=r["residual_events_per_shot"], components=r["components"]))
# Pareto set: maximise coverage, minimise regression (point estimates); nothing-committed anchors the origin.
by_cov = sorted(points, key=lambda p: (-p["coverage"], p["rd"]))
pareto, best = [], math.inf
for p in by_cov:
    if p["rd"] < best - 1e-12:
        pareto.append(p); best = p["rd"]
pareto_names = {p["name"] for p in pareto}
for p in points: p["pareto"] = p["name"] in pareto_names
# A readable staircase: drop near-duplicate frontier points (same coverage and regression within noise,
# only a redundant extra condition) and the trivial sub-1% coverage anchors; keep the simplest name per step.
distinct, last = [], None
for p in pareto:
    if p["coverage"] < 1.0 and p["name"] != "nothing committed": continue
    if last is None or p["coverage"] < last["coverage"] - 0.5 or p["rd"] < last["rd"] - 0.1:
        distinct.append(p); last = p
    elif p["name"].count(",") < last["name"].count(","):
        distinct[-1] = p; last = p
# Within a tight noise window around each step, prefer the rule with the fewest conditions.
for i, step in enumerate(distinct):
    cands = [q for q in points if q["coverage"] >= step["coverage"] - 0.2 and q["rd"] <= step["rd"] + 0.02]
    distinct[i] = min(cands, key=lambda q: (q["name"].count(","), -q["coverage"], q["rd"]))
seen, unique = set(), []
for q in distinct:
    if q["name"] not in seen: seen.add(q["name"]); unique.append(q)
distinct = unique
distinct_names = {p["name"] for p in distinct}
for p in points: p["distinct"] = p["name"] in distinct_names

att = an["attribution"]
def wilson(k, n):
    if not n: return None
    z = 1.96; ph = k / n; den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den; h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return [100 * (c - h), 100 * (c + h)]
culprit = []
for s_all, s_int, s_wall in zip(att["by_size"], att["by_size_interior"], att["by_size_wall"]):
    row = dict(band=s_all["band"].replace("-+", "+"))
    for key, s in (("interior", s_int), ("wall", s_wall)):
        row[key] = None if s["rate"] is None else dict(rate=100 * s["rate"], n=s["components"], ci=wilson(s["culprits"], s["components"]))
    culprit.append(row)
lessons_data = dict(regressed=att["regressed_shots"], no_single=att["shots_without_single_culprit"],
                    interior_rate=100 * att["by_wall"][0]["rate"], wall_rate=100 * att["by_wall"][1]["rate"])

data = dict(shots=N, global_failures=gfail, global_ler=100 * gfail / N, global_paper=1e3 * PAPER_NONE, patches=PATCHES, rounds=ROUNDS, pieces=PIECES, paper_fit_p001=6 * 28 * 8 ** -7 / 500, global_ler_obs=(100 * gfail_obs / (N * O)) if gfail_obs is not None else None, num_observables=O, points=points, pareto=[p["name"] for p in pareto], distinct=[p["name"] for p in distinct], culprit=culprit, lessons=lessons_data,
            cell=fr["cell"])
knee = next((p for p in pareto if p["rd"] <= 1.1), pareto[-1])
zero = next((p for p in pareto if p["hi"] <= 0.05), None)

html = r"""<title>Which Clusters Can L1 Commit</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{color-scheme:light;--page:#f7f7f4;--surface:#fcfcfb;--ink:#0b0b0b;--ink-2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);
  --s1:#2a78d6;--s2:#eb6834;--s1-wash:rgba(42,120,214,.10);--accent:#2a78d6;--tip:#ffffff;--font:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;--mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#ffffff;--ink-2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s1-wash:rgba(57,135,229,.12);--accent:#3987e5;--tip:#242423}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#ffffff;--ink-2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s1-wash:rgba(57,135,229,.12);--accent:#3987e5;--tip:#242423}
body{margin:0;background:var(--page);color:var(--ink);font-family:var(--font);font-size:15px;line-height:1.5}
main{max-width:1040px;margin:0 auto;padding:40px 24px 64px;display:flex;flex-direction:column;gap:40px}
h1{font-size:30px;font-weight:600;letter-spacing:-.01em;line-height:1.15;margin:0 0 8px;text-wrap:balance}
h2{font-size:19px;font-weight:600;margin:0 0 4px}
p{max-width:68ch;margin:0}
.lede{color:var(--ink-2);font-size:16px;max-width:70ch}
.eyebrow{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin-bottom:12px;font-weight:500}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px}
.tile{background:var(--surface);border:1px solid var(--ring);border-radius:6px;padding:16px 18px}
.tile .label{font-size:13px;color:var(--ink-2)}
.tile .value{font-size:30px;font-weight:600;margin-top:4px;line-height:1.1}
.tile .sub{font-size:13px;color:var(--muted);margin-top:6px}
section{display:flex;flex-direction:column;gap:12px}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:6px;padding:20px 20px 12px;position:relative}
.legend{display:flex;gap:20px;flex-wrap:wrap;font-size:13px;color:var(--ink-2);margin:4px 0 8px}
.legend span{display:inline-flex;align-items:center;gap:8px}
.key{width:10px;height:10px;border-radius:50%;display:inline-block}
.keyline{width:22px;height:0;border-top:2px solid var(--ink);display:inline-block}
.keyring{width:10px;height:10px;border-radius:50%;display:inline-block;box-shadow:0 0 0 2px var(--surface),0 0 0 3.5px var(--ink)}
svg{display:block;width:100%;height:auto;font-family:var(--font)}
.grid line{stroke:var(--grid);stroke-width:1}
.axis line,.axis path{stroke:var(--axis);stroke-width:1;fill:none}
.tick{fill:var(--muted);font-size:12px;font-variant-numeric:tabular-nums}
.axlabel{fill:var(--ink-2);font-size:13px}
.pt{stroke:var(--surface);stroke-width:2}
.pt.s1{fill:var(--s1)}.pt.s2{fill:var(--s2)}
.pt.dim{opacity:.35}
.front{fill:none;stroke:var(--ink);stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.plabel{fill:var(--ink);font-size:12px;font-family:var(--mono)}
.plabel.small{fill:var(--ink-2)}
.hit{fill:transparent;cursor:crosshair}
.ci{stroke-width:2;stroke-linecap:round}
.ci.s1{stroke:var(--s1)}.ci.s2{stroke:var(--s2)}
.tooltip{position:absolute;pointer-events:none;background:var(--tip);color:var(--ink);border:1px solid var(--ring);border-radius:6px;padding:10px 12px;font-size:13px;box-shadow:0 6px 24px rgba(0,0,0,.14);min-width:220px;display:none;z-index:2}
.tooltip .v{font-weight:600;font-size:15px}
.tooltip .r{display:flex;justify-content:space-between;gap:16px;color:var(--ink-2)}
.tooltip .r b{color:var(--ink);font-weight:500;font-variant-numeric:tabular-nums}
table{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}
th{text-align:left;font-weight:500;color:var(--ink-2);font-size:12px;letter-spacing:.04em;text-transform:uppercase;padding:8px 10px;border-bottom:1px solid var(--axis)}
td{padding:8px 10px;border-bottom:1px solid var(--grid)}
td.n,th.n{text-align:right}
td.rule{font-family:var(--mono);font-size:13px}
.scroll{overflow-x:auto}
.lessons{display:grid;grid-template-columns:repeat(auto-fit,minmax(225px,1fr));gap:16px}
.lesson{background:var(--surface);border:1px solid var(--ring);border-radius:6px;padding:16px 18px}
.lesson h3{margin:0 0 6px;font-size:15px;font-weight:600}
.lesson p{font-size:14px;color:var(--ink-2);max-width:none}
.note{font-size:13px;color:var(--muted);max-width:none}
details summary{cursor:pointer;color:var(--ink-2);font-size:14px}
.toggle{position:absolute;top:14px;right:16px;font-size:12px;color:var(--ink-2);display:flex;gap:6px;align-items:center;flex-wrap:wrap;justify-content:flex-end;max-width:60%}
.toggle button{font:inherit;background:var(--page);color:var(--ink);border:1px solid var(--ring);border-radius:4px;padding:2px 8px;cursor:pointer}
.toggle button[aria-pressed="true"]{background:var(--ink);color:var(--surface);border-color:var(--ink)}
.toggle button:focus-visible,.hit:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
@media (prefers-reduced-motion:no-preference){.pt{transition:r .12s ease}}
</style>
<main>
<header>
  <div class="eyebrow">Port-wall Patch-UF frontend · d = 7, SI1000 p = 0.003 · 100,000 paired shots · non-claim-bearing</div>
  <h1>Which clusters can L1 commit?</h1>
  <p class="lede">Every commit rule trades syndrome coverage against logical error. This page plots that trade-off for {NRULES} rules built from cluster size, wall contact, and the confidence margin, decoded exactly from one retained corpus, and asks the underlying question directly: which committed clusters cause the regressions?</p>
</header>

<div class="tiles" id="tiles"></div>

<section>
  <h2>The Pareto frontier</h2>
  <p>Coverage is the share of lane-owned detector events removed before Global MWPM. The vertical axis can show either the regression, the paired risk difference against Global MWPM on the same shots in percentage points on a logarithmic scale where rules within noise of zero sit on the floor, or the logical error rate itself, the share of shots the full decoder gets wrong, with Global MWPM alone drawn as the reference. The two views are the same frontier: a rule's failures are Global's failures plus its net regressions. Ringed points are Pareto-optimal: no rule has both more coverage and less error.</p>
  <div class="card" id="frontier-card">
    <div class="toggle" role="group" aria-label="Vertical axis" id="ymodes"><span>y-axis</span><button data-mode="paper" aria-pressed="true">LER per patch-round (paper unit)</button><button data-mode="log" aria-pressed="false">regression, log</button><button data-mode="lin" aria-pressed="false">regression, linear</button><button data-mode="ler" aria-pressed="false">LER per shot</button><button data-mode="lerobs" aria-pressed="false" hidden>LER per observable</button></div>
    <div class="legend"><span><i class="key" style="background:var(--s1)"></i>interior-only rules</span><span><i class="key" style="background:var(--s2)"></i>rules that also commit wall-touching clusters</span><span><i class="keyring"></i>Pareto-optimal</span><span><i class="keyline"></i>frontier</span></div>
    <div id="frontier"></div>
    <div class="tooltip" id="tip"></div>
  </div>
  <p class="note">Hover or focus a point for its rule, paired counts, and 95% interval. Wall-touching rules never reach the frontier below 77% coverage: for the same coverage an interior-only rule is always less wrong.</p>
</section>

<section>
  <h2>Why: culprit rate by cluster size</h2>
  <p>For each of the {REGRESSED} regressed shots, every committed component was restored to the residual on its own and the shot re-decoded. A component whose restoration alone fixes the shot is a culprit. The rate is the share of culprits among committed components of that kind in regressed shots, with 95% Wilson intervals; the vertical axis is logarithmic.</p>
  <div class="card">
    <div class="legend"><span><i class="key" style="background:var(--s1)"></i>interior clusters</span><span><i class="key" style="background:var(--s2)"></i>wall-touching clusters</span></div>
    <div id="culprit"></div>
    <div class="tooltip" id="tip2"></div>
  </div>
</section>

<section>
  <h2>What we learn</h2>
  <div class="lessons" id="lessons"></div>
</section>

<section>
  <h2>The frontier as a staircase</h2>
  <p>Pareto-optimal rules, thinned to distinct steps: where several rules sit at the same coverage and regression within noise, only the simplest is listed. Every ringed point is in the full table below.</p>
  <div class="scroll"><table id="ptable"><thead><tr><th>rule</th><th class="n">coverage</th><th class="n">LER per shot</th><th class="n">regression, pp</th><th class="n">95% CI</th><th class="n">regressions</th><th class="n">recoveries</th><th class="n">residual events / shot</th></tr></thead><tbody></tbody></table></div>
  <details><summary>All rules in the grid (table view)</summary><div class="scroll"><table id="alltable"><thead><tr><th>rule</th><th class="n">coverage</th><th class="n">regression, pp</th><th class="n">95% CI</th><th class="n">failures</th><th class="n">residual events / shot</th></tr></thead><tbody></tbody></table></div></details>
  <p class="note">Method: the union-find run does not depend on the commit rule, so each rule's residual is rebuilt from the retained committed components and Global MWPM is decoded once over all shots. The everything-committed rule reproduces the corpus's retained treatment predictions bit for bit. Intervals are Wald on the paired difference. Global MWPM failed {GFAIL} of {SHOTS} shots.</p>
</section>
</main>
<script>
const DATA = {DATA_JSON};
const fmt = (x, d=2) => x.toLocaleString(undefined, {minimumFractionDigits:d, maximumFractionDigits:d});
const svgNS = "http://www.w3.org/2000/svg";
function el(tag, attrs={}, parent=null){ const e=document.createElementNS(svgNS, tag); for(const [k,v] of Object.entries(attrs)) e.setAttribute(k, v); if(parent) parent.appendChild(e); return e; }
function txt(e, s){ e.textContent = s; return e; }

/* tiles */
(function(){
  const P = DATA.points, pareto = P.filter(p=>p.pareto).sort((a,b)=>b.coverage-a.coverage);
  const all = P.find(p=>p.name==="everything committed");
  const knee = pareto.find(p=>p.rd<=1.1) || pareto[pareto.length-1];
  const zero = pareto.filter(p=>p.hi<=0.05).sort((a,b)=>b.coverage-a.coverage)[0];
  const tiles = [
    ["Global MWPM alone, per patch-round", fmt(DATA.global_paper,2)+"×10⁻³", "the paper's unit · "+fmt(100*DATA.global_failures/DATA.shots,1)+"% per shot · "+DATA.shots.toLocaleString()+" paired shots"],
    ["Everything committed", fmt(all.paper,2)+"×10⁻³", "×"+fmt(all.mult,3)+" · +"+fmt(all.rd)+" pp per shot · "+fmt(all.coverage,1)+"% coverage"],
    ["Best rule near 1 pp", fmt(knee.paper,2)+"×10⁻³", "×"+fmt(knee.mult,3)+" · "+knee.name+" · "+fmt(knee.coverage,1)+"% coverage"],
    ["Most coverage at ≈0 pp", zero ? fmt(zero.coverage,1)+"%" : "—", zero ? zero.name+" · "+fmt(zero.paper,2)+"×10⁻³, ×"+fmt(zero.mult,3) : "no rule"]
  ];
  const root = document.getElementById("tiles");
  for(const [l,v,s] of tiles){ const d=document.createElement("div"); d.className="tile"; const a=document.createElement("div"); a.className="label"; a.textContent=l; const b=document.createElement("div"); b.className="value"; b.textContent=v; const c=document.createElement("div"); c.className="sub"; c.textContent=s; d.append(a,b,c); root.appendChild(d); }
})();

/* frontier chart */
let yMode = "paper";
function paretoOf(points, key){ const s=points.slice().sort((a,b)=>(b.coverage-a.coverage)||(a[key]-b[key])); const out=[]; let best=Infinity; for(const p of s){ if(p[key]<best-1e-9){ out.push(p); best=p[key]; } } return new Set(out.map(p=>p.name)); }
function drawFrontier(){
  const host = document.getElementById("frontier"); host.innerHTML = "";
  const W=960, H=520, m={t:16,r:24,b:52,l:64};
  const P = DATA.points; const floor = 0.01, top = 20;
  const isPaper = yMode==="paper"; const isLer = isPaper||yMode==="ler"||yMode==="lerobs"; const key = isPaper?"paper":(yMode==="lerobs"?"ler_obs":(isLer?"ler":"rd"));
  const gref = isPaper?DATA.global_paper:(yMode==="lerobs"?DATA.global_ler_obs:DATA.global_ler);
  const vals = P.map(p=>p[key]).filter(v=>v!=null); const vmax = Math.max(...vals), vmin = Math.min(...vals);
  const lerLo = isPaper?Math.floor((Math.min(vmin,gref)-0.02)*10)/10:Math.floor((Math.min(vmin,gref)-0.5)), lerHi = isPaper?Math.ceil((vmax+0.02)*10)/10:Math.ceil(vmax+0.5);
  const svg = el("svg",{viewBox:`0 0 ${W} ${H}`,role:"img","aria-label":"Coverage versus "+(isLer?"logical error rate":"regression")+" for "+P.length+" commit rules with the Pareto frontier"},host);
  const x = c => m.l + (c/90)*(W-m.l-m.r);
  const y = yMode==="log" ? (r => { const v=Math.max(r,floor); return m.t + (1-(Math.log10(v)-Math.log10(floor))/(Math.log10(top)-Math.log10(floor)))*(H-m.t-m.b); })
          : yMode==="lin" ? (r => m.t + (1-Math.max(r,0)/12)*(H-m.t-m.b))
          : (r => m.t + (1-(r-lerLo)/(lerHi-lerLo))*(H-m.t-m.b));
  const g = el("g",{class:"grid"},svg);
  let yticks;
  if(yMode==="log") yticks=[0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20]; else if(yMode==="lin") yticks=[0,2,4,6,8,10,12];
  else { const step=isPaper?0.1:((lerHi-lerLo)>12?2:1); yticks=[]; for(let t=lerLo;t<=lerHi+1e-9;t+=step) yticks.push(+t.toFixed(2)); }
  for(const t of yticks){ el("line",{x1:m.l,x2:W-m.r,y1:y(t),y2:y(t)},g); txt(el("text",{x:m.l-10,y:y(t)+4,"text-anchor":"end",class:"tick"},svg), yMode==="log"&&t===floor?"≈0":(isPaper?fmt(t,1):(isLer?t+"%":String(t)))); }
  for(const t of [0,10,20,30,40,50,60,70,80,90]){ el("line",{x1:x(t),x2:x(t),y1:m.t,y2:H-m.b},g); txt(el("text",{x:x(t),y:H-m.b+20,"text-anchor":"middle",class:"tick"},svg), t+"%"); }
  const ax = el("g",{class:"axis"},svg); el("line",{x1:m.l,x2:W-m.r,y1:H-m.b,y2:H-m.b},ax); el("line",{x1:m.l,x2:m.l,y1:m.t,y2:H-m.b},ax);
  txt(el("text",{x:(m.l+W-m.r)/2,y:H-10,"text-anchor":"middle",class:"axlabel"},svg),"coverage: share of syndrome removed before Global MWPM");
  txt(el("text",{x:16,y:(m.t+H-m.b)/2,"text-anchor":"middle",class:"axlabel",transform:`rotate(-90 16 ${(m.t+H-m.b)/2})`},svg), isPaper?"logical error per patch per round, ×10⁻³  (the paper's unit)":(yMode==="lerobs"?"logical error rate per observable":(isLer?"logical error rate per shot (any of "+DATA.num_observables+" observables)":"regression vs Global MWPM, pp")));
  if(isLer){ el("line",{x1:m.l,x2:W-m.r,y1:y(gref),y2:y(gref),stroke:"var(--ink-2)","stroke-width":1,"stroke-dasharray":"none"},svg); txt(el("text",{x:W-m.r-6,y:y(gref)-6,"text-anchor":"end",class:"plabel small"},svg),"Global MWPM alone  "+fmt(gref)+(isPaper?"×10⁻³":"%")); }
  const pset = paretoOf(P.filter(p=>p[key]!=null), key);
  const pareto = P.filter(p=>pset.has(p.name)).sort((a,b)=>a.coverage-b.coverage);
  el("path",{class:"front",d:pareto.map((p,i)=>(i?"L":"M")+x(p.coverage)+" "+y(p[key])).join(" ")},svg);
  const pts = el("g",{},svg);
  for(const p of P.filter(p=>!pset.has(p.name)&&p[key]!=null)) el("circle",{class:"pt dim "+(p.interior?"s1":"s2"),cx:x(p.coverage),cy:y(p[key]),r:4},pts);
  for(const p of pareto){ el("circle",{class:"pt",cx:x(p.coverage),cy:y(p[key]),r:7.5,fill:"var(--ink)"},pts); el("circle",{class:"pt "+(p.interior?"s1":"s2"),cx:x(p.coverage),cy:y(p[key]),r:4.5},pts); }
  const stair = P.filter(p=>p.distinct).sort((a,b)=>b.coverage-a.coverage);
  const labelled = stair.filter(p=>p.name==="everything committed" || p.name==="interior only" || p.name==="size ≤ 2, interior only" || p===stair.find(q=>q.rd<=1.1) || p===stair.find(q=>q.rd<=0.25) || p===stair.find(q=>q.hi<=0.05));
  const placed=[];
  const val = p => isPaper ? fmt(p.paper)+"×10⁻³ · ×"+fmt(p.mult,3) : (isLer ? fmt(p[key])+"%" : "+"+fmt(p.rd)+" pp");
  for(const p of labelled){ if(p[key]==null) continue; const left = p.coverage>=45; let ly=y(p[key])-14, lx=x(p.coverage)+(left?-12:12);
    while(placed.some(q=>Math.abs(q[1]-ly)<15&&Math.abs(q[0]-lx)<260)) ly-=15; placed.push([lx,ly]);
    el("line",{x1:x(p.coverage),y1:y(p[key])-8,x2:lx+(left?4:-4),y2:ly+3,stroke:"var(--axis)","stroke-width":1},svg);
    txt(el("text",{x:lx,y:ly,class:"plabel","text-anchor":left?"end":"start"},svg), p.name+"  "+val(p)); }
  const tip = document.getElementById("tip"), card = document.getElementById("frontier-card");
  const hits = el("g",{},svg);
  for(const p of P){ if(p[key]==null) continue; const h = el("circle",{class:"hit",cx:x(p.coverage),cy:y(p[key]),r:12,tabindex:0,role:"button","aria-label":p.name},hits);
    const show = ev=>{ tip.innerHTML=""; const v=document.createElement("div"); v.className="v"; v.textContent=p.name; tip.appendChild(v);
      const rows=[["coverage",fmt(p.coverage,1)+"%"],["LER per patch-round (paper unit)",fmt(p.paper,3)+"×10⁻³ · ×"+fmt(p.mult,3)],["paired 95% CI, per patch-round",fmt(p.paper_lo,3)+" to "+fmt(p.paper_hi,3)+"×10⁻³"],["logical error rate per shot",fmt(p.ler)+"%"]]; if(p.ler_obs!=null) rows.push(["logical error rate per observable",fmt(p.ler_obs)+"%"]);
      rows.push(["regression vs Global",(p.rd>=0?"+":"")+fmt(p.rd)+" pp"],["95% CI",fmt(p.lo)+" to "+fmt(p.hi)],["regressions / recoveries",p.regressions.toLocaleString()+" / "+p.recoveries.toLocaleString()],["residual events per shot",fmt(p.residual,0)],["committed components",p.components.toLocaleString()],["Pareto-optimal",pset.has(p.name)?"yes":"no"]);
      for(const [k,val] of rows){ const r=document.createElement("div"); r.className="r"; const a=document.createElement("span"); a.textContent=k; const b=document.createElement("b"); b.textContent=val; r.append(a,b); tip.appendChild(r); }
      tip.style.display="block"; const rect=card.getBoundingClientRect(), sr=svg.getBoundingClientRect(); const px=sr.left-rect.left+(x(p.coverage)/W)*sr.width, py=sr.top-rect.top+(y(p[key])/H)*sr.height; tip.style.left=Math.min(px+14, rect.width-240)+"px"; tip.style.top=Math.max(8, py-10)+"px"; };
    h.addEventListener("pointerenter",show); h.addEventListener("focus",show); h.addEventListener("pointerleave",()=>tip.style.display="none"); h.addEventListener("blur",()=>tip.style.display="none"); }
}
for(const b of document.querySelectorAll("#ymodes button")){ if(b.dataset.mode==="lerobs" && DATA.global_ler_obs!=null) b.hidden=false;
  b.addEventListener("click",()=>{ yMode=b.dataset.mode; for(const o of document.querySelectorAll("#ymodes button")) o.setAttribute("aria-pressed", o===b?"true":"false"); drawFrontier(); }); }
drawFrontier();

/* culprit chart */
(function(){
  const host=document.getElementById("culprit"); const W=960,H=400,m={t:16,r:130,b:52,l:64};
  const svg=el("svg",{viewBox:`0 0 ${W} ${H}`,role:"img","aria-label":"Culprit rate by cluster size for interior and wall-touching clusters"},host);
  const bands=DATA.culprit; const floor=0.03, top=60;
  const x=i=>m.l+(i+0.5)/bands.length*(W-m.l-m.r);
  const y=r=>{const v=Math.max(r,floor); return m.t+(1-(Math.log10(v)-Math.log10(floor))/(Math.log10(top)-Math.log10(floor)))*(H-m.t-m.b);};
  const g=el("g",{class:"grid"},svg);
  for(const t of [0.03,0.1,0.3,1,3,10,30]){ el("line",{x1:m.l,x2:W-m.r,y1:y(t),y2:y(t)},g); txt(el("text",{x:m.l-10,y:y(t)+4,"text-anchor":"end",class:"tick"},svg), t+"%"); }
  const ax=el("g",{class:"axis"},svg); el("line",{x1:m.l,x2:W-m.r,y1:H-m.b,y2:H-m.b},ax); el("line",{x1:m.l,x2:m.l,y1:m.t,y2:H-m.b},ax);
  bands.forEach((b,i)=>txt(el("text",{x:x(i),y:H-m.b+20,"text-anchor":"middle",class:"tick"},svg), b.band));
  txt(el("text",{x:(m.l+W-m.r)/2,y:H-10,"text-anchor":"middle",class:"axlabel"},svg),"cluster size, defects");
  txt(el("text",{x:16,y:(m.t+H-m.b)/2,"text-anchor":"middle",class:"axlabel",transform:`rotate(-90 16 ${(m.t+H-m.b)/2})`},svg),"culprit rate among committed components");
  const tip=document.getElementById("tip2"), card=host.parentElement;
  for(const [key,cls,dx] of [["interior","s1",-9],["wall","s2",9]]){
    const pts=bands.map((b,i)=>b[key]?{i,cx:x(i)+dx,...b[key]}:null).filter(Boolean);
    el("path",{class:"front",d:pts.map((p,k)=>(k?"L":"M")+p.cx+" "+y(p.rate)).join(" "),style:`stroke:var(--${cls});stroke-width:2`},svg);
    for(const p of pts){ el("line",{class:"ci "+cls,x1:p.cx,x2:p.cx,y1:y(p.ci[0]),y2:y(p.ci[1])},svg); el("circle",{class:"pt "+cls,cx:p.cx,cy:y(p.rate),r:4.5},svg);
      const h=el("circle",{class:"hit",cx:p.cx,cy:y(p.rate),r:12,tabindex:0,role:"button","aria-label":key+" size "+bands[p.i].band},svg);
      const show=()=>{ tip.innerHTML=""; const v=document.createElement("div"); v.className="v"; v.textContent=fmt(p.rate)+"% culprits"; tip.appendChild(v);
        for(const [k,val] of [[key+" clusters, size "+bands[p.i].band,""],["95% CI",fmt(p.ci[0])+" to "+fmt(p.ci[1])+"%"],["components in regressed shots",p.n.toLocaleString()]]){ const r=document.createElement("div"); r.className="r"; const a=document.createElement("span"); a.textContent=k; const bb=document.createElement("b"); bb.textContent=val; r.append(a,bb); tip.appendChild(r);}
        tip.style.display="block"; const rect=card.getBoundingClientRect(), sr=svg.getBoundingClientRect(); const px=sr.left-rect.left+(p.cx/W)*sr.width, py=sr.top-rect.top+(y(p.rate)/H)*sr.height; tip.style.left=Math.min(px+14,rect.width-240)+"px"; tip.style.top=Math.max(8,py-10)+"px"; };
      h.addEventListener("pointerenter",show); h.addEventListener("focus",show); h.addEventListener("pointerleave",()=>tip.style.display="none"); h.addEventListener("blur",()=>tip.style.display="none"); }
    const last=pts[pts.length-1]; txt(el("text",{x:last.cx+10,y:y(last.rate)+4,class:"plabel small"},svg), key==="interior"?"interior "+fmt(last.rate,1)+"%":"wall "+fmt(last.rate,1)+"%");
  }
})();

/* lessons */
(function(){
  const L=DATA.lessons, P=DATA.points, c=DATA.culprit;
  const pair=c.find(b=>b.band==="2-2").interior, big=c[c.length-1].interior, wallpair=c.find(b=>b.band==="2-2").wall, single=c.find(b=>b.band==="1-1").wall;
  const items=[
    ["Size is a steep, monotone risk factor", `Among interior clusters the culprit rate climbs from ${fmt(pair.rate)}% for adjacent pairs to ${fmt(big.rate,1)}% for clusters of 13 or more, a factor of about ${Math.round(big.rate/pair.rate)}, with non-overlapping intervals band to band. A large cluster has many internal pairings and the peeling picks one; the alternatives are exactly what MWPM weighs.`],
    ["Wall contact multiplies the risk at every size", `A wall-touching pair is a culprit ${fmt(wallpair.rate)}% of the time against ${fmt(pair.rate)}% for an interior pair, and a lone defect committed to the real wall (${fmt(single.rate)}%) is as risky as an interior cluster of five or six. Overall wall clusters run ${fmt(L.wall_rate)}% against ${fmt(L.interior_rate)}% interior: the wall choice is the yoke decision made blind.`],
    ["Coverage and accuracy trade along a smooth frontier", `There is no plateau of free coverage. The frontier runs from ${fmt(P.find(p=>p.name==="everything committed").coverage,0)}% coverage at +${fmt(P.find(p=>p.name==="everything committed").rd,1)} pp down to about 29% at zero, and the cheapest interior-only rule, adjacent pairs, sits at +${fmt(P.find(p=>p.name==="size ≤ 2, interior only").rd)} pp for ${fmt(P.find(p=>p.name==="size ≤ 2, interior only").coverage,0)}%. Adding a margin threshold to a size rule moves along the frontier, not off it.`],
    ["A quarter of regressions are interactions", `${L.no_single.toLocaleString()} of ${L.regressed.toLocaleString()} regressed shots (${fmt(100*L.no_single/L.regressed,0)}%) have no single-component culprit: two or more commits together push MWPM to a wrong pairing. No rule applied one cluster at a time can catch them, which is the floor the margin-based rules approach.`]
  ];
  const root=document.getElementById("lessons");
  for(const [h,p] of items){ const d=document.createElement("div"); d.className="lesson"; const a=document.createElement("h3"); a.textContent=h; const b=document.createElement("p"); b.textContent=p; d.append(a,b); root.appendChild(d); }
})();

/* tables */
(function(){
  const P=DATA.points; const tb=document.querySelector("#ptable tbody");
  for(const p of P.filter(p=>p.distinct).sort((a,b)=>b.coverage-a.coverage)){ const tr=document.createElement("tr");
    for(const [v,cls] of [[p.name,"rule"],[fmt(p.coverage,1)+"%","n"],[fmt(p.ler)+"%","n"],[(p.rd>=0?"+":"")+fmt(p.rd),"n"],[fmt(p.lo)+" to "+fmt(p.hi),"n"],[p.regressions.toLocaleString(),"n"],[p.recoveries.toLocaleString(),"n"],[fmt(p.residual,0),"n"]]){ const td=document.createElement("td"); td.className=cls; td.textContent=v; tr.appendChild(td);} tb.appendChild(tr); }
  const ta=document.querySelector("#alltable tbody");
  for(const p of P.slice().sort((a,b)=>b.coverage-a.coverage)){ const tr=document.createElement("tr");
    for(const [v,cls] of [[p.name,"rule"],[fmt(p.coverage,1)+"%","n"],[(p.rd>=0?"+":"")+fmt(p.rd),"n"],[fmt(p.lo)+" to "+fmt(p.hi),"n"],[p.failures.toLocaleString(),"n"],[fmt(p.residual,0),"n"]]){ const td=document.createElement("td"); td.className=cls; td.textContent=v; tr.appendChild(td);} ta.appendChild(tr); }
})();
</script>
"""
html = html.replace("{DATA_JSON}", json.dumps(data)).replace("{NRULES}", str(len(points))).replace("{REGRESSED}", f"{att['regressed_shots']:,}").replace("{GFAIL}", f"{gfail:,}").replace("{SHOTS}", f"{N:,}")
out_html.write_text(html)
print(f"pareto set ({len(pareto)}):")
for p in pareto: print(f"  {p['name']:38s} cov {p['coverage']:5.1f}%  rd {p['rd']:+.2f} [{p['lo']:+.2f},{p['hi']:+.2f}]")
print("wrote", out_html)
