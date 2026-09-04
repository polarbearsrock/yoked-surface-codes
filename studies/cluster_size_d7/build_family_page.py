"""Build the two-ladder page (interior-only size caps, interior-only margin thresholds) from frontier_interior.json."""
import json, math, sys
from pathlib import Path

cell = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("out/cluster_size_study_d7/d7_p0.003_100k")
out_html = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("out/cluster_size_study_d7/ladders.html")
fr = json.load(open(cell / "frontier_interior.json"))
N, O = fr["shots"], fr["num_observables"]; gfail, gfail_obs = fr["global_failures"], fr["global_observable_failures"]
# The paper's unit: logical error per patch per round, via the same sinter conversion step3_plot uses
# (pieces = patches * rounds, values = observables).  Reported here in units of 1e-3.
import sinter
prov = json.load(open(cell / "provenance.json")); PATCHES, ROUNDS = prov["cell"]["patches"], prov["cell"]["rounds"]; PIECES = PATCHES * ROUNDS
def paper(p_shot): return sinter.shot_error_rate_to_piece_error_rate(p_shot, pieces=PIECES, values=O)
PAPER_NONE = paper(gfail / N)

def name(r):
    if r["family"] == "none": return "nothing committed"
    if r["family"] == "size": return "interior, any size" if r["value"] is None else f"interior, size ≤ {r['value']}"
    if r["family"] == "diameter": return "interior, any diameter" if r["value"] is None else f"interior, diameter ≤ {r['value']}"
    return "interior, any margin" if r["value"] == 0 else f"interior, margin > {r['value']:g}"

rows = []
for r in fr["rows"]:
    rows.append(dict(family=r["family"], value=r["value"], name=name(r), coverage=r["coverage_pct"],
                     rd=r["risk_difference_pp"], lo=r["ci_pp"][0], hi=r["ci_pp"][1],
                     rd_obs=r["observable_risk_difference_pp"], lo_obs=r["observable_ci_pp"][0], hi_obs=r["observable_ci_pp"][1],
                     ler=100 * r["failures"] / N, ler_obs=100 * r["observable_failures"] / (N * O),
                     paper=1e3 * paper(r["failures"] / N), paper_lo=1e3 * paper(gfail / N + r["ci_pp"][0] / 100), paper_hi=1e3 * paper(gfail / N + r["ci_pp"][1] / 100),
                     mult=paper(r["failures"] / N) / PAPER_NONE,
                     failures=r["failures"], regressions=r["regressions"], recoveries=r["recoveries"],
                     residual=r["residual_events_per_shot"], components=r["components"], committed_defects=r["committed_defects"]))
none = next(r for r in rows if r["family"] == "none")
# Coverage ceiling: where the lane-owned defects go under interior-only rules (from the cell's component records).
import numpy as np
z = np.load(cell / "cell.npz"); size_a, comm_a, port_a, bdry_a = z["component_size"], z["component_committed"], z["component_port"], z["component_boundary"]
L = fr["lane_owned_detector_events"]
ceiling = dict(committed_interior=100 * float(size_a[comm_a & ~bdry_a].sum()) / L, committed_wall=100 * float(size_a[comm_a & bdry_a].sum()) / L,
               deferred_port=100 * float(size_a[port_a].sum()) / L, deferred_tie=100 * float(size_a[~comm_a & ~port_a].sum()) / L)
# Interior clusters have even parity, so odd caps duplicate the even cap below them; thresholds that
# commit nothing collapse onto the origin.  Keep one point per distinct rule.
def thin(ladder, key):
    """Keep steps that add at least 0.1% of lane-owned defects over the last kept step, plus the last step."""
    L = fr["lane_owned_detector_events"]; out, last = [], -1e18
    ordered = sorted(ladder, key=key)
    for i, r in enumerate(ordered):
        if r["committed_defects"] - last >= 0.001 * L or i == len(ordered) - 1:
            if out and i == len(ordered) - 1 and r["committed_defects"] - last < 0.001 * L: out[-1] = r
            else: out.append(r)
            last = r["committed_defects"]
    return out
size_rows = thin([r for r in rows if r["family"] == "size" and (r["value"] is None or r["value"] % 2 == 0)], key=lambda r: (r["value"] is None, r["value"] or 0))
margin_rows = [r for r in rows if r["family"] == "margin" and r["coverage"] > 0]
# Diameter caps beyond the largest interior cluster add nothing; keep caps that add committed defects, plus "any".
diameter_rows = thin([r for r in rows if r["family"] == "diameter"], key=lambda r: (r["value"] is None, r["value"] or 0))

def marginal(ladder):
    """Net regressions per 1,000 added committed defects between consecutive ladder steps (per shot / per observable)."""
    out, prev = [], none
    for r in ladder:
        dd = r["committed_defects"] - prev["committed_defects"]
        dn = (r["regressions"] - r["recoveries"]) - (prev["regressions"] - prev["recoveries"])
        r["step_defects"] = dd; r["step_per_1000"] = (1000 * dn / dd) if dd else None
        prev = r
    return ladder
size_rows = marginal(sorted(size_rows, key=lambda r: (r["value"] is None, r["value"] or 0)))
margin_rows = marginal(sorted(margin_rows, key=lambda r: -r["value"]))  # from strictest (least coverage) to loosest
diameter_rows = marginal(diameter_rows)

timing = json.load(open(cell / "l2_timing.json")) if (cell / "l2_timing.json").exists() else None
def _tkey(r): return f"{r['family']}:{r['value']}"
timing_rows = {_tkey(r): r for r in timing["rows"]} if timing else {}
def attach_timing(ladder, family):
    for r in ladder:
        t = timing_rows.get(f"{family}:{r['value']}")
        r["t"] = None if t is None else dict(mean=t["mean_ms"], p99=t["p99_ms"], max=t["max_ms"], rel_mean=100*t["rel_mean"], rel_p99=100*t["rel_p99"], rel_max=100*t["rel_max"], events=t["mean_events"])
attach_timing(size_rows, "size"); attach_timing(margin_rows, "margin"); attach_timing(diameter_rows, "diameter")
tnone = timing_rows.get("none:None"); none["t"] = None if tnone is None else dict(mean=tnone["mean_ms"], p99=tnone["p99_ms"], max=tnone["max_ms"], rel_mean=100.0, rel_p99=100.0, rel_max=100.0, events=tnone["mean_events"])
validation = json.load(open(cell / "l2_validation.json")) if (cell / "l2_validation.json").exists() else None
proxy_note = None
if validation:
    c0 = validation["correlations"]["none:None"]; rho = sorted((v["spearman"], k) for k, v in c0.items()); slow = validation["baseline_slowest_1pct"]
    proxy_note = (f" Structural work proxies taken from the sparse blossom paper (matched-edge count, matched weight, longest matched path, region volume, largest region, sum of squared region sizes) "
                  f"track the mean across rules (Spearman {validation['across_rules']['q_max']['mean_spearman']:.2f}) but not which shots are slow: shot by shot their Spearman with measured time is "
                  f"{rho[0][0]:.2f} to {rho[-1][0]:.2f}, and the slowest 1% of baseline shots carry the same proxy values as the rest ("
                  + ", ".join(f"{k} ×{v['slow_mean']/v['rest_mean']:.2f}" for k, v in slow.items() if k in ("q_max", "path_w_max", "volume")) + "). The axis is therefore measured time.")
data = dict(shots=N, num_observables=O, timing_note=(timing["host_note"] if timing else None), proxy_note=proxy_note, global_ler=100 * gfail / N, global_ler_obs=100 * gfail_obs / (N * O),
            global_paper=1e3 * PAPER_NONE, patches=PATCHES, rounds=ROUNDS, pieces=PIECES, paper_fit_p001=6 * 28 * 8 ** -7 / 500,
            interior_share=fr["interior_component_share"], none=none, size=size_rows, margin=margin_rows, diameter=diameter_rows,
            diameter_histogram=fr.get("interior_diameter_histogram"), size_histogram=fr.get("interior_size_histogram"), ceiling=ceiling)

html = r"""<title>Size, Margin, and Diameter Ladders</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{color-scheme:light;--page:#f7f7f4;--surface:#fcfcfb;--ink:#0b0b0b;--ink-2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);
  --s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--accent:#2a78d6;--tip:#ffffff;--goal:#16a34a;--font:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;--mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#ffffff;--ink-2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--accent:#3987e5;--tip:#242423;--goal:#4ade80}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#ffffff;--ink-2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--accent:#3987e5;--tip:#242423;--goal:#4ade80}
body{margin:0;background:var(--page);color:var(--ink);font-family:var(--font);font-size:15px;line-height:1.5}
main{max-width:1040px;margin:0 auto;padding:40px 24px 64px;display:flex;flex-direction:column;gap:36px}
h1{font-size:30px;font-weight:600;letter-spacing:-.01em;line-height:1.15;margin:0 0 8px;text-wrap:balance}
h2{font-size:19px;font-weight:600;margin:0 0 4px}
p{max-width:68ch;margin:0}
.lede{color:var(--ink-2);font-size:16px;max-width:70ch}
.eyebrow{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin-bottom:12px;font-weight:500}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px}
.tile{background:var(--surface);border:1px solid var(--ring);border-radius:6px;padding:16px 18px}
.tile .label{font-size:13px;color:var(--ink-2)}
.tile .value{font-size:28px;font-weight:600;margin-top:4px;line-height:1.1}
.tile .sub{font-size:13px;color:var(--muted);margin-top:6px}
.controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap;font-size:13px;color:var(--ink-2)}
.controls button{font:inherit;background:var(--surface);color:var(--ink);border:1px solid var(--ring);border-radius:4px;padding:4px 10px;cursor:pointer}
.controls button[aria-pressed="true"]{background:var(--ink);color:var(--surface);border-color:var(--ink)}
.controls button:focus-visible,.hit:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
section{display:flex;flex-direction:column;gap:12px}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:6px;padding:20px 20px 12px;position:relative}
svg.chart{display:block;width:100%;height:auto;font-family:var(--font)}
.legend svg{width:16px;height:16px;display:inline-block;flex:none}
.grid line{stroke:var(--grid);stroke-width:1}
.axis line{stroke:var(--axis);stroke-width:1}
.tick{fill:var(--muted);font-size:12px;font-variant-numeric:tabular-nums}
.axlabel{fill:var(--ink-2);font-size:13px}
.pt{stroke:var(--surface);stroke-width:2}
.front{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.plabel{fill:var(--ink);font-size:12px;font-family:var(--mono)}
.legend{display:flex;gap:6px 22px;flex-wrap:wrap;font-size:13px;color:var(--muted);margin:6px 0 4px}
.legend span{display:inline-flex;align-items:center;gap:8px}
.legend b{color:var(--ink);font-weight:500;font-family:var(--mono);font-size:12.5px;white-space:nowrap}
.legend em{font-style:normal;font-variant-numeric:tabular-nums;white-space:nowrap}
.plabel.small{fill:var(--ink-2)}
.hit{fill:transparent;cursor:crosshair}
.tooltip{position:absolute;pointer-events:none;background:var(--tip);color:var(--ink);border:1px solid var(--ring);border-radius:6px;padding:10px 12px;font-size:13px;box-shadow:0 6px 24px rgba(0,0,0,.14);min-width:230px;display:none;z-index:2}
.tooltip .v{font-weight:600;font-size:15px}
.tooltip .r{display:flex;justify-content:space-between;gap:16px;color:var(--ink-2)}
.tooltip .r b{color:var(--ink);font-weight:500;font-variant-numeric:tabular-nums}
table{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}
th{text-align:left;font-weight:500;color:var(--ink-2);font-size:12px;letter-spacing:.04em;text-transform:uppercase;padding:8px 10px;border-bottom:1px solid var(--axis)}
td{padding:7px 10px;border-bottom:1px solid var(--grid)}
td.n,th.n{text-align:right}
td.rule{font-family:var(--mono);font-size:13px;white-space:nowrap}
.scroll{overflow-x:auto}
.note{font-size:13px;color:var(--muted);max-width:none}
.key{width:10px;height:10px;border-radius:50%;display:inline-block}
h3.ct{margin:0 0 6px;font-size:14px;font-weight:500;color:var(--ink-2)}
.read{display:grid;grid-template-columns:repeat(auto-fit,minmax(225px,1fr));gap:16px}
.read div{background:var(--surface);border:1px solid var(--ring);border-radius:6px;padding:16px 18px}
.read h3{margin:0 0 6px;font-size:15px;font-weight:600}
.read p{font-size:14px;color:var(--ink-2);max-width:none}
details summary{cursor:pointer;color:var(--ink-2);font-size:14px}
</style>
<main>
<header>
  <div class="eyebrow">Port-wall Patch-UF frontend · d = 7, SI1000 p = 0.003 · 100,000 paired shots · interior clusters only · non-claim-bearing</div>
  <h1>Size, margin, and diameter ladders</h1>
  <p class="lede">Three one-parameter commit rules, each restricted to clusters that touched no wall. The size ladder commits every interior cluster of at most k defects. The margin ladder commits every interior cluster whose margin exceeds a threshold. The diameter ladder commits every interior cluster whose spanning forest is at most h hops across. Each ladder is a Pareto frontier of coverage against error in its own right, every step on it optimal for its family; all three are drawn on the same axes so the trades can be compared directly. Hover any point for its numbers; the caption under each chart lists a few reference steps.</p>
</header>

<div class="tiles" id="tiles"></div>

<p class="note" id="unit-note"></p>

<div class="controls" role="group" aria-label="Vertical axis" id="ymodes"><span>y-axis</span><button data-mode="paper" aria-pressed="true">LER per patch-round (paper unit)</button><button data-mode="log" aria-pressed="false">regression, log</button><button data-mode="lin" aria-pressed="false">regression, linear</button><button data-mode="ler" aria-pressed="false">LER per shot</button><button data-mode="lerobs" aria-pressed="false">LER per observable</button></div>

<section>
  <h2>Cluster-size ladder</h2>
  <p>Interior clusters with at most k defects are committed, for even k from 2 upward. Interior clusters always hold an even number of defects, so an odd cap is the same rule as the even cap below it, and caps that add under 0.1% of coverage over the previous step are folded into the last step shown. Coverage is the share of lane-owned detector events removed before Global MWPM.</p>
  <div class="card" id="size-card"><div id="size"></div><div class="legend" id="legend-size"></div><div class="tooltip" id="tip-size"></div></div>
</section>

<section>
  <h2>Margin ladder</h2>
  <p>Interior clusters whose margin exceeds τ are committed, for τ from 0 to 1.8 in steps of 0.05; above 1.8 no cluster qualifies. The margin is the smallest remaining slack on any unused edge touching the cluster, in log-likelihood weight units; τ = 0 defers only exact ties.</p>
  <div class="card" id="margin-card"><div id="margin"></div><div class="legend" id="legend-margin"></div><div class="tooltip" id="tip-margin"></div></div>
</section>

<section>
  <h2>Diameter ladder</h2>
  <p>Interior clusters whose forest diameter is at most h hops are committed, for every h that adds at least 0.1% of coverage; the last step shown is the largest cap, which is the same rule as "any diameter". The diameter is the longest path between two vertices of the cluster along the edges that merged it; h = 1 is an adjacent pair.</p>
  <div class="card" id="diameter-card"><div id="diameter"></div><div class="legend" id="legend-diameter"></div><div class="tooltip" id="tip-diameter"></div></div>
</section>

<p class="note" id="ceiling"></p>

<section id="l2-section" hidden>
  <h2>The same ladders against L2 time</h2>
  <p>Coverage counts detector events removed; it is not a latency. Here the horizontal axis is the Global MWPM decode time on each rule's residual, as a percentage of the time on the untouched syndrome, so 100% is "the frontend saved nothing". Three statistics over the 100,000 shots are shown as separate charts: the mean, the 99th percentile, and the maximum, since a deadline is set by the tail. Vertical axis as selected above.</p>
  <div class="legend" style="margin:0"><span><i class="key" style="background:var(--s1)"></i>size ladder</span><span><i class="key" style="background:var(--s2)"></i>margin ladder</span><span><i class="key" style="background:var(--s3)"></i>diameter ladder</span><span><i class="key" style="background:var(--muted)"></i>nothing committed (100%)</span><span><svg width="15" height="15" viewBox="-10 -10 20 20" aria-hidden="true"><path d="M0,-9.5 L2.7,-3.6 L9,-2.9 L4.4,1.4 L5.6,7.7 L0,4.6 L-5.6,7.7 L-4.4,1.4 L-9,-2.9 L-2.7,-3.6 Z" fill="var(--goal)"/></svg>goal: MWPM's accuracy at the fastest residual</span></div>
  <div class="card"><h3 class="ct">x = mean L2 decode time</h3><div id="l2-mean"></div><div class="tooltip" id="tip-l2-mean"></div></div>
  <div class="card"><h3 class="ct">x = 99th-percentile L2 decode time</h3><div id="l2-p99"></div><div class="tooltip" id="tip-l2-p99"></div></div>
  <div class="card"><h3 class="ct">x = maximum L2 decode time</h3><div id="l2-max"></div><div class="tooltip" id="tip-l2-max"></div></div>
  <p class="note" id="l2-note"></p>
  <div class="scroll"><table id="l2-table"><thead><tr><th>rule</th><th class="n">coverage</th><th class="n">regression, pp</th><th class="n">mean ms</th><th class="n">p99 ms</th><th class="n">max ms</th><th class="n">mean %</th><th class="n">p99 %</th><th class="n">max %</th><th class="n">residual events / shot</th></tr></thead><tbody></tbody></table></div>
</section>


<section>
  <h2>Size ladder, step by step</h2>
  <div class="scroll"><table id="size-table"><thead><tr><th>rule</th><th class="n">coverage</th><th class="n">LER per shot</th><th class="n">LER per obs.</th><th class="n">per patch-round, ×10⁻³</th><th class="n">× baseline</th><th class="n">regression, pp</th><th class="n">95% CI</th><th class="n">added defects at this step</th><th class="n">net regressions per 1,000 added</th></tr></thead><tbody></tbody></table></div>
</section>
<section>
  <h2>Diameter ladder, step by step</h2>
  <div class="scroll"><table id="diameter-table"><thead><tr><th>rule</th><th class="n">coverage</th><th class="n">LER per shot</th><th class="n">LER per obs.</th><th class="n">per patch-round, ×10⁻³</th><th class="n">× baseline</th><th class="n">regression, pp</th><th class="n">95% CI</th><th class="n">added defects at this step</th><th class="n">net regressions per 1,000 added</th></tr></thead><tbody></tbody></table></div>
</section>
<section>
  <h2>Margin ladder, step by step</h2>
  <p class="note">Listed from the strictest threshold to the loosest, so each step adds coverage.</p>
  <div class="scroll"><table id="margin-table"><thead><tr><th>rule</th><th class="n">coverage</th><th class="n">LER per shot</th><th class="n">LER per obs.</th><th class="n">per patch-round, ×10⁻³</th><th class="n">× baseline</th><th class="n">regression, pp</th><th class="n">95% CI</th><th class="n">added defects at this step</th><th class="n">net regressions per 1,000 added</th></tr></thead><tbody></tbody></table></div>
  <p class="note">Method: the union-find run does not depend on the commit rule, so each rule's residual is rebuilt from the retained committed components and Global MWPM is decoded once over all {SHOTS} shots. Intervals are Wald on the paired difference against Global MWPM, which failed {GFAIL} shots ({GFAILP}%) and {GFAILOBS}% of observables.</p>
</section>
</main>
<script>
const DATA = {DATA_JSON};
const fmt = (x, d=2) => x.toLocaleString(undefined, {minimumFractionDigits:d, maximumFractionDigits:d});
const svgNS = "http://www.w3.org/2000/svg";
function el(tag, attrs={}, parent=null){ const e=document.createElementNS(svgNS, tag); for(const [k,v] of Object.entries(attrs)) e.setAttribute(k, v); if(parent) parent.appendChild(e); return e; }
function txt(e, s){ e.textContent = s; return e; }
function star(cx,cy,r){let d="";for(let i=0;i<10;i++){const a=-Math.PI/2+i*Math.PI/5, rr=i%2?r*0.48:r; d+=(i?"L":"M")+(cx+rr*Math.cos(a)).toFixed(1)+" "+(cy+rr*Math.sin(a)).toFixed(1);} return d+"Z";}
let yMode = "paper";
const SHAPES = ["square","diamond","triangle","triangle-down","star"];
function shapePath(kind, cx, cy, r){
  if(kind==="square") return `M${cx-r} ${cy-r}h${2*r}v${2*r}h${-2*r}z`;
  if(kind==="diamond") return `M${cx} ${cy-r*1.25}L${cx+r*1.25} ${cy}L${cx} ${cy+r*1.25}L${cx-r*1.25} ${cy}z`;
  if(kind==="triangle") return `M${cx} ${cy-r*1.2}L${cx+r*1.15} ${cy+r*0.85}L${cx-r*1.15} ${cy+r*0.85}z`;
  if(kind==="triangle-down") return `M${cx} ${cy+r*1.2}L${cx+r*1.15} ${cy-r*0.85}L${cx-r*1.15} ${cy-r*0.85}z`;
  let d=""; for(let i=0;i<10;i++){ const a=-Math.PI/2+i*Math.PI/5, rr=i%2?r*0.55:r*1.35; d+=(i?"L":"M")+(cx+rr*Math.cos(a))+" "+(cy+rr*Math.sin(a)); } return d+"z";
}
function paretoSet(rows, key){ const s=rows.slice().sort((a,b)=>(b.coverage-a.coverage)||(a[key]-b[key])); const out=new Set(); let best=Infinity; for(const p of s){ if(p[key]<best-1e-9){ out.add(p.name); best=p[key]; } } return out; }

function drawLadder(id, rows, cls, labelPick, tipId){
  const host=document.getElementById(id); host.innerHTML="";
  const W=960,H=440,m={t:18,r:30,b:52,l:64};
  const isPaper=yMode==="paper"; const isLer=isPaper||yMode==="ler"||yMode==="lerobs"; const key=isPaper?"paper":(yMode==="lerobs"?"ler_obs":(isLer?"ler":"rd"));
  const gref=isPaper?DATA.global_paper:(yMode==="lerobs"?DATA.global_ler_obs:DATA.global_ler);
  const all=[DATA.none].concat(rows); const vals=all.map(p=>p[key]); const vmax=Math.max(...vals);
  const floor=0.01, top=20; const lerLo=isPaper?Math.floor((Math.min(...vals,gref)-0.02)*10)/10:Math.floor(Math.min(...vals,gref)-0.3), lerHi=isPaper?Math.ceil((vmax+0.02)*10)/10:Math.ceil(vmax+0.3);
  const svg=el("svg",{class:"chart",viewBox:`0 0 ${W} ${H}`,role:"img","aria-label":id+" ladder: coverage versus "+(isLer?"logical error rate":"regression")},host);
  const x=c=>m.l+(c/90)*(W-m.l-m.r);
  const y= yMode==="log" ? (r=>{const v=Math.max(r,floor); return m.t+(1-(Math.log10(v)-Math.log10(floor))/(Math.log10(top)-Math.log10(floor)))*(H-m.t-m.b);})
        : yMode==="lin" ? (r=>m.t+(1-Math.max(r,0)/12)*(H-m.t-m.b))
        : (r=>m.t+(1-(r-lerLo)/(lerHi-lerLo))*(H-m.t-m.b));
  const g=el("g",{class:"grid"},svg);
  let yticks; if(yMode==="log") yticks=[0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20]; else if(yMode==="lin") yticks=[0,2,4,6,8,10,12];
  else { const span=lerHi-lerLo; const step=isPaper?0.1:(span>12?2:(span>4?1:0.5)); yticks=[]; for(let t=lerLo;t<=lerHi+1e-9;t+=step) yticks.push(+t.toFixed(2)); }
  for(const t of yticks){ el("line",{x1:m.l,x2:W-m.r,y1:y(t),y2:y(t)},g); txt(el("text",{x:m.l-10,y:y(t)+4,"text-anchor":"end",class:"tick"},svg), yMode==="log"&&t===floor?"≈0":(isPaper?fmt(t,1):(isLer?t+"%":String(t)))); }
  for(const t of [0,10,20,30,40,50,60,70,80,90]){ el("line",{x1:x(t),x2:x(t),y1:m.t,y2:H-m.b},g); txt(el("text",{x:x(t),y:H-m.b+20,"text-anchor":"middle",class:"tick"},svg), t+"%"); }
  const ax=el("g",{class:"axis"},svg); el("line",{x1:m.l,x2:W-m.r,y1:H-m.b,y2:H-m.b},ax); el("line",{x1:m.l,x2:m.l,y1:m.t,y2:H-m.b},ax);
  txt(el("text",{x:(m.l+W-m.r)/2,y:H-10,"text-anchor":"middle",class:"axlabel"},svg),"coverage: share of syndrome removed before Global MWPM");
  txt(el("text",{x:16,y:(m.t+H-m.b)/2,"text-anchor":"middle",class:"axlabel",transform:`rotate(-90 16 ${(m.t+H-m.b)/2})`},svg), isPaper?"logical error per patch per round, ×10⁻³  (the paper's unit)":(yMode==="lerobs"?"logical error rate per observable":(isLer?"logical error rate per shot":"regression vs Global MWPM, pp")));
  if(isLer){ el("line",{x1:m.l,x2:W-m.r,y1:y(gref),y2:y(gref),stroke:"var(--ink-2)","stroke-width":1},svg); txt(el("text",{x:W-m.r-6,y:y(gref)-6,"text-anchor":"end",class:"plabel small"},svg),"Global MWPM alone  "+fmt(gref)+(isPaper?"×10⁻³":"%")); }
  const ordered=all.slice().sort((a,b)=>a.coverage-b.coverage);
  // The line joins measured steps only: nothing exists between the origin and the first step, so
  // that segment is not drawn.  The origin (nothing committed) is a neutral reference dot.
  const steps=ordered.filter(p=>p.name!=="nothing committed");
  el("path",{class:"front",style:`stroke:var(--${cls})`,d:steps.map((p,i)=>(i?"L":"M")+x(p.coverage)+" "+y(p[key])).join(" ")},svg);
  // Plain dots for every step; a one-line caption under the chart names the landmark steps with
  // their numbers, so the values are reachable without hovering.
  for(const p of ordered) el("circle",{class:"pt",cx:x(p.coverage),cy:y(p[key]),r:4.5,fill:p.name==="nothing committed"?"var(--muted)":`var(--${cls})`},svg);
  const goalX=DATA.ceiling.committed_interior, goalY=isLer?gref:0; el("path",{class:"pt",d:star(x(goalX),y(goalY),8),fill:"var(--goal)"},svg);
  const val=p=>isPaper?fmt(p.paper)+"×10⁻³ · ×"+fmt(p.mult,3):(isLer?fmt(p[key])+"%":((p.rd>=0?"+":"")+fmt(p.rd)+" pp"));
  const leg=document.getElementById("legend-"+id); leg.innerHTML="";
  for(const p of rows.filter(labelPick)){ const span=document.createElement("span"); const b=document.createElement("b"); b.textContent=p.name.replace("interior, ",""); const e=document.createElement("em"); e.textContent=fmt(p.coverage,1)+"% · "+val(p); span.append(b,e); leg.appendChild(span); }
  { const span=document.createElement("span"); span.innerHTML='<svg width="15" height="15" viewBox="-10 -10 20 20" aria-hidden="true"><path d="M0,-9.5 L2.7,-3.6 L9,-2.9 L4.4,1.4 L5.6,7.7 L0,4.6 L-5.6,7.7 L-4.4,1.4 L-9,-2.9 L-2.7,-3.6 Z" fill="var(--goal)"/></svg>'; const b=document.createElement("b"); b.textContent="goal"; const e=document.createElement("em"); e.textContent="interior ceiling "+fmt(goalX,1)+"% at MWPM's accuracy"; span.append(b,e); leg.appendChild(span); }
  const tip=document.getElementById(tipId), card=host.parentElement;
  for(const p of all){ const h=el("circle",{class:"hit",cx:x(p.coverage),cy:y(p[key]),r:12,tabindex:0,role:"button","aria-label":p.name},svg);
    const show=()=>{ tip.innerHTML=""; const v=document.createElement("div"); v.className="v"; v.textContent=p.name; tip.appendChild(v);
      const rows=[["coverage",fmt(p.coverage,1)+"%"],["LER per patch-round (paper unit)",fmt(p.paper,3)+"×10⁻³ · ×"+fmt(p.mult,3)],["paired 95% CI, per patch-round",fmt(p.paper_lo,3)+" to "+fmt(p.paper_hi,3)+"×10⁻³"],["LER per shot",fmt(p.ler)+"%"],["LER per observable",fmt(p.ler_obs,3)+"%"],["regression vs Global",(p.rd>=0?"+":"")+fmt(p.rd)+" pp"],["95% CI",fmt(p.lo)+" to "+fmt(p.hi)],["regressions / recoveries",p.regressions.toLocaleString()+" / "+p.recoveries.toLocaleString()],["residual events per shot",fmt(p.residual,0)]];
      if(p.step_per_1000!=null) rows.push(["net regressions per 1,000 added defects",fmt(p.step_per_1000)]);
      for(const [k,val] of rows){ const r=document.createElement("div"); r.className="r"; const a=document.createElement("span"); a.textContent=k; const b=document.createElement("b"); b.textContent=val; r.append(a,b); tip.appendChild(r); }
      tip.style.display="block"; const rect=card.getBoundingClientRect(), sr=svg.getBoundingClientRect(); const px=sr.left-rect.left+(x(p.coverage)/W)*sr.width, py=sr.top-rect.top+(y(p[key])/H)*sr.height; tip.style.left=Math.min(px+14,rect.width-250)+"px"; tip.style.top=Math.max(8,py-10)+"px"; };
    h.addEventListener("pointerenter",show); h.addEventListener("focus",show); h.addEventListener("pointerleave",()=>tip.style.display="none"); h.addEventListener("blur",()=>tip.style.display="none"); }
}
function drawAll(){
  drawLadder("size", DATA.size, "s1", p=>[2,4,8,null].includes(p.value), "tip-size");
  drawLadder("margin", DATA.margin, "s2", p=>[0,0.5,1,1.5].includes(p.value), "tip-margin");
  drawLadder("diameter", DATA.diameter, "s3", p=>[1,2,4,8,null].includes(p.value), "tip-diameter");
}
for(const b of document.querySelectorAll("#ymodes button")) b.addEventListener("click",()=>{ yMode=b.dataset.mode; for(const o of document.querySelectorAll("#ymodes button")) o.setAttribute("aria-pressed", o===b?"true":"false"); drawAll(); drawL2All(); });
drawAll(); drawL2All();

/* L2 time charts */
function drawL2(id, stat, tipId){
  const host=document.getElementById(id); host.innerHTML="";
  const W=960,H=420,m={t:18,r:30,b:52,l:64};
  const isPaper=yMode==="paper"; const isLer=isPaper||yMode==="ler"||yMode==="lerobs"; const key=isPaper?"paper":(yMode==="lerobs"?"ler_obs":(isLer?"ler":"rd")); const gref=isPaper?DATA.global_paper:(yMode==="lerobs"?DATA.global_ler_obs:DATA.global_ler);
  const fams=[["size",DATA.size,"s1"],["margin",DATA.margin,"s2"],["diameter",DATA.diameter,"s3"]];
  const pts=fams.flatMap(([f,rows,cls])=>rows.filter(p=>p.t).map(p=>({...p,fam:f,cls}))); const allv=pts.map(p=>p[key]).concat([DATA.none[key]]);
  const floor=0.01, top=20; const lerLo=isPaper?Math.floor((Math.min(...allv,gref)-0.02)*10)/10:Math.floor(Math.min(...allv,gref)-0.3), lerHi=isPaper?Math.ceil((Math.max(...allv)+0.02)*10)/10:Math.ceil(Math.max(...allv)+0.3);
  const xs=pts.map(p=>p.t["rel_"+stat]); const xmin=Math.max(0, Math.floor((Math.min(...xs)-5)/10)*10), xmax=Math.max(100, Math.ceil((Math.max(...xs)+5)/10)*10);
  const svg=el("svg",{class:"chart",viewBox:`0 0 ${W} ${H}`,role:"img","aria-label":"ladders against "+stat+" L2 decode time"},host);
  const x=v=>m.l+((v-xmin)/(xmax-xmin))*(W-m.l-m.r);
  const y= yMode==="log" ? (r=>{const v=Math.max(r,floor); return m.t+(1-(Math.log10(v)-Math.log10(floor))/(Math.log10(top)-Math.log10(floor)))*(H-m.t-m.b);})
        : yMode==="lin" ? (r=>m.t+(1-Math.max(r,0)/12)*(H-m.t-m.b)) : (r=>m.t+(1-(r-lerLo)/(lerHi-lerLo))*(H-m.t-m.b));
  const g=el("g",{class:"grid"},svg);
  let yticks; if(yMode==="log") yticks=[0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20]; else if(yMode==="lin") yticks=[0,2,4,6,8,10,12]; else { const span=lerHi-lerLo; const step=isPaper?0.1:(span>12?2:(span>4?1:0.5)); yticks=[]; for(let t=lerLo;t<=lerHi+1e-9;t+=step) yticks.push(+t.toFixed(2)); }
  for(const t of yticks){ el("line",{x1:m.l,x2:W-m.r,y1:y(t),y2:y(t)},g); txt(el("text",{x:m.l-10,y:y(t)+4,"text-anchor":"end",class:"tick"},svg), yMode==="log"&&t===floor?"≈0":(isPaper?fmt(t,1):(isLer?t+"%":String(t)))); }
  for(let t=xmin;t<=xmax;t+=10){ el("line",{x1:x(t),x2:x(t),y1:m.t,y2:H-m.b},g); txt(el("text",{x:x(t),y:H-m.b+20,"text-anchor":"middle",class:"tick"},svg), t+"%"); }
  const ax=el("g",{class:"axis"},svg); el("line",{x1:m.l,x2:W-m.r,y1:H-m.b,y2:H-m.b},ax); el("line",{x1:m.l,x2:m.l,y1:m.t,y2:H-m.b},ax);
  txt(el("text",{x:(m.l+W-m.r)/2,y:H-10,"text-anchor":"middle",class:"axlabel"},svg), stat+" Global MWPM decode time on the residual, % of nothing committed");
  txt(el("text",{x:16,y:(m.t+H-m.b)/2,"text-anchor":"middle",class:"axlabel",transform:`rotate(-90 16 ${(m.t+H-m.b)/2})`},svg), isPaper?"logical error per patch per round, ×10⁻³  (the paper's unit)":(yMode==="lerobs"?"logical error rate per observable":(isLer?"logical error rate per shot":"regression vs Global MWPM, pp")));
  if(isLer){ el("line",{x1:m.l,x2:W-m.r,y1:y(gref),y2:y(gref),stroke:"var(--ink-2)","stroke-width":1},svg); }
  el("circle",{class:"pt",cx:x(100),cy:y(DATA.none[key]),r:4.5,fill:"var(--muted)"},svg);
  el("path",{class:"pt",d:star(x(Math.min(...pts.map(p=>p.t["rel_"+stat]))),y(isLer?gref:0),8),fill:"var(--goal)"},svg);
  const tip=document.getElementById(tipId), card=host.parentElement;
  for(const [f,rows,cls] of fams){ const rr=rows.filter(p=>p.t).slice().sort((a,b)=>a.t["rel_"+stat]-b.t["rel_"+stat]);
    el("path",{class:"front",style:`stroke:var(--${cls})`,d:rr.map((p,i)=>(i?"L":"M")+x(p.t["rel_"+stat])+" "+y(p[key])).join(" ")},svg);
    for(const p of rr){ el("circle",{class:"pt",cx:x(p.t["rel_"+stat]),cy:y(p[key]),r:4.5,fill:`var(--${cls})`},svg);
      const h=el("circle",{class:"hit",cx:x(p.t["rel_"+stat]),cy:y(p[key]),r:11,tabindex:0,role:"button","aria-label":p.name},svg);
      const show=()=>{ tip.innerHTML=""; const v=document.createElement("div"); v.className="v"; v.textContent=p.name; tip.appendChild(v);
        for(const [k,val] of [["coverage",fmt(p.coverage,1)+"%"],["LER per patch-round (paper unit)",fmt(p.paper,3)+"×10⁻³ · ×"+fmt(p.mult,3)],["regression vs Global",(p.rd>=0?"+":"")+fmt(p.rd)+" pp"],["L2 mean",fmt(p.t.mean,3)+" ms ("+fmt(p.t.rel_mean,1)+"%)"],["L2 p99",fmt(p.t.p99,3)+" ms ("+fmt(p.t.rel_p99,1)+"%)"],["L2 max",fmt(p.t.max,2)+" ms ("+fmt(p.t.rel_max,1)+"%)"],["residual events per shot",fmt(p.t.events,0)]]){ const r=document.createElement("div"); r.className="r"; const a=document.createElement("span"); a.textContent=k; const b=document.createElement("b"); b.textContent=val; r.append(a,b); tip.appendChild(r); }
        tip.style.display="block"; const rect=card.getBoundingClientRect(), sr=svg.getBoundingClientRect(); const px=sr.left-rect.left+(x(p.t["rel_"+stat])/W)*sr.width, py=sr.top-rect.top+(y(p[key])/H)*sr.height; tip.style.left=Math.min(px+14,rect.width-250)+"px"; tip.style.top=Math.max(8,py-10)+"px"; };
      h.addEventListener("pointerenter",show); h.addEventListener("focus",show); h.addEventListener("pointerleave",()=>tip.style.display="none"); h.addEventListener("blur",()=>tip.style.display="none"); } }
}
function drawL2All(){ if(!DATA.none.t) return; document.getElementById("l2-section").hidden=false; drawL2("l2-mean","mean","tip-l2-mean"); drawL2("l2-p99","p99","tip-l2-p99"); drawL2("l2-max","max","tip-l2-max"); }
(function(){ if(!DATA.none.t) return; const tb=document.querySelector("#l2-table tbody");
  const rows=[["none",[DATA.none]],["size",DATA.size],["margin",DATA.margin],["diameter",DATA.diameter]].flatMap(([f,rs])=>rs.filter(p=>p.t));
  for(const p of rows){ const tr=document.createElement("tr"); for(const [v,cls] of [[p.name,"rule"],[fmt(p.coverage,1)+"%","n"],[(p.rd>=0?"+":"")+fmt(p.rd),"n"],[fmt(p.t.mean,3),"n"],[fmt(p.t.p99,3),"n"],[fmt(p.t.max,2),"n"],[fmt(p.t.rel_mean,1),"n"],[fmt(p.t.rel_p99,1),"n"],[fmt(p.t.rel_max,1),"n"],[fmt(p.t.events,0),"n"]]){ const td=document.createElement("td"); td.className=cls; td.textContent=v; tr.appendChild(td);} tb.appendChild(tr); }
  const M=DATA.margin.filter(p=>p.t), S=DATA.size.filter(p=>p.t); const m15=M.find(p=>p.value===1.5), m05=M.find(p=>p.value===0.5), s2=S.find(p=>p.value===2), sAll=S.find(p=>p.value===null);
  document.getElementById("l2-note").textContent=`${DATA.timing_note}.${DATA.proxy_note||""} At margin > 1.5 (${fmt(m15.coverage,0)}% coverage, +${fmt(m15.rd)} pp) the mean L2 time is ${fmt(m15.t.rel_mean,0)}% of the baseline, the p99 ${fmt(m15.t.rel_p99,0)}%, the max ${fmt(m15.t.rel_max,0)}%. At margin > 0.5 (${fmt(m05.coverage,0)}%, +${fmt(m05.rd)} pp): mean ${fmt(m05.t.rel_mean,0)}%, p99 ${fmt(m05.t.rel_p99,0)}%, max ${fmt(m05.t.rel_max,0)}%. Interior pairs (${fmt(s2.coverage,0)}%, +${fmt(s2.rd)} pp): mean ${fmt(s2.t.rel_mean,0)}%, p99 ${fmt(s2.t.rel_p99,0)}%, max ${fmt(s2.t.rel_max,0)}%. Every interior cluster (${fmt(sAll.coverage,0)}%, +${fmt(sAll.rd)} pp): mean ${fmt(sAll.t.rel_mean,0)}%, p99 ${fmt(sAll.t.rel_p99,0)}%, max ${fmt(sAll.t.rel_max,0)}%.`;
})();

/* tiles and ceiling */
(function(){
  const C=DATA.ceiling, S=DATA.size, sAll=S.find(p=>p.value===null), s2=S.find(p=>p.value===2);
  const rel=(p,base)=>"+"+fmt(100*(p-base)/base,1)+"% relative";
  const m15=DATA.margin.find(p=>p.value===1.5);
  const tiles=[
    ["Global MWPM alone, per patch-round", fmt(DATA.global_paper,2)+"×10⁻³", "the paper's unit · "+fmt(DATA.global_ler,2)+"% per shot · "+fmt(DATA.global_ler_obs,2)+"% per observable"],
    ["Interior pairs", fmt(s2.paper,2)+"×10⁻³", "×"+fmt(s2.mult,3)+" the baseline · "+fmt(s2.coverage,1)+"% coverage · +"+fmt(s2.rd)+" pp per shot"],
    ["Interior, any size", fmt(sAll.paper,2)+"×10⁻³", "×"+fmt(sAll.mult,3)+" the baseline · "+fmt(sAll.coverage,1)+"% coverage · +"+fmt(sAll.rd)+" pp per shot"],
    ["Margin > 1.5", fmt(m15.paper,2)+"×10⁻³", "×"+fmt(m15.mult,3)+" the baseline · "+fmt(m15.coverage,1)+"% coverage · +"+fmt(m15.rd)+" pp per shot"]
  ];
  document.getElementById("unit-note").textContent=`The paper's unit. The Yoked Surface Codes figures report logical error per patch per round: the per-shot failure of the whole block, spread over its ${DATA.patches} patches × ${DATA.rounds} rounds = ${DATA.pieces} patch-rounds with the ${DATA.num_observables} observables treated as independent, using the same sinter shot-to-piece conversion as the paper's plotting script. Global MWPM alone comes out at ${fmt(DATA.global_paper,2)}×10⁻³; confidence intervals on the frontend arms are the paired intervals converted endpoint-wise. The ratio between two decoders is the same in every unit. The absolute values are not comparable with the paper's: this cell runs at p = 0.003 to make failures countable, while the paper's 1D fit at its p = 0.001 gives about ${DATA.paper_fit_p001.toExponential(1).replace("e-","×10⁻")} per patch-round for d = 7 and this block.`;
  const root=document.getElementById("tiles");
  for(const [l,v,sub] of tiles){ const d=document.createElement("div"); d.className="tile"; const a=document.createElement("div"); a.className="label"; a.textContent=l; const b=document.createElement("div"); b.className="value"; b.textContent=v; const c=document.createElement("div"); c.className="sub"; c.textContent=sub; d.append(a,b,c); root.appendChild(d); }
  document.getElementById("ceiling").textContent=`Coverage ceiling. Interior-only rules saturate at ${fmt(C.committed_interior,1)}% of lane-owned detector events; the remaining ${fmt(100-C.committed_interior,1)}% always reaches Global MWPM: ${fmt(C.deferred_port,1)}% sits in clusters that touched the yoke side and are deferred by the port-wall rule, ${fmt(C.deferred_tie,1)}% in clusters deferred at an exact growth tie, and ${fmt(C.committed_wall,1)}% in clusters that touched the real wall, which these ladders leave to MWPM by design. The first two are excluded by the decoder itself, so even committing everything stops at ${fmt(C.committed_interior+C.committed_wall,1)}%. That floor is a property of the lane geometry, one real wall and one yoke wall, not of the commit rule.`;
})();

/* tables */
function fill(id, rows){ const tb=document.querySelector("#"+id+" tbody");
  for(const p of rows){ const tr=document.createElement("tr");
    for(const [v,cls] of [[p.name,"rule"],[fmt(p.coverage,1)+"%","n"],[fmt(p.ler)+"%","n"],[fmt(p.ler_obs,3)+"%","n"],[fmt(p.paper,3),"n"],[fmt(p.mult,3),"n"],[(p.rd>=0?"+":"")+fmt(p.rd),"n"],[fmt(p.lo)+" to "+fmt(p.hi),"n"],[p.step_defects.toLocaleString(),"n"],[p.step_per_1000==null?"—":fmt(p.step_per_1000),"n"]]){ const td=document.createElement("td"); td.className=cls; td.textContent=v; tr.appendChild(td);} tb.appendChild(tr);} }
fill("size-table", DATA.size); fill("margin-table", DATA.margin); fill("diameter-table", DATA.diameter);

</script>
"""
html = (html.replace("{DATA_JSON}", json.dumps(data)).replace("{SHOTS}", f"{N:,}").replace("{GFAIL}", f"{gfail:,}")
            .replace("{GFAILP}", f"{100 * gfail / N:.2f}").replace("{GFAILOBS}", f"{100 * gfail_obs / (N * O):.2f}"))
out_html.write_text(html)
print("size ladder:")
for r in size_rows: print(f"  size<={r['value'] if r['value'] is not None else 'inf':>4}  cov {r['coverage']:5.1f}%  rd {r['rd']:+.2f}  per1000 {r['step_per_1000'] if r['step_per_1000'] is None else round(r['step_per_1000'],2)}")
print("margin ladder:")
for r in margin_rows: print(f"  margin>{r['value']:<4}  cov {r['coverage']:5.1f}%  rd {r['rd']:+.2f} [{r['lo']:+.2f},{r['hi']:+.2f}]  per1000 {r['step_per_1000'] if r['step_per_1000'] is None else round(r['step_per_1000'],2)}")
print("wrote", out_html)
