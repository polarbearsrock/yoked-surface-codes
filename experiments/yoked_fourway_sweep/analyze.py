"""Publish normalized accuracy, paired uncertainty, and separately measured latency."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from run_uf import configure,RECIPE


def render(output,launch,results,complete):
    import numpy as np
    from core import NAMES,LABELS
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    colors=('#304B78','#597CA4','#BC7944','#387B66')
    distances=launch['distances']
    fig,ax=plt.subplots(figsize=(8.5,5),constrained_layout=True)
    for k,name in enumerate(NAMES):
        values=np.array([results[str(d)]['variants'][name]['normalized_ler'] for d in distances])
        bounds=np.array([results[str(d)]['variants'][name]['normalized_ci95'] for d in distances])
        nonzero=values>0
        ax.errorbar(np.array(distances)[nonzero],values[nonzero],
            yerr=np.stack((values[nonzero]-bounds[nonzero,0],bounds[nonzero,1]-values[nonzero])),
            label=LABELS[k],color=colors[k],marker='o',capsize=3,linewidth=1.6)
        if (~nonzero).any():
            ax.scatter(np.array(distances)[~nonzero],bounds[~nonzero,1],color=colors[k],marker='v')
    ax.set_yscale('log')
    ax.set_xticks(distances)
    ax.set_xlabel('Surface-code distance d')
    ax.set_ylabel('Effective LER per physical patch per round')
    ax.set_title('Six yoked patches · SI1000 p=0.003 · r=4d\n100,000 paired shots per distance · exact 95% intervals')
    ax.grid(alpha=.2,which='both')
    ax.legend(fontsize=9)
    ax.spines[['top','right']].set_visible(False)
    fig.savefig(output/'normalized_ler.png',dpi=180)
    fig.savefig(output/'normalized_ler.svg')
    plt.close(fig)
    if complete:
        fig,axes=plt.subplots(1,2,figsize=(11.5,4.4),constrained_layout=True)
        for metric,ax,title in zip(('median_ms','p99_ms'),axes,('Median latency','p99 latency')):
            for k,name in enumerate(NAMES):
                values=[results[str(d)]['latency']['variants'][name][metric] for d in distances]
                ax.plot(distances,values,'o-',color=colors[k],label=LABELS[k])
            ax.set_yscale('log')
            ax.set_xticks(distances)
            ax.set_xlabel('Surface-code distance d')
            ax.set_ylabel('Wall time per six-patch block (ms)')
            ax.set_title(title)
            ax.grid(alpha=.2,which='both')
            ax.spines[['top','right']].set_visible(False)
        axes[0].legend(fontsize=8)
        fig.suptitle('Warm CPU decoding · one thread · batch size one\n1,000 paired blocks per distance; includes confidence and L2',fontsize=11)
        fig.savefig(output/'latency.png',dpi=180)
        fig.savefig(output/'latency.svg')
        plt.close(fig)
    with (output/'accuracy.csv').open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['distance','rounds','patches','p','shots','variant','block_failures',
            'block_ler','normalized_ler','normalized_ci95_low','normalized_ci95_high'])
        for d in distances:
            for name in NAMES:
                v=results[str(d)]['variants'][name]
                writer.writerow([d,4*d,6,.003,100000,name,v['block_failures'],v['block_failure_rate'],
                    v['normalized_ler'],*v['normalized_ci95']])
    if complete:
        with (output/'latency.csv').open('w',newline='') as stream:
            writer=csv.writer(stream)
            columns=['mean_ms','median_ms','p95_ms','p99_ms','max_ms','mean_process_ms','amortized_mean_us_per_patch_round']
            writer.writerow(['distance','rounds','variant','samples','threads',*columns])
            for d in distances:
                for name in NAMES:
                    v=results[str(d)]['latency']['variants'][name]
                    writer.writerow([d,4*d,name,1000,1,*[v[c] for c in columns]])
    lines=['# Four-way distance sweep: normalized LER and latency','',
        'Six rotated surface-code patches; two ideal yokes; SI1000 p=0.003 (0.3%); CZ extraction; '
        'r=4d rounds; ideal time boundaries. There are 100,000 paired shots per distance. '
        'All variants use the same plain PyMatching outer MWPM.', '',
        'The four configurations are correlated MWPM + complementary gap, correlated MWPM + '
        'MPP cluster gap, correlated UF + cluster gap, and BP5 + single-pass weighted UF + cluster gap. '
        'BP5 uses five sum-product flooding iterations, damping 0.5 and LLR clip 30, with the existing '
        'posterior-to-edge projection; all confidence scores are uncapped and uncalibrated.', '',
        '## Normalized logical error rates','',
        '| d | r | cMWPM + complementary | cMWPM + cluster | cUF + cluster | BP5 + weighted UF + cluster |',
        '|---:|---:|---:|---:|---:|---:|']
    for d in distances:
        cells=[f'{results[str(d)]["variants"][name]["normalized_ler"]:.6g}' for name in NAMES]
        lines.append('| '+ ' | '.join([str(d),str(4*d),*cells])+' |')
    lines+=['','Values are effective LER per physical patch per round, computed as '
        '`sinter.shot_error_rate_to_piece_error_rate(block_ler, pieces=6*r, values=8)`. '
        'The measured failure event is any tracked patch observable being wrong after L2 for the '
        'whole six-patch block. This is the established effective normalization, not a directly '
        'measured independent patch hazard. Exact binomial confidence limits are transformed '
        'through the same function. Zero failures, if present, retain nonzero upper limits.','',
        '| d | Configuration | Failures / 100,000 | Normalized LER | Exact normalized 95% CI |',
        '|---:|---|---:|---:|---:|']
    for d in distances:
        for k,name in enumerate(NAMES):
            v=results[str(d)]['variants'][name]
            low,high=v['normalized_ci95']
            lines.append(f'| {d} | {LABELS[k]} | {v["block_failures"]:,} | {v["normalized_ler"]:.6g} | {low:.6g}–{high:.6g} |')
    lines+=['','All six paired comparisons at every distance are in results.json, with 10,000 '
        'whole-shot bootstrap replicates for block and normalized differences. McNemar p-values '
        'are exploratory and unadjusted. The d=7 accuracy result is reused exactly from the '
        'preceding corrected experiment; larger distances use new saved samples.','',
        '## CPU latency','']
    if complete:
        lines+=['Each cell is **median / p99 milliseconds per full six-patch block**.','',
            '| d | cMWPM + complementary | cMWPM + cluster | cUF + cluster | BP5 + weighted UF + cluster |',
            '|---:|---:|---:|---:|---:|']
        for d in distances:
            cells=[]
            for name in NAMES:
                v=results[str(d)]['latency']['variants'][name]
                cells.append(f'{v["median_ms"]:.3f} / {v["p99_ms"]:.3f}')
            lines.append('| '+' | '.join([str(d),*cells])+' |')
        lines+=['','Latency uses 1,000 uniformly selected saved shots per distance, the same rows '
            'for every variant, 32 warm-up rows, batch size one and one CPU thread pinned to one '
            'physical core. Variant order is balanced and interleaved. All six patches are decoded '
            'serially within a block. The timer includes local input preparation, L1, confidence '
            'extraction and L2. Simulation, file I/O, construction, compilation and external '
            'post-decode correctness checks are excluded. The collection jobs are finished before '
            'latency measurement starts.','',
            'These are measurements of the current software implementation, including Python '
            'adapters and built-in validation. The complementary-gap adapter reconstructs frozen '
            'weighted matching graphs; that work is included. They are not pure PyMatching-engine '
            'benchmarks, hardware latency estimates or online streaming deadlines. Every timed '
            'prediction is compared with the corresponding accuracy-run prediction after the timer. '
            'Raw wall/process times, row IDs and variant order are retained in d*/latency.npz.','',
            'latency.csv additionally reports mean, p95, empirical maximum, process time and '
            'amortized mean microseconds per patch-round. Dividing elapsed block time by 6*r '
            'is only an amortized work measure. The p99 estimates are empirical tails from '
            '1,000 samples, not worst-case bounds.']
    else:
        lines+=['Latency measurement is pending; the accuracy sweep is complete.']
    lines+=['','## Implementation and reproducibility','',
        f'PyMatching **{launch["pymatching_version"]}** supplies the native correlated MWPM and all '
        'forced complementary matches. The MPP extension compiles the same pinned PyMatching '
        f'2.4.0 engine, commit `{launch["mpp_build_metadata"]["dependencies"]["pymatching"]}`, '
        'to expose matching-radius confidence. MWPM variants are required to agree on each '
        'local reference prediction and chosen-class matching cost during collection.','',
        f'Stim **{launch["stim_fork"]["version"]}** is the Dante fork at '
        f'`{launch["stim_fork"]["source_commit"]}`. Its installed native binary hash is verified. '
        'New sampling seeds are 202609290700 + d. The unchanged d=7 seed is 202609290707.','',
        'Accuracy collection uses 32 MWPM processes followed by 32 OpenMP threads for UF/BP, '
        'on the same 32 distinct physical cores. Collection wall time is recorded separately '
        'and is not used to infer single-shot latency. UF adapters are checked against independent '
        'Python oracles, prior native implementations, one-versus-32-thread execution and the '
        'separate dispatch used for latency. All corrections and final yoke parities are checked.','',
        'launch.json, uf_build.json, source_manifest.json and source_snapshot/ record dependencies, '
        'revisions, original local changes, commands and binary hashes. Each d*/ directory retains '
        'the exact sample, all four paired predictions and confidence values, detailed uncertainty '
        'and validation records. See protocol.md and the retained experiment recipes.','']
    (output/'report.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--accuracy-only',action='store_true')
    args=parser.parse_args()
    output=args.output.resolve()
    target=output/'source_snapshot'/RECIPE
    manifest=json.loads((output/'source_manifest.json').read_text())
    for name in ('analyze.py','metrics.py'):
        source=Path(__file__).resolve().parent/name
        if source!=target/name:
            shutil.copyfile(source,target/name)
        manifest[str(RECIPE/name)]=hashlib.sha256((target/name).read_bytes()).hexdigest()
    (output/'source_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    if Path(__file__).resolve()!=target/'analyze.py':
        import os
        os.execv(sys.executable,[sys.executable,str(target/'analyze.py'),*sys.argv[1:]])
    launch=configure(output,1)
    from core import NAMES,sha,write_json
    from metrics import summarize
    for name,expected in manifest.items():
        if sha(output/'source_snapshot'/name)!=expected:
            raise ValueError(f'Frozen source mismatch: {name}')
    results={}
    for d in launch['distances']:
        results[str(d)]=summarize(output,launch,d)
        print(f'd={d}: normalized LER '+json.dumps({name:results[str(d)]['variants'][name]['normalized_ler'] for name in NAMES}),flush=True)
    complete=all('latency' in r for r in results.values())
    if not args.accuracy_only and not complete:
        raise ValueError('All five latency measurements must finish before final publication')
    write_json(output/'results.json',dict(protocol='protocol.md',launch='launch.json',distances=results,
        state='complete' if complete else 'accuracy_complete',created_utc=datetime.now(timezone.utc).isoformat()))
    render(output,launch,results,complete)
    files=[p for p in output.rglob('*') if p.is_file() and p not in (output/'artifact_manifest.json',output/'progress.json')]
    write_json(output/'artifact_manifest.json',{str(p.relative_to(output)):sha(p) for p in sorted(files)})
    write_json(output/'progress.json',dict(state='complete' if complete else 'accuracy_complete',
        distances=launch['distances'],shots_per_distance=100000,report=str(output/'report.md'),results_sha256=sha(output/'results.json')))
    print(f'Report: {output/"report.md"}',flush=True)


if __name__=='__main__':
    main()
