"""Prepare three appended Slides pages from inspected native exemplars."""
from __future__ import annotations
import argparse
import copy
import csv
import json
from pathlib import Path


def main(workspace: Path, output: Path, source: Path):
    deck = json.loads((workspace/'raw-template.json').read_text())['structuredContent']
    results = json.loads((source/'results.json').read_text())
    specs = [
        dict(id='hd_yoked_accuracy_100k_20260930', exemplar='hd_bp_accuracy_100k_20260929',
             figure='normalized_ler', role='Absolute normalized accuracy across distance for all four decoders.',
             title='Yoked-code accuracy across distance', heading1='Six-patch experiment',
             body1='SI1000 p = 0.3%\n4d rounds; two ideal yokes\n100k paired shots per d\nPyMatching v2.4.0',
             heading2='BP5 + weighted UF',
             body2='Five BP iterations\nDamping = 0.5\nCluster gap; one UF pass',
             caption='Effective LER per physical patch per round; sinter normalization: pieces = 6r, values = 8.',
             alt='Log-scale effective LER per physical patch per round versus d=7,9,11,13,15, for four decoders. Error bars are transformed exact 95% binomial intervals. Six patches, two ideal yokes, SI1000 p=0.003, r=4d. BP5 plus weighted UF plus cluster gap closely tracks correlated MWPM; correlated UF has higher LER.'),
        dict(id='hd_yoked_relative_100k_20260930', exemplar='hd_bp_relative_100k_20260929',
             figure='relative_ler', role='Show the accuracy penalty relative to complementary-gap MWPM.',
             title='LER gap versus complementary MWPM', heading1='BP5 + weighted UF',
             body1='−1.3% at d = 7\n+13.8% at d = 15', heading2='Correlated UF',
             body2='+28.7% at d = 7\n+138.0% at d = 15',
             caption='Baseline: correlated MWPM + complementary gap. Increase = 100 × (normalized LER / baseline − 1).',
             alt='Percentage increase in effective normalized LER relative to correlated MWPM plus complementary gap, whose baseline is zero. BP5 plus weighted UF ranges from -1.3% at d=7 to +13.8% at d=15; correlated UF ranges from +28.7% to +138.0%. Error bars are 95% paired whole-shot bootstrap intervals, 10000 replicates. Small gaps do not establish statistical equivalence.'),
        dict(id='hd_yoked_latency_100k_20260930', exemplar='hd_bp_latency_100k_20260929',
             figure='serial_latency', role='Compare end-to-end serial software latency, including confidence and L2.',
             title='Median serial latency across distance', heading1='At d = 15 (median)',
             body1='MWPM + comp.: 1,046 ms\nMWPM + cluster: 25.8 ms\ncUF + cluster: 309 ms\nBP5 + wUF: 1,185 ms',
             heading2='Measured CPU cost',
             body2='Confidence + L2 included\nBP5 is 45.9× slower\nthan MWPM + cluster',
             caption='Warm CPU API timing; six patches serial. Includes confidence + L2; excludes sampling and setup.',
             alt='Log-scale median warm single-thread decode time for one full six-patch block. 1000 paired blocks per distance, with input preparation, local decoding, confidence and L2 included. At d=15, correlated MWPM plus complementary gap is 1046.1 ms, correlated MWPM plus cluster gap is 25.8 ms, correlated UF plus cluster gap is 309.3 ms, and BP5 plus weighted UF plus cluster gap is 1184.8 ms. These are current software measurements on a shared CPU host, including Python adapters and built-in validation, not hardware latency or streaming deadlines.'),
    ]
    common_notes = (
        'Completed experiment: six rotated surface-code patches, two ideal yokes, SI1000 p=0.003, '
        'CZ extraction, r=4d, ideal time boundaries. Distances 7,9,11,13,15, 100000 paired accuracy shots per distance. '
        'The d=7 accuracy data are reused exactly from the prior corrected experiment; other distances are new.\n\n'
        'All arms share plain PyMatching MWPM at L2. Correlated MWPM is PyMatching 2.4.0; the MPP matching-radius '
        'cluster score uses the same pinned native engine. Stim is the Dante fork 1.17.dev0. BP5 uses exactly five '
        'sum-product flooding iterations, damping 0.5, LLR clip 30, and posterior-to-edge projection -log(clip(sum(p),1e-15,1)). '
        'The BP5 arm ends with a single weighted UF pass and an uncapped cluster gap; no correlation-discount pass follows BP.\n\n'
        'A block failure is any of the twelve tracked patch observables wrong after L2. Normalization is '
        'sinter.shot_error_rate_to_piece_error_rate(block_ler,pieces=6*r,values=8). This is an effective per-physical-patch '
        'per-round normalization, not a directly measured independent patch hazard. Absolute confidence limits are '
        'exact binomial limits transformed through the same function. Relative ratios and their 95% intervals use '
        '10000 whole paired shot bootstrap draws, reusing the exact draws from the original analysis. '
        'Comparisons are exploratory; no statistical equivalence or hardware speedup is claimed.\n\n'
        'Latency: 1000 uniformly selected saved shots per distance, identical rows for all arms; 32 distinct warm-up shots; '
        'batch size one, one thread pinned to CPU0 on AMD EPYC 9374F; six patches serial. Balanced interleaved arm order. '
        'Warm API scope includes input preparation, L1, confidence, L2, Python adapters and built-in validation. '
        'Excludes sampling, IO, decoder construction, compilation and external correctness comparison. All timed '
        'predictions agree with the accuracy data. Shared host; results are current software latency, not a pure engine '
        'benchmark or hardware/streaming latency. The figure plots MEDIAN, whereas the older single-patch reference slide '
        'plots MEAN over a different workload and timer scope.\n\n'
        'Accuracy collection used 32 MWPM processes followed by 32 OpenMP threads for UF/BP. These parallel collection '
        'times are not used as single-shot latency.\n\n'
        f'Source: {source.resolve()}\n'
        'Recipe: experiments/yoked_fourway_sweep/plot_for_slides.py\n'
        f'Figures, data, provenance: {output.resolve()}\n\n')
    table = (output/'figures/plot_values.csv').read_text()
    plan = dict(presentation_id=deck['presentationId'], source_revision=deck['revisionId'],
        source_order=[s['objectId'] for s in deck['slides']], slides=specs,
        method='Append three duplicate exemplars in the existing deck, per explicit user choice.',
        media_mapping={'each exemplar chart': 'replace with named 16:9 scientific figure',
                       'inherited UW logo and gold rule': 'keep'},
        notes_mapping='Replace copied single-patch study notes with the six-patch protocol and exact plotted data.',
        text_mapping='Replace native title, four right-column text slots and caption in place; retain native styles.',
        order='Place the three new slides at the end in accuracy, relative accuracy, latency order.')
    for spec in specs:
        exemplar = next(s for s in deck['slides'] if s['objectId']==spec['exemplar'])
        existing_ids = {s['objectId'] for s in deck['slides']}
        assert spec['id'] not in existing_ids
        elements = {e['objectId'].removeprefix(spec['exemplar']+'_'):e for e in exemplar['pageElements']}
        assert len([e for e in elements.values() if 'image' in e])==1
        assert 'image' in elements['10']
        mapping = {spec['exemplar']:spec['id']}
        mapping.update({e['objectId']:spec['id']+'_'+suffix for suffix,e in elements.items()})
        requests = [{'duplicateObject':{'objectId':spec['exemplar'],'objectIds':mapping}}]
        for key, value in [('1',spec['title']),('4',spec['heading1']),('5',spec['body1']),
                           ('6',spec['heading2']),('7',spec['body2']),('caption',spec['caption'])]:
            e = elements[key]
            styles = [t['textRun']['style'] for t in e['shape']['text']['textElements'] if 'textRun' in t]
            assert styles and all(s==styles[0] for s in styles), 'Mixed-style shape needs run-level replacement'
            style = copy.deepcopy(styles[0])
            oid = spec['id']+'_'+key
            requests.extend([
                {'deleteText':{'objectId':oid,'textRange':{'type':'ALL'}}},
                {'insertText':{'objectId':oid,'insertionIndex':0,'text':value}},
                {'updateTextStyle':{'objectId':oid,'textRange':{'type':'ALL'},'style':style,'fields':','.join(style)}}])
        image = str((output/'figures'/f"{spec['figure']}.png").resolve())
        requests.extend([
            {'replaceImage':{'imageObjectId':spec['id']+'_10','url':image,'imageReplaceMethod':'CENTER_INSIDE'}},
            {'updatePageElementAltText':{'objectId':spec['id']+'_10','title':spec['title'],'description':spec['alt']}}])
        payload = dict(presentation_id=deck['presentationId'], image_uris=image, requests=requests)
        (output/f"{spec['figure']}_requests.json").write_text(json.dumps(payload,indent=2)+'\n')
        (output/f"{spec['figure']}_speaker_notes.txt").write_text(spec['title']+'\n\n'+common_notes+table)
    (output/'slide-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps({'slides':len(specs),'ids':[s['id'] for s in specs],'output':str(output)}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    a=p.parse_args()
    main(a.workspace,a.output,a.source)
