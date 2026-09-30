"""Prepare native section dividers anchored to the original slide numbers."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path


def main(workspace: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    deck=json.loads((workspace/'raw-template.json').read_text())['structuredContent']
    source=next(s for s in deck['slides'] if s['objectId']=='hd_cover')
    elements={e['objectId']:e for e in source['pageElements']}
    assert not any('image' in e or 'video' in e for e in source['pageElements'])
    specs=[
        dict(id='hd_experiment_1_divider_20260930',after_original_slide=10,
             anchor='hd_cold_timing',number=1,
             title='Experiment 1\nHierarchical decoder accuracy',
             heading='MWPM and UF confidence methods',
             detail='6 patches · 1M paired shots/d · d = 7–15 · SI1000 p = 0.3% · r = 4d',
             notes='Section 1: the completed paired 1M-shot hierarchical decoder comparison. Six patches, two ideal yokes, distances 7,9,11,13,15, SI1000 p=0.003, r=4d. Compare complementary-gap MWPM, MWPM with MPP cluster confidence, and correlated UF with cluster gap. Following slides show effective normalized LER per physical patch per round and its relative difference from the complementary-gap baseline.'),
        dict(id='hd_experiment_2_divider_20260930',after_original_slide=12,
             anchor='hd_ler_relative_1m_20260927',number=2,
             title='Experiment 2\nSingle-patch BP + weighted UF',
             heading='Fixed BP budgets: 1, 2, 5 and 10 iterations',
             detail='1 patch · 100k paired shots/d · d = 7–13 · SI1000 p = 0.3% · r = 4d',
             notes='Section 2: the single-patch fixed-BP-budget experiment. One patch, zero yokes, SI1000 p=0.003, distances 7,9,11,13, r=4d, and 100000 paired accuracy shots per distance. Compare BP1/BP2/BP5/BP10 plus weighted UF with the saved decoder baselines. Accuracy is LER per full memory shot; the following slides also show serial mean latency and the BP/UF timing breakdown.'),
        dict(id='hd_experiment_3_divider_20260930',after_original_slide=16,
             anchor='hd_bp_breakdown_100k_20260929',number=3,
             title='Experiment 3\nSix-patch BP5 + weighted UF',
             heading='Accuracy and latency across four decoders',
             detail='6 patches · 100k paired shots/d · d = 7–15 · SI1000 p = 0.3% · r = 4d',
             notes='Section 3: the four-way yoked-code sweep completed September 30. Six patches, two ideal yokes, SI1000 p=0.003, r=4d, distances 7,9,11,13,15. Compare correlated MWPM plus complementary gap, correlated MWPM plus cluster gap, correlated UF plus cluster gap, and BP5 plus single-pass weighted UF plus cluster gap. Accuracy uses 100000 paired shots per distance and effective LER per physical patch per round. Timings use 1000 paired warm single-thread six-patch blocks per distance; the following figure plots median API latency, including confidence and L2.'),
    ]
    original_order=[s['objectId'] for s in deck['slides']]
    for s in specs:
        assert original_order[s['after_original_slide']-1]==s['anchor']
        assert s['id'] not in original_order
    expected=[]
    for sid in original_order:
        expected.append(sid)
        expected.extend(s['id'] for s in specs if s['anchor']==sid)
    title_runs=[t['textRun']['style'] for t in elements['hd_cover_title']['shape']['text']['textElements'] if 'textRun' in t]
    assert all(style==title_runs[0] for style in title_runs)
    design=json.loads((workspace/'design-system.json').read_text())
    cover_design=next(s for s in design['slides'] if s['slideId']=='hd_cover')
    title_lineage=next(p for p in cover_design['placeholderLineage'] if p['objectId']=='hd_cover_title')
    title_style=copy.deepcopy(title_lineage['effectiveTextStylesByNesting']['0'])
    body=elements['hd_cover_sub']['shape']['text']['textElements']
    body_runs=[t for t in body if 'textRun' in t]
    assert [t['textRun']['content'] for t in body_runs]==['The decoding hierarchy\n','Gidney et al., Yoked surface codes (2025)','\n']
    heading_style=copy.deepcopy(body_runs[0]['textRun']['style'])
    detail_style=copy.deepcopy(body_runs[1]['textRun']['style'])
    detail_style.pop('link',None)
    detail_style['underline']=False
    requests=[]
    for spec in specs:
        sid=spec['id']
        mapping={'hd_cover':sid,**{eid:sid+eid.removeprefix('hd_cover') for eid in elements}}
        requests.append({'duplicateObject':{'objectId':'hd_cover','objectIds':mapping}})
        requests.extend([
            {'deleteText':{'objectId':sid+'_title','textRange':{'type':'ALL'}}},
            {'insertText':{'objectId':sid+'_title','insertionIndex':0,'text':spec['title']}},
            {'updateTextStyle':{'objectId':sid+'_title','textRange':{'type':'ALL'},
                'style':{'fontSize':title_style['fontSize']},
                'fields':'fontSize,fontFamily,weightedFontFamily,bold'}}])
        # Edit the two differently styled paragraphs independently, from the end.
        # Preserve paragraph boundaries and reconstruct each run's original style.
        requests.extend([
            {'deleteText':{'objectId':sid+'_sub','textRange':{'type':'FIXED_RANGE','startIndex':23,'endIndex':64}}},
            {'insertText':{'objectId':sid+'_sub','insertionIndex':23,'text':spec['detail']}},
            {'deleteText':{'objectId':sid+'_sub','textRange':{'type':'FIXED_RANGE','startIndex':0,'endIndex':22}}},
            {'insertText':{'objectId':sid+'_sub','insertionIndex':0,'text':spec['heading']}},
            {'updateTextStyle':{'objectId':sid+'_sub','textRange':{'type':'FIXED_RANGE','startIndex':0,'endIndex':len(spec['heading'])+1},
                'style':heading_style,'fields':','.join(heading_style)+',link'}},
            {'updateTextStyle':{'objectId':sid+'_sub','textRange':{'type':'FIXED_RANGE','startIndex':len(spec['heading'])+1,
                'endIndex':len(spec['heading'])+1+len(spec['detail'])},
                'style':detail_style,'fields':','.join(detail_style)+',link'}}])
    plan=dict(presentation_id=deck['presentationId'],source_revision=deck['revisionId'],
        source_order=original_order,expected_order=expected,slides=specs,
        exemplar='hd_cover',layout='p6',method='Duplicate rendered native title exemplar three times; edit native text in place.',
        media_mapping={'inherited purple background, gold rule, UW logo and university wordmark':'keep'},
        text_mapping={'title':'replace; preserve 33pt title style','subtitle':'replace each styled paragraph separately; preserve 18pt/12pt hierarchy; remove stale citation link','slide number':'keep automatic numbering'},
        notes_mapping='Replace copied cover notes with an accurate experiment overview.',
        requested_anchors='Original slide numbers 10,12,16, captured before inserting anything.')
    (output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    (output/'create_requests.json').write_text(json.dumps(dict(presentation_id=deck['presentationId'],requests=requests,
        write_control={'requiredRevisionId':deck['revisionId']}),indent=2)+'\n')
    print(json.dumps({'dividers':3,'requests':len(requests),'expected_slide_numbers':[expected.index(s['id'])+1 for s in specs]}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.workspace,a.output)
