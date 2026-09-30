"""Retain and summarize paired accuracy arrays, with the existing normalization."""
from __future__ import annotations
import itertools
import json
from pathlib import Path
import shutil
import numpy as np
from scipy.stats import binomtest
import sinter
import stim
from core import NAMES,LABELS,sha,write_json
from yoked.hierarchical._collect import SampleSet
from yoked.hierarchical._patch_graphs import PatchGraphs


def normalize(rate,distance):
    return float(sinter.shot_error_rate_to_piece_error_rate(float(rate),pieces=6*4*distance,values=8))


def retain_arrays(output,launch,distance):
    directory=output/f'd{distance}'
    directory.mkdir(exist_ok=True)
    if distance==7:
        old=Path(launch['baseline'])
        assert sha(directory/'paired.npz')==sha(old/'data/paired.npz')
    else:
        work=Path(launch['work'])
        mroot=work/f'mwpm/d{distance}'
        uroot=work/f'uf/d{distance}'
        m_manifest=json.loads((mroot/'manifest.json').read_text())
        mwpm_done=json.loads((work/'mwpm/completion.json').read_text())
        assert mwpm_done['distance_manifests'][str(distance)]==sha(mroot/'manifest.json')
        for name,digest in m_manifest['artifacts'].items():
            assert sha(mroot/name)==digest,name
        u_done=json.loads((uroot/'completion.json').read_text())
        assert u_done['predictions_sha256']==sha(uroot/'predictions.npz')
        assert u_done['configuration_sha256']==sha(uroot/'configuration.json')
        with np.load(mroot/'predictions.npz') as saved:
            m={k:saved[k] for k in saved.files}
        with np.load(uroot/'predictions.npz') as saved:
            u={k:saved[k] for k in saved.files}
        np.testing.assert_array_equal(m['row_ids'],u['row_ids'])
        np.testing.assert_array_equal(m['actual'],u['actual'])
        references=np.stack((m['reference'],m['reference'],u['reference'][:,0],u['reference'][:,1]),axis=1)
        predictions=np.stack((m['gap_prediction'],m['mpp_prediction'],u['prediction'][:,0],u['prediction'][:,1]),axis=1)
        confidence=np.stack((m['complementary_gap'],m['cluster_score'],u['gaps'][:,0],u['gaps'][:,1]),axis=1)
        failure=np.any(predictions^m['actual'][:,None],axis=2)
        l1_failure=np.any(references^m['actual'][:,None],axis=2)
        np.savez_compressed(directory/'paired.npz',row_ids=m['row_ids'],actual=m['actual'],
            predictions=predictions,references=references,confidence=confidence,
            block_failures=failure,l1_block_failures=l1_failure,variant_names=np.array(NAMES))
        if not (directory/'sample/sample.json').exists():
            shutil.copytree(mroot/'sample',directory/'sample',dirs_exist_ok=True)
        for kind,source,files in [('mwpm',mroot,('predictions.npz','results.json','manifest.json')),
                                 ('uf',uroot,('predictions.npz','configuration.json','completion.json'))]:
            (directory/kind).mkdir(exist_ok=True)
            for name in files:
                shutil.copyfile(source/name,directory/kind/name)
    with np.load(directory/'paired.npz') as saved:
        arrays={k:saved[k] for k in saved.files}
    sample=SampleSet.load(directory/'sample')
    assert sample.shots==launch['shots_per_distance']==100000
    assert sample.parameters.distance==distance and sample.parameters.rounds==4*distance
    assert sample.parameters.patches==6 and sample.parameters.yokes==2 and sample.parameters.p==.003
    np.testing.assert_array_equal(arrays['variant_names'],NAMES)
    np.testing.assert_array_equal(arrays['row_ids'],np.arange(sample.shots))
    actual=np.unpackbits(sample.actual_packed,axis=1,count=12,bitorder='little').astype(bool)
    np.testing.assert_array_equal(arrays['actual'],actual)
    assert np.isfinite(arrays['confidence']).all() and (arrays['confidence']>=0).all()
    patches=PatchGraphs.from_yoked_dem(stim.DetectorErrorModel(sample.dem_text),num_patches=6)
    yoke=np.column_stack([(sample.detectors_packed[:,k//8]>>(k%8))&1 for k in patches.yoke_detector_ids]).astype(bool)
    for k in range(4):
        np.testing.assert_array_equal(arrays['predictions'][:,k].reshape(sample.shots,6,2).sum(axis=1)%2,yoke)
    np.testing.assert_array_equal(arrays['block_failures'],np.any(arrays['predictions']^actual[:,None],axis=2))
    return sample,arrays


def summarize(output,launch,distance):
    sample,arrays=retain_arrays(output,launch,distance)
    directory=output/f'd{distance}'
    failures=arrays['block_failures']
    shots=sample.shots
    patterns,counts=np.unique(failures,axis=0,return_counts=True)
    seed=202609294000+distance
    draws=np.random.default_rng(seed).multinomial(shots,counts/shots,size=10000)
    rates=(draws@patterns.astype(np.int64))/shots
    normalized=np.array([[normalize(p,distance) for p in row] for row in rates])
    result=dict(distance=distance,rounds=4*distance,shots=shots,parameters=sample.parameters.to_json(),
        seed=sample.seed,sample=dict(sample.identities),variants={},paired={},
        normalization=dict(function='sinter.shot_error_rate_to_piece_error_rate',pieces=6*4*distance,values=8,
            meaning='Effective per physical patch per round, inferred from whole-block failures'),
        failure_definition='Any tracked patch observable wrong after L2; whole six-patch, r=4d shot',
        bootstrap=dict(seed=seed,replicates=10000,unit='whole paired shot'),
        reused_accuracy=(distance==7),paired_arrays_sha256=sha(directory/'paired.npz'))
    for k,name in enumerate(NAMES):
        count=int(failures[:,k].sum())
        rate=count/shots
        interval=binomtest(count,shots).proportion_ci(method='exact')
        l1_wrong=arrays['references'][:,k]^arrays['actual']
        l1_failed=l1_wrong.any(axis=1)
        result['variants'][name]=dict(label=LABELS[k],block_failures=count,block_failure_rate=rate,
            block_ci95=[interval.low,interval.high],normalized_ler=normalize(rate,distance),
            normalized_ci95=[normalize(interval.low,distance),normalize(interval.high,distance)],
            l1_block_failures=int(l1_failed.sum()),l1_failed_patch_sectors=int(l1_wrong.sum()),
            l2_repairs=int((l1_failed&~failures[:,k]).sum()),
            l2_regressions=int((~l1_failed&failures[:,k]).sum()))
    for a,b in itertools.combinations(range(4),2):
        repairs=int((failures[:,a]&~failures[:,b]).sum())
        regressions=int((~failures[:,a]&failures[:,b]).sum())
        arate=failures[:,a].mean()
        brate=failures[:,b].mean()
        result['paired'][NAMES[b]+'__versus__'+NAMES[a]]=dict(baseline=NAMES[a],candidate=NAMES[b],
            repairs=repairs,regressions=regressions,block_difference=float(brate-arate),
            block_difference_ci95=np.percentile(rates[:,b]-rates[:,a],[2.5,97.5]).tolist(),
            normalized_difference=normalize(brate,distance)-normalize(arate,distance),
            normalized_difference_ci95=np.percentile(normalized[:,b]-normalized[:,a],[2.5,97.5]).tolist(),
            normalized_rate_ratio=normalize(brate,distance)/normalize(arate,distance) if arate else None,
            mcnemar_exact_p=float(binomtest(regressions,repairs+regressions).pvalue) if repairs+regressions else 1.)
    if (directory/'latency.json').exists():
        latency=json.loads((directory/'latency.json').read_text())
        assert latency['raw_sha256']==sha(directory/'latency.npz')
        assert latency['predictions_match_accuracy_run']
        result['latency']=latency
    if distance!=7:
        u=json.loads((directory/'uf/completion.json').read_text())
        result['collection']=dict(uf_pair_seconds=u['elapsed_seconds_this_collection'],
            uf_pair_native_and_packing_seconds=u['native_and_packing_seconds'],uf_pair_outer_seconds=u['outer_seconds'])
    write_json(directory/'summary.json',result)
    return result
