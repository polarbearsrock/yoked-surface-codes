"""Collect the two UF variants after the PyMatching phase, with resumable shards."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

RECIPE=Path('experiments/yoked_fourway_sweep')


def configure(output,threads=32):
    launch=json.loads((output/'launch.json').read_text())
    os.environ.update(DANTE_REPO=launch['snapshot_repository'],YOKED_MPP_BUILD=launch['mpp_build'],
        OMP_NUM_THREADS=str(threads),OMP_THREAD_LIMIT='32',OMP_DYNAMIC='FALSE',
        OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    os.environ.pop('OMP_PLACES',None)
    os.environ.pop('OMP_PROC_BIND',None)
    os.sched_setaffinity(0,launch['cpu_affinity'] if threads>1 else launch['cpu_affinity'][:1])
    sys.path.insert(0,str(Path(launch['snapshot_repository'])/'src'))
    return launch


def freeze(output):
    target=output/'source_snapshot'/RECIPE
    launch=json.loads((output/'launch.json').read_text())
    work=Path(launch['work'])
    manifest=json.loads((output/'source_manifest.json').read_text())
    for name in ('README.md','kernel.cc','core.py','run_uf.py'):
        source=Path(__file__).resolve().parent/name
        digest=hashlib.sha256(source.read_bytes()).hexdigest()
        if source != target/name:
            if any(work.glob('uf/d*/configuration.json')) and (target/name).exists() and hashlib.sha256((target/name).read_bytes()).hexdigest()!=digest:
                raise ValueError('Refusing to change decoder sources during collection')
            shutil.copyfile(source,target/name)
        manifest[str(RECIPE/name)]=digest
    (output/'source_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    return target/'run_uf.py'


def collect_distance(output,launch,distance,library,command,sources,validate_only=False):
    import numpy as np
    import pymatching,stim
    from core import FIELDS,NAMES,sha,write_json,verify,make_native,finish_outer
    from yoked.hierarchical._collect import SampleSet
    from yoked.hierarchical._patch_graphs import PatchGraphs
    work=Path(launch['work'])
    sample_path=output/'d7/sample' if distance==7 else work/f'mwpm/d{distance}/sample'
    sample=SampleSet.load(sample_path)
    assert sample.shots==100000 and sample.seed==launch['sampling_seed_base']+distance
    assert sample.parameters.distance==distance and sample.parameters.rounds==4*distance
    assert sample.parameters.p==.003 and sample.parameters.patches==6 and sample.parameters.yokes==2
    patches=PatchGraphs.from_yoked_dem(stim.DetectorErrorModel(sample.dem_text),num_patches=6)
    retained=output/f'd{distance}'
    retained.mkdir(exist_ok=True)
    root=work/f'uf/d{distance}'
    (root/'shards').mkdir(parents=True,exist_ok=True)
    config=dict(distance=distance,variants=NAMES[2:],fields=FIELDS,source_sha256=sources,
        library_sha256=sha(library),build_command=command,threads=32,batch_size=1250,
        sample=dict(sample.identities),seed=sample.seed,shots=sample.shots,parameters=sample.parameters.to_json(),
        bp_iterations=5,bp_damping=.5,bp_llr_clip=30.,bp_uf_passes=1,bp_conditional_discounts=False,
        bp_projection='-log(clip(sum of supporting fault posteriors,1e-15,1))',cluster_gap_cap=None,
        confidence_calibration='none',l2='plain PyMatching v2 MWPM',pymatching=pymatching.__version__)
    config_path=root/'configuration.json'
    if config_path.exists() and json.loads(config_path.read_text())!=json.loads(json.dumps(config)):
        raise ValueError('Configuration differs from the saved run')
    validation_path=retained/'uf_verification.json'
    if not validation_path.exists():
        fresh,_=stim.Circuit(sample.circuit_text).compile_detector_sampler(seed=202609293000+distance).sample(shots=16,separate_observables=True)
        start=time.perf_counter()
        validation=verify(library,patches,fresh)
        validation.update(elapsed_seconds=time.perf_counter()-start,source_sha256=sources,
            library_sha256=sha(library),validation_seed=202609293000+distance)
        write_json(validation_path,validation)
    else:
        validation=json.loads(validation_path.read_text())
        assert validation['source_sha256']==sources and validation['library_sha256']==sha(library)
    print(f'd={distance}: native/Python corrections and gaps, prior implementation parity, and 1/32-thread parity verified.',flush=True)
    if validate_only or distance==7:
        return
    if not (work/'mwpm/completion.json').exists():
        raise RuntimeError('Wait for the MWPM collection to finish before starting UF collection')
    if (root/'completion.json').exists():
        done=json.loads((root/'completion.json').read_text())
        assert done['predictions_sha256']==sha(root/'predictions.npz')
        print(f'd={distance}: completed UF results reused.',flush=True)
        return
    write_json(config_path,config)
    decoders=[make_native(library,p.local_dem) for p in patches]
    started=time.perf_counter()
    for start in range(0,sample.shots,1250):
        stop=min(start+1250,sample.shots)
        stem=root/'shards'/f'{start:06d}-{stop:06d}'
        shard,meta=stem.with_suffix('.npz'),stem.with_suffix('.json')
        if meta.exists():
            saved=json.loads(meta.read_text())
            assert saved['sha256']==sha(shard) and saved['configuration_sha256']==sha(config_path)
            continue
        rows=np.arange(start,stop)
        detectors,actual=sample.rows(rows)
        count=len(rows)
        reference=np.zeros((count,2,12),dtype=bool)
        gaps=np.zeros((count,2,12))
        statistics=np.zeros((count,2,6,len(FIELDS)))
        stage=time.perf_counter()
        for k,(patch,decoder) in enumerate(zip(patches,decoders)):
            packed=np.packbits(patch.local_syndromes(detectors),axis=1,bitorder='little')
            decoded,_,_=decoder.decode(packed,32)
            statistics[:,:,k]=decoded
            masks=decoded[:,:,0].astype(np.uint8)
            reference[:,:,2*k]=masks&1
            reference[:,:,2*k+1]=(masks>>1)&1
            gaps[:,:,2*k:2*k+2]=decoded[:,:,1:3]
        native_seconds=time.perf_counter()-stage
        stage=time.perf_counter()
        prediction=np.zeros_like(reference)
        residual=np.zeros_like(reference)
        sigma=np.zeros((count,2,2),dtype=bool)
        tied=np.zeros_like(sigma)
        for variant in range(2):
            prediction[:,variant],residual[:,variant],sigma[:,variant],tied[:,variant]=finish_outer(patches,detectors,reference[:,variant],gaps[:,variant])
            np.testing.assert_array_equal(prediction[:,variant].reshape(count,6,2).sum(axis=1)%2,detectors[:,patches.yoke_detector_ids])
        outer_seconds=time.perf_counter()-stage
        temporary=shard.with_suffix('.partial')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream,row_ids=rows,actual=actual,reference=reference,gaps=gaps,
                prediction=prediction,residual=residual,sigma=sigma,outer_tied=tied,statistics=statistics)
        temporary.replace(shard)
        write_json(meta,dict(start=start,stop=stop,sha256=sha(shard),configuration_sha256=sha(config_path),
            native_and_packing_seconds=native_seconds,outer_seconds=outer_seconds))
        elapsed=time.perf_counter()-started
        write_json(output/'progress.json',dict(state='uf_running',distance=distance,completed_shots=stop,
            target_shots=100000,threads=32,elapsed_seconds=elapsed,updated_utc=datetime.now(timezone.utc).isoformat()))
        print(f'd={distance}: UF/BP {stop}/100000; elapsed {elapsed:.1f}s',flush=True)
    for decoder in decoders:
        decoder.close()
    chunks,metas={},[]
    for path in sorted((root/'shards').glob('*.npz')):
        meta=json.loads(path.with_suffix('.json').read_text())
        assert meta['sha256']==sha(path) and meta['configuration_sha256']==sha(config_path)
        metas.append(meta)
        with np.load(path) as saved:
            for name in saved.files:
                chunks.setdefault(name,[]).append(saved[name])
    arrays={k:np.concatenate(v) for k,v in chunks.items()}
    np.testing.assert_array_equal(arrays['row_ids'],np.arange(sample.shots))
    np.savez_compressed(root/'predictions.npz',**arrays)
    write_json(root/'completion.json',dict(state='complete',shots=sample.shots,
        predictions_sha256=sha(root/'predictions.npz'),configuration_sha256=sha(config_path),
        verification_sha256=sha(validation_path),elapsed_seconds_this_collection=time.perf_counter()-started,
        native_and_packing_seconds=sum(x['native_and_packing_seconds'] for x in metas),
        outer_seconds=sum(x['outer_seconds'] for x in metas)))
    (retained/'uf').mkdir(exist_ok=True)
    for name in ('predictions.npz','configuration.json','completion.json'):
        shutil.copyfile(root/name,retained/'uf'/name)
    print(f'd={distance}: both UF variants complete.',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--distances',type=int,nargs='+',default=[7,9,11,13,15])
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    output=args.output.resolve()
    frozen=freeze(output)
    if Path(__file__).resolve()!=frozen:
        os.execv(sys.executable,[sys.executable,str(frozen),*sys.argv[1:]])
    launch=configure(output)
    import pymatching,stim
    from core import build,sha,write_json
    assert pymatching.__version__=='2.4.0'
    assert stim.__version__==launch['stim_fork']['version']
    assert sha(launch['stim_fork']['native_module'])==launch['stim_fork']['native_sha256']
    sources={str(RECIPE/name):sha(frozen.parent/name) for name in ('README.md','kernel.cc','core.py','run_uf.py')}
    library,command=build(frozen.parent/'kernel.cc',Path(launch['work'])/'uf-build')
    write_json(output/'uf_build.json',dict(library=str(library),sha256=sha(library),command=command,source_sha256=sources))
    for distance in args.distances:
        collect_distance(output,launch,distance,library,command,sources,args.validate_only)
    if not args.validate_only and args.distances==[7,9,11,13,15]:
        write_json(output/'progress.json',dict(state='accuracy_decoding_complete',distances=args.distances,shots_per_distance=100000))


if __name__=='__main__':
    main()
