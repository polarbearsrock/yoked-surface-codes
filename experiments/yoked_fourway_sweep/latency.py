"""Warm single-thread, batch-one latency; never inferred from parallel throughput."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from run_uf import configure,RECIPE


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve()
    frozen=output/'source_snapshot'/RECIPE/'latency.py'
    if Path(__file__).resolve()!=frozen:
        if any(output.glob('d*/latency.npz')) and frozen.exists() and frozen.read_bytes()!=Path(__file__).read_bytes():
            raise ValueError('Refusing to change a recorded latency recipe')
        shutil.copyfile(__file__,frozen)
        manifest=json.loads((output/'source_manifest.json').read_text())
        manifest[str(RECIPE/'latency.py')]=hashlib.sha256(frozen.read_bytes()).hexdigest()
        (output/'source_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
        os.execv(sys.executable,[sys.executable,str(frozen),*sys.argv[1:]])
    launch=configure(output,1)
    import numpy as np
    import pymatching,stim
    from core import NAMES,sha,write_json,UfHierarchicalDecoder
    from yoked.hierarchical._collect import SampleSet
    from yoked.hierarchical._patch_graphs import PatchGraphs
    from yoked.hierarchical._paper_decoder import PaperHierarchicalDecoder
    from yoked.hierarchical._mpp import MppHierarchicalDecoder
    assert pymatching.__version__=='2.4.0'
    work=Path(launch['work'])
    assert (work/'mwpm/completion.json').exists()
    for d in launch['distances'][1:]:
        assert (work/f'uf/d{d}/completion.json').exists()
    build=json.loads((output/'uf_build.json').read_text())
    library=Path(build['library'])
    assert sha(library)==build['sha256']
    protocol=dict(shots=1000,warmup_shots=32,threads=1,cpu=launch['cpu_affinity'][0],
        timer='time.perf_counter_ns; also process_time_ns',batch_size=1,
        scope='Warm API call, one full six-patch block; local preparation, L1, confidence and L2; six patches serial',
        excluded='Simulation, file I/O, object construction, compilation, and post-timing comparison',
        implementation='Current software adapters including Python overhead and built-in solver validation; not pure matching engine or hardware latency',
        gc='disabled only within timed measurement loops for all variants',order='randomized blocks of four balanced cyclic orders',
        library_sha256=build['sha256'],recipe_sha256=sha(frozen),pymatching=pymatching.__version__)
    write_json(output/'latency_protocol.json',protocol)
    for distance in launch['distances']:
        directory=output/f'd{distance}'
        if (directory/'latency.json').exists():
            record=json.loads((directory/'latency.json').read_text())
            assert record['protocol']==protocol and record['raw_sha256']==sha(directory/'latency.npz')
            print(f'd={distance}: completed latency reused.',flush=True)
            continue
        sample=SampleSet.load(directory/'sample')
        with np.load(directory/'paired.npz') as saved:
            expected=saved['predictions']
            np.testing.assert_array_equal(saved['variant_names'],NAMES)
        rng=np.random.default_rng(launch['latency_selection_seed_base']+distance)
        selected=rng.choice(sample.shots,size=1032,replace=False)
        # SampleSet requires increasing row IDs; sorting preserves uniform subsets.
        warm_rows,rows=np.sort(selected[:32]),np.sort(selected[32:])
        warm,_=sample.rows(warm_rows)
        detectors,_=sample.rows(rows)
        dem=stim.DetectorErrorModel(sample.dem_text)
        patches=PatchGraphs.from_yoked_dem(dem,num_patches=6)
        setup=time.perf_counter()
        decoders=[PaperHierarchicalDecoder(dem,num_patches=6),
            MppHierarchicalDecoder(dem,num_patches=6,method='dijkstra',verify=False),
            UfHierarchicalDecoder(library,patches,0),UfHierarchicalDecoder(library,patches,1)]
        setup_seconds=time.perf_counter()-setup
        for variant,decoder in enumerate(decoders):
            for row,data in zip(warm_rows,warm):
                np.testing.assert_array_equal(decoder.decode(data),expected[row,variant])
        orders=np.empty((1000,4),dtype=np.int64)
        for start in range(0,1000,4):
            base=rng.permutation(4)
            rotations=rng.permutation(4)
            for j,shift in enumerate(rotations):
                orders[start+j]=np.roll(base,int(shift))
        wall=np.zeros((1000,4),dtype=np.int64)
        cpu=np.zeros_like(wall)
        predictions=np.zeros((1000,4,12),dtype=bool)
        checkpoint=work/f'd{distance}_latency_checkpoint.npz'
        completed_rows=0
        if checkpoint.exists():
            with np.load(checkpoint) as saved:
                np.testing.assert_array_equal(saved['row_ids'],rows)
                np.testing.assert_array_equal(saved['order'],orders)
                assert str(saved['recipe_sha256'])==protocol['recipe_sha256']
                completed_rows=int(saved['completed_rows'])
                wall[:]=saved['wall_ns']
                cpu[:]=saved['process_ns']
                predictions[:]=saved['predictions']
            np.testing.assert_array_equal(predictions[:completed_rows],expected[rows[:completed_rows]])
        load_before=Path('/proc/loadavg').read_text().strip()
        started=time.perf_counter()
        was_enabled=gc.isenabled()
        gc.disable()
        try:
            for i,data in enumerate(detectors):
                if i<completed_rows:
                    continue
                for variant in orders[i]:
                    cpu_start=time.process_time_ns()
                    wall_start=time.perf_counter_ns()
                    result=decoders[variant].decode(data)
                    wall[i,variant]=time.perf_counter_ns()-wall_start
                    cpu[i,variant]=time.process_time_ns()-cpu_start
                    predictions[i,variant]=result
                    np.testing.assert_array_equal(result,expected[rows[i],variant])
                if (i+1)%50==0:
                    elapsed=time.perf_counter()-started
                    temporary=checkpoint.with_suffix('.partial')
                    with temporary.open('wb') as stream:
                        np.savez_compressed(stream,row_ids=rows,order=orders,completed_rows=i+1,
                            recipe_sha256=protocol['recipe_sha256'],wall_ns=wall,process_ns=cpu,
                            predictions=predictions)
                    temporary.replace(checkpoint)
                    write_json(output/'progress.json',dict(state='latency_running',distance=distance,
                        completed_shots=i+1,target_shots=1000,threads=1,elapsed_seconds=elapsed))
                    print(f'd={distance}: latency {i+1}/1000 paired blocks; elapsed {elapsed:.1f}s',flush=True)
        finally:
            if was_enabled:
                gc.enable()
            for decoder in decoders[2:]:
                decoder.close()
        temporary=work/f'd{distance}_latency.partial'
        with temporary.open('wb') as stream:
            np.savez_compressed(stream,row_ids=rows,warmup_row_ids=warm_rows,order=orders,
                wall_ns=wall,process_ns=cpu,predictions=predictions,variant_names=np.array(NAMES))
        shutil.copyfile(temporary,directory/'latency.npz')
        temporary.unlink()
        variants={}
        for k,name in enumerate(NAMES):
            values=wall[:,k]/1e6
            variants[name]=dict(mean_ms=float(values.mean()),median_ms=float(np.median(values)),
                p95_ms=float(np.percentile(values,95)),p99_ms=float(np.percentile(values,99)),
                max_ms=float(values.max()),mean_process_ms=float(cpu[:,k].mean()/1e6),
                amortized_mean_us_per_patch_round=float(values.mean()*1000/(6*4*distance)))
        write_json(directory/'latency.json',dict(distance=distance,protocol=protocol,variants=variants,
            setup_seconds=setup_seconds,measurement_wall_seconds=time.perf_counter()-started,
            selection_seed=launch['latency_selection_seed_base']+distance,
            raw_sha256=sha(directory/'latency.npz'),predictions_match_accuracy_run=True,
            load_average_before=load_before,load_average_after=Path('/proc/loadavg').read_text().strip()))
        print(f'd={distance}: latency complete: '+json.dumps(variants),flush=True)
    write_json(output/'progress.json',dict(state='latency_complete',distances=launch['distances']))


if __name__=='__main__':
    main()
