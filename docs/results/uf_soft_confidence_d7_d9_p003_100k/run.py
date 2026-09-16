"""Run both saved-shot roles and then analyze one distance; no new sampling."""
from pathlib import Path
import argparse
import os

from yoked.hierarchical._uf_soft_collect import collect_features
from yoked.hierarchical._uf_soft_analysis import analyze


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('distance', type=int, choices=(7, 9))
    parser.add_argument('--workers', type=int, default=56)
    parser.add_argument('--out', type=Path, help='raw run directory; defaults to the original TMPDIR run')
    args = parser.parse_args()
    scratch = Path(os.environ['TMPDIR'])
    original = scratch / 'hier-three-decoders-d7-d9-p003-100k-3MxG5S'
    old9 = scratch / 'hier-d9-p003-m2'
    d = args.distance
    run_root = args.out if args.out is not None else scratch / 'uf-soft-d7-d9-100k-QV55RQ'
    output = run_root / f'd{d}'
    calibration = original / 'd7/calibration' if d == 7 else old9 / 'calibration'
    evaluation = original / f'd{d}/evaluation'
    sample = evaluation / 'sample' if d == 7 else old9 / 'evaluation/sample'
    collect_features(calibration, output / 'calibration', workers=args.workers, chunk_size=32)
    collect_features(evaluation, output / 'evaluation', sample_dir=sample,
                     workers=args.workers, chunk_size=32)
    result = analyze(calibration, evaluation, output / 'calibration', output / 'evaluation', output / 'analysis')
    for name, row in result['methods'].items():
        print(f"d={d} {name}: {row['failures']}/{row['shots']}, LER={row['normalized_ler']:.9g}", flush=True)


if __name__ == '__main__':
    main()
