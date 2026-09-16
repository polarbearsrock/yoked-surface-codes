"""Compare a logged readout edge in the full and reduced shot 44875."""
import argparse
import json

import numpy as np

from analyze import Analysis, CASE_ROOTS, write
from yoked.decoders import UnionFindDecoder


def main():
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    analysis = Analysis(7)
    shot = 44875
    reduced = json.loads((args.root / f'reduced_{shot}.json').read_text())
    directory = next(root / f'shot_{shot}' for root in CASE_ROOTS if (root / f'shot_{shot}').exists())
    with np.load(directory / 'fault_responses.npz') as data:
        minimal = np.logical_xor.reduce(data['detectors'][reduced['fault_ids']], axis=0)
    rows = []
    for name, syndrome in [('full', analysis.syndrome(shot)), ('reduced', minimal)]:
        first = analysis.uf._decode(syndrome)
        wu = analysis.weights(first.selected_edges)
        second, growth = UnionFindDecoder(analysis.regraph(wu))._decode_state(syndrome)
        mf = analysis.matrix.edge_ids_of(analysis.native.decode_to_edges_array(syndrome))
        wm = analysis.weights(mf)
        cm = analysis.matrix.edge_ids_of(analysis.native.decode_to_edges_array(syndrome, enable_correlations=True))
        e = analysis.matrix.edge_ids[(5351, 5639)]
        u, v = analysis.graph.endpoints[e]
        rows.append(dict(shot=shot, pattern=name, edge=e, endpoints=[u, v],
                         fired=syndrome[[u, v]].astype(int).tolist(),
                         original_weight=float(analysis.matrix.weights[e]),
                         U_weight=float(wu[e]), M_weight=float(wm[e]),
                         U_selected=e in second.selected_edges, M_selected=e in cm,
                         forest=e in growth.forest, same_cluster=growth.find(u) == growth.find(v),
                         root_sizes=[growth.size[growth.find(x)] for x in (u, v)],
                         root_parities=[int(growth.parity[growth.find(x)]) for x in (u, v)],
                         root_has_boundary=[growth.terminal[growth.find(x)] for x in (u, v)],
                         CUF_selected=list(second.selected_edges), CMWPM_selected=cm))
    write(args.root / 'readout_edge_details.json', rows)


if __name__ == '__main__':
    main()
