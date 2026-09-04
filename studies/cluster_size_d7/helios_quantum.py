"""Helios growth quantum for a cell: max lane edge weight / w_max, from the compiled treatment projection."""
import json, sys
from pathlib import Path
import gen
from yoked._yoked_memory_circuits import yoked_magic_memory_circuit
from yoked.decoding._patch_uf_decoder import PATCH_UF_V1_POLICY, PatchUFTreatmentDecoder
cell = Path(sys.argv[1]); prov = json.load(open(cell / "provenance.json")); c = prov["cell"]
circuit = yoked_magic_memory_circuit(patch_diameter=c["d"], rounds=c["rounds"], noise=gen.NoiseModel.si1000(c["p"]), style=c["style"], yokes=c["yokes"], num_patches=c["patches"], remove_x_yoke=False)
dem = circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
treat = PatchUFTreatmentDecoder(policy=PATCH_UF_V1_POLICY).compile_decoder_for_dem(dem=dem)
ws = [w.integer * 2.0 ** w.binary_exponent for w in treat.projection.exact_weights]
json.dump(dict(max_lane_edge_weight=max(ws), min_lane_edge_weight=min(ws), w_max=16, quantum=max(ws) / 16), open(cell / "helios_quantum.json", "w"), indent=1)
print("quantum", max(ws) / 16)
