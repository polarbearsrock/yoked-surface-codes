from typing import List, Optional, Set

import stim


class Layer:
    def copy(self) -> 'Layer':
        raise NotImplementedError()

    def touched(self) -> Set[int]:
        raise NotImplementedError()

    def to_z_basis(self) -> List['Layer']:
        return [self]

    def append_into_stim_circuit(self, out: stim.Circuit) -> None:
        raise NotImplementedError()

    def locally_optimized(self, next_layer: Optional['Layer']) -> List[Optional['Layer']]:
        return [self, next_layer]

    def is_vacuous(self) -> bool:
        return False

    def requires_tick_before(self) -> bool:
        return True

    def implies_eventual_tick_after(self) -> bool:
        return True


def append_sorted_pair_gate(out: stim.Circuit, gate: str, targets1: List[int], targets2: List[int]) -> None:
    pairs = []
    for k in range(len(targets1)):
        t1 = targets1[k]
        t2 = targets2[k]
        t1, t2 = sorted([t1, t2])
        pairs.append((t1, t2))
    for pair in sorted(pairs):
        out.append(gate, pair)
