"""Production exact-arithmetic weighted-UF lane engine.

The initial production lifecycle uses a deterministic rebuilt event heap per
atomic batch.  It is intentionally simple: it establishes exact decisions and
literal heap/memory counters before later performance work.  Replacing this
lifecycle changes production-policy metrics and requires new golden traces.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import numbers
from collections.abc import Iterable
from fractions import Fraction

from yoked.decoding._patch_uf_reference import (
    BudgetExceeded,
    BudgetLimits,
    CensoredComponent,
    CompletedComponent,
    LaneGraphProtocol,
    LaneOutcome,
    UFCounters,
    UFEdge,
    UFLaneGraph,
    UFPolicy,
    _run_lane_persistent,
    as_fraction,
)


@functools.total_ordering
@dataclasses.dataclass(frozen=True, eq=False)
class Dyadic:
    """Canonical exact value ``mantissa * 2**exponent``."""

    mantissa: int
    exponent: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.mantissa, bool) or not isinstance(
            self.mantissa, numbers.Integral
        ):
            raise TypeError("Dyadic mantissa must be an integer")
        if isinstance(self.exponent, bool) or not isinstance(
            self.exponent, numbers.Integral
        ):
            raise TypeError("Dyadic exponent must be an integer")
        mantissa = int(self.mantissa)
        exponent = int(self.exponent)
        if mantissa == 0:
            exponent = 0
        else:
            while mantissa % 2 == 0:
                mantissa //= 2
                exponent += 1
        object.__setattr__(self, "mantissa", mantissa)
        object.__setattr__(self, "exponent", exponent)

    @classmethod
    def from_value(cls, value: object) -> "Dyadic":
        if isinstance(value, cls):
            return value
        fraction = as_fraction(value)
        denominator = fraction.denominator
        if denominator & (denominator - 1):
            raise ValueError("production exact values must be dyadic rationals")
        return cls(fraction.numerator, -(denominator.bit_length() - 1))

    def as_fraction(self) -> Fraction:
        if self.exponent >= 0:
            return Fraction(self.mantissa << self.exponent)
        return Fraction(self.mantissa, 1 << -self.exponent)

    def _coerce(self, other: object) -> "Dyadic" | NotImplemented:
        try:
            return Dyadic.from_value(other)
        except (TypeError, ValueError):
            return NotImplemented

    def __hash__(self) -> int:
        return hash((self.mantissa, self.exponent))

    def __eq__(self, other: object) -> bool:
        value = self._coerce(other)
        if value is NotImplemented:
            return False
        return (self.mantissa, self.exponent) == (value.mantissa, value.exponent)

    def __lt__(self, other: object) -> bool:
        value = self._coerce(other)
        if value is NotImplemented:
            return NotImplemented
        exponent = min(self.exponent, value.exponent)
        left = self.mantissa << (self.exponent - exponent)
        right = value.mantissa << (value.exponent - exponent)
        return left < right

    def __neg__(self) -> "Dyadic":
        return Dyadic(-self.mantissa, self.exponent)

    def __add__(self, other: object) -> "Dyadic":
        value = self._coerce(other)
        if value is NotImplemented:
            return NotImplemented
        exponent = min(self.exponent, value.exponent)
        return Dyadic(
            (self.mantissa << (self.exponent - exponent))
            + (value.mantissa << (value.exponent - exponent)),
            exponent,
        )

    def __radd__(self, other: object) -> "Dyadic":
        return self + other

    def __sub__(self, other: object) -> "Dyadic":
        value = self._coerce(other)
        if value is NotImplemented:
            return NotImplemented
        return self + (-value)

    def __rsub__(self, other: object) -> "Dyadic":
        value = self._coerce(other)
        if value is NotImplemented:
            return NotImplemented
        return value - self

    def divide_int(self, divisor: int) -> "Dyadic":
        if isinstance(divisor, bool) or not isinstance(divisor, numbers.Integral):
            raise TypeError("divisor must be an integer")
        divisor = int(divisor)
        if divisor <= 0 or divisor & (divisor - 1):
            raise ValueError("exact Dyadic division requires a positive power of two")
        return Dyadic(self.mantissa, self.exponent - (divisor.bit_length() - 1))

    def __float__(self) -> float:
        return math.ldexp(float(self.mantissa), self.exponent)


class _TickOps:
    """Plain-integer arithmetic at one conservatively exact binary tick."""

    def __init__(self, tick_exponent: int) -> None:
        self.tick_exponent = tick_exponent

    def convert(self, value: object) -> int:
        dyadic = Dyadic.from_value(value)
        shift = dyadic.exponent - self.tick_exponent
        if shift < 0:
            raise ValueError("tick exponent does not exactly represent an input")
        return dyadic.mantissa << shift

    def zero(self) -> int:
        return 0

    def divide_int(self, value: int, divisor: int) -> int:
        quotient, remainder = divmod(value, divisor)
        if remainder:
            raise ArithmeticError(
                "integer tick bound was insufficient for an exact event time"
            )
        return quotient


def _tick_exponent(graph: LaneGraphProtocol, policy: UFPolicy) -> int:
    concrete = UFLaneGraph.from_protocol(graph)
    exponents = [Dyadic.from_value(edge.weight).exponent for edge in concrete.edges]
    exponents.append(Dyadic.from_value(policy.tau).exponent)
    # Every new factor of two in an event time accompanies a two-active-side
    # closure.  Along a lane trajectory there are at most num_vertices-1
    # successful component unions; one extra bit covers the initial closure.
    return min(exponents, default=0) - concrete.num_vertices - 1


def _from_ticks(value: object, exponent: int) -> object:
    if isinstance(value, int) and not isinstance(value, bool):
        return Dyadic(value, exponent)
    return value


def _convert_tick_outcome(outcome: LaneOutcome, exponent: int) -> LaneOutcome:
    completed = tuple(
        dataclasses.replace(
            component,
            exact_margin=(
                None
                if component.exact_margin is None
                else _from_ticks(component.exact_margin, exponent)
            ),
            event_batch_times=tuple(
                _from_ticks(value, exponent) for value in component.event_batch_times
            ),
            last_membership_event_time=_from_ticks(
                component.last_membership_event_time, exponent
            ),
            maximum_incident_half_edge_charge=_from_ticks(
                component.maximum_incident_half_edge_charge, exponent
            ),
        )
        for component in outcome.completed_components
    )
    censored = tuple(
        dataclasses.replace(
            component,
            event_batch_times=tuple(
                _from_ticks(value, exponent) for value in component.event_batch_times
            ),
            last_membership_event_time=_from_ticks(
                component.last_membership_event_time, exponent
            ),
            maximum_incident_half_edge_charge=_from_ticks(
                component.maximum_incident_half_edge_charge, exponent
            ),
        )
        for component in outcome.censored_components
    )
    return dataclasses.replace(
        outcome,
        completed_components=completed,
        censored_components=censored,
        terminal_event_time=_from_ticks(outcome.terminal_event_time, exponent),
    )


@dataclasses.dataclass(frozen=True)
class CompiledUFLane:
    """Immutable production lane with graph-invariant exact data cached."""

    graph: UFLaneGraph
    policy: UFPolicy
    tick_exponent: int
    tick_weights: tuple[int, ...]
    tick_tau: int
    adjacency: tuple[tuple[int, ...], ...]

    def run(self, defects: Iterable[int]) -> LaneOutcome:
        """Runs one defect pattern without recompiling lane topology or weights."""

        ops = _TickOps(self.tick_exponent)
        outcome = _run_lane_persistent(
            self.graph,
            defects,
            self.policy,
            ops=ops,
            prepared_graph=self.graph,
            prepared_weights=self.tick_weights,
            prepared_tau=self.tick_tau,
            prepared_adjacency=self.adjacency,
        )
        return _convert_tick_outcome(outcome, self.tick_exponent)


def compile_lane(graph: LaneGraphProtocol, policy: UFPolicy) -> CompiledUFLane:
    """Compiles graph-invariant data for repeated exact production runs."""

    if not isinstance(policy, UFPolicy):
        raise TypeError("policy must be UFPolicy")
    concrete = UFLaneGraph.from_protocol(graph)
    exponent = _tick_exponent(concrete, policy)
    ops = _TickOps(exponent)
    adjacency_lists: list[list[int]] = [
        [] for _ in range(concrete.num_vertices)
    ]
    for k, edge in enumerate(concrete.edges):
        adjacency_lists[edge.source].append(k)
        if edge.target is not None:
            adjacency_lists[edge.target].append(k)
    return CompiledUFLane(
        graph=concrete,
        policy=policy,
        tick_exponent=exponent,
        tick_weights=tuple(ops.convert(edge.weight) for edge in concrete.edges),
        tick_tau=ops.convert(policy.tau),
        adjacency=tuple(tuple(values) for values in adjacency_lists),
    )


def run_lane(
    graph: LaneGraphProtocol,
    defects: Iterable[int],
    policy: UFPolicy,
) -> LaneOutcome:
    """Runs the exact-dyadic production lifecycle on one lane."""

    return compile_lane(graph, policy).run(defects)


__all__ = [
    "BudgetExceeded",
    "BudgetLimits",
    "CensoredComponent",
    "CompiledUFLane",
    "CompletedComponent",
    "Dyadic",
    "LaneGraphProtocol",
    "LaneOutcome",
    "UFCounters",
    "UFEdge",
    "UFLaneGraph",
    "UFPolicy",
    "compile_lane",
    "run_lane",
]
