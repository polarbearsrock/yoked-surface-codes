"""Generic authenticated-row field access and scalar coercion.

This slice of :mod:`yoked.decoding.oracle.policy_analysis` provides the
path/alias lookups and fail-closed scalar coercions used when reading fields
out of immutable ledger rows.  It inherits the package's downstream-only
contract: it never imports circuit generation, sampling, matching, or
decoding code.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ._contract import PolicyAnalysisError


def _record_value(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in row:
            return row[name]
    return None


def _deep_values(value: Any, key: str) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, Mapping):
        if key in value:
            found.append(value[key])
        for child in value.values():
            found.extend(_deep_values(child, key))
    elif isinstance(value, list):
        for child in value:
            found.extend(_deep_values(child, key))
    return found


def _one_deep(value: Any, names: Sequence[str], *, required: bool = False) -> Any:
    for name in names:
        found = _deep_values(value, name)
        if found:
            first = found[0]
            if any(item != first for item in found[1:]):
                raise PolicyAnalysisError(
                    f"ambiguous values for semantic field {name!r}"
                )
            return first
    if required:
        raise PolicyAnalysisError(
            f"missing semantic field; expected one of {list(names)}"
        )
    return None


def _at(row: Mapping[str, Any], *paths: str, required: bool = False) -> Any:
    for path in paths:
        current: Any = row
        ok = True
        for part in path.split("."):
            if not isinstance(current, Mapping) or part not in current:
                ok = False
                break
            current = current[part]
        if ok:
            return current
    if required:
        raise PolicyAnalysisError(f"missing semantic field; expected one of {paths}")
    return None


def _as_nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise PolicyAnalysisError(f"{name} must be an integer")
    value = int(value)
    if value < 0:
        raise PolicyAnalysisError(f"{name} must be nonnegative")
    return value


def _as_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise PolicyAnalysisError(f"{name} must be Boolean")
    return value


def _float_value(row: Mapping[str, Any], name: str) -> float | None:
    raw = _at(row, name, f"evaluation.{name}", f"oracle.{name}")
    raw_hex = _at(row, f"{name}_hex", f"evaluation.{name}_hex", f"oracle.{name}_hex")
    if raw is None and raw_hex is None:
        return None
    if raw_hex is not None:
        if not isinstance(raw_hex, str):
            raise PolicyAnalysisError(f"{name}_hex must be a string")
        try:
            exact = float.fromhex(raw_hex)
        except ValueError as ex:
            raise PolicyAnalysisError(f"invalid {name}_hex") from ex
        if raw is not None and float(raw) != exact:
            raise PolicyAnalysisError(f"{name} and {name}_hex disagree")
        result = exact
    else:
        result = float(raw)
    if not math.isfinite(result):
        raise PolicyAnalysisError(f"{name} must be finite")
    return result


def _required_float(row: Mapping[str, Any], name: str) -> float:
    value = _float_value(row, name)
    if value is None:
        raise PolicyAnalysisError(f"missing required finite float {name!r}")
    return value
