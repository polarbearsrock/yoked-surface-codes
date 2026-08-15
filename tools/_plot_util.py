from typing import Any, Dict, Iterable

import sinter


class FieldToMetadataWrapper:
    """Provides attribute-style access to JSON metadata used in CLI filters."""

    def __init__(self, metadata: Any):
        self._metadata = metadata

    def __getattr__(self, item: str) -> Any:
        if isinstance(self._metadata, dict):
            return self._metadata.get(item)
        return None


def common_json_properties(stats: Iterable[sinter.TaskStats]) -> Dict[str, Any]:
    """Returns scalar metadata fields that are identical across all stats."""

    stats = list(stats)
    values: Dict[str, set[Any]] = {}
    for stat in stats:
        if isinstance(stat.json_metadata, dict):
            for key in stat.json_metadata:
                values[key] = set()
    for stat in stats:
        if isinstance(stat.json_metadata, dict):
            for key in values:
                value = stat.json_metadata.get(key)
                if value is None or isinstance(value, (float, str, int)):
                    values[key].add(value)
    if "decoder" not in values:
        values["decoder"] = {stat.decoder for stat in stats}
    return {
        key: next(iter(possible_values))
        for key, possible_values in values.items()
        if len(possible_values) == 1
    }
