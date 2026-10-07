"""
Serialization utilities.

This module converts internal Python objects, such as dataclasses, into
JSON-safe dictionaries and lists. It does not write files.
"""

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def to_json_safe(data: Any) -> Any:
    """Convert supported Python objects into JSON-safe values.

    Args:
        data: A dataclass instance, dictionary, list, tuple, Path, or primitive
            JSON-compatible value.

    Returns:
        A JSON-safe Python value made from dictionaries, lists, strings,
        numbers, booleans, or None.

    Raises:
        TypeError: If the object cannot be converted into a JSON-safe value.
    """
    if is_dataclass(data) and not isinstance(data, type):
        return to_json_safe(asdict(data))

    if isinstance(data, dict):
        return {
            str(key): to_json_safe(value)
            for key, value in data.items()
        }

    if isinstance(data, (list, tuple)):
        return [to_json_safe(item) for item in data]

    if isinstance(data, Path):
        return str(data)

    if isinstance(data, (str, int, float, bool)) or data is None:
        return data

    raise TypeError(
        f"Object of type {type(data).__name__} is not JSON serializable."
    )
