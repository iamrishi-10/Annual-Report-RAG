"""
JSON writing utilities.

This module provides reusable helpers for saving dictionaries and lists as
readable JSON files.
"""

import json
from pathlib import Path

from src.utils.logger import get_logger


logger = get_logger(__name__)


def save_json(
    data: dict | list,
    output_path: Path | str,
) -> None:
    """Save a dictionary or list as a JSON file.

    Args:
        data: The dictionary or list to save.
        output_path: Destination path for the JSON file.

    Raises:
        TypeError: If data is not a dictionary or list.
    """
    resolved_path = Path(output_path)

    try:
        if not isinstance(data, (dict, list)):
            raise TypeError("JSON data must be a dictionary or list.")

        resolved_path.parent.mkdir(parents=True, exist_ok=True)

        with resolved_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)

        logger.info("Saved JSON file to %s", resolved_path)

    except Exception:
        logger.exception("Failed to save JSON file to %s", resolved_path)
        raise
