import os
from typing import Optional

from font_dataset.font import load_font_with_exclusion


INPUT_SIZE = 512

_DEFAULT_FONT_CONFIG = "configs/font.yml"
_DEFAULT_FONT_CACHE = "font_list_cache.bin"


def compute_font_count(
    config_path: Optional[str] = None, cache_path: Optional[str] = None
) -> int:
    """Return the number of fonts defined by the current dataset configuration."""
    resolved_config = config_path or os.environ.get(
        "FONT_CONFIG_PATH", _DEFAULT_FONT_CONFIG
    )
    resolved_cache = cache_path or os.environ.get(
        "FONT_CACHE_PATH", _DEFAULT_FONT_CACHE
    )
    try:
        font_mapping = load_font_with_exclusion(
            config_path=resolved_config, cache_path=resolved_cache
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Font config file not found at '{resolved_config}'."
        ) from exc

    if not font_mapping:
        raise RuntimeError(
            "No fonts were found. Verify the dataset configuration and contents."
        )
    return len(font_mapping)


def refresh_font_count(
    config_path: Optional[str] = None, cache_path: Optional[str] = None
) -> int:
    """Recalculate FONT_COUNT; useful when datasets change at runtime."""
    global FONT_COUNT
    FONT_COUNT = compute_font_count(
        config_path=config_path, cache_path=cache_path
    )
    return FONT_COUNT


FONT_COUNT = refresh_font_count()
