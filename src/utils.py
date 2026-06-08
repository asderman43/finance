from typing import Any
import datetime as dt
from pathlib import Path

def _extract(section: dict, key: str, where: str, errors: list[str]) -> Any:
    """Pull ``key`` from ``section``; record an error instead of raising."""
    if not isinstance(section, dict) or key not in section:
        errors.append(f"missing required key: '{where}.{key}'")
        return None
    return section[key]
 
 
def _coerce_date(value: Any, where: str, errors: list[str]) -> dt.date | None:
    """Accept a date, a datetime, or an ISO 'YYYY-MM-DD' string."""
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str):
        try:
            return dt.date.fromisoformat(value.strip())
        except ValueError:
            errors.append(
                f"'{where}' is not a valid ISO date (YYYY-MM-DD): {value!r}"
            )
            return None
    errors.append(f"'{where}' must be a date, got {type(value).__name__}")
    return None


def _resolve_dir(dir_str: str | None) -> Path | None:
    """Anchor a config dir under the current working directory.

    Strips leading slashes so an absolute-looking path like
    '/logs/regimes/' doesn't resolve to the drive root
    (C:\\logs\\regimes on Windows) — it becomes <cwd>/logs/regimes.
    """
    if dir_str is None:
        return None
    return Path.cwd() / str(dir_str).lstrip("/\\")