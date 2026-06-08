import datetime as dt
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import yaml

from src.utils import _coerce_date, _extract, _resolve_dir

# hmmlearn's accepted covariance types
VALID_COVARIANCE_TYPES = frozenset({"spherical", "diag", "full", "tied"})
# logging level names that map to real levels
VALID_LOG_LEVELS = frozenset(logging._nameToLevel.keys())


@dataclass(frozen=True)
class SetupConfig:
    num_rand: int
    class_min: int
    class_max: int
    type: str
    n_iter: int = 100


@dataclass(frozen=True)
class StockConfig:
    ticker: str
    start: dt.date
    end: dt.date


@dataclass(frozen=True)
class LoggingConfig:
    dir: Path        # was str
    level: str

@dataclass(frozen=True)
class OutputConfig:
    dir: Path        # was str

@dataclass(frozen=True)
class Config:
    setup: SetupConfig
    stock: StockConfig
    logging: LoggingConfig
    output: OutputConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def load_config(path: str | Path) -> Config:
    """Load, parse, and validate the YAML config at ``path``.

    Raises:
        Exception: if the file is missing, unparseable, or any field is
            missing/invalid. All problems found are reported together.
    """
    path = Path(path)
    if not path.is_file():
        raise Exception(f"config file not found: {path}")

    try:
        with path.open() as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise Exception(f"could not parse YAML in {path}: {e}") from e

    if not isinstance(raw, dict):
        raise Exception(f"{path}: top level must be a mapping/dict")

    errors: list[str] = []

    # ---- setup section ----
    setup_raw = raw.get("setup")
    if not isinstance(setup_raw, dict):
        errors.append("missing required section: 'setup'")
        setup_raw = {}

    num_rand = _extract(setup_raw, "num_rand", "setup", errors)
    class_min = _extract(setup_raw, "class_min", "setup", errors)
    class_max = _extract(setup_raw, "class_max", "setup", errors)
    cov_type = _extract(setup_raw, "type", "setup", errors)
    n_iter = setup_raw.get("n_iter", 100)  # optional, defaulted

    # type must be valid
    if cov_type is not None and cov_type not in VALID_COVARIANCE_TYPES:
        errors.append(
            f"setup.type {cov_type!r} is not valid; "
            f"choose one of {sorted(VALID_COVARIANCE_TYPES)}"
        )

    # numeric sanity
    if isinstance(num_rand, int) and num_rand < 1:
        errors.append(f"setup.num_rand must be >= 1, got {num_rand}")
    if isinstance(class_min, int) and class_min < 1:
        errors.append(f"setup.class_min must be >= 1, got {class_min}")
    if (
        isinstance(class_min, int)
        and isinstance(class_max, int)
        and class_min > class_max
    ):
        errors.append(
            f"setup.class_min ({class_min}) must be <= setup.class_max ({class_max})"
        )

    # ---- stock section ----
    stock_raw = raw.get("stock")
    if not isinstance(stock_raw, dict):
        errors.append("missing required section: 'stock'")
        stock_raw = {}

    ticker = _extract(stock_raw, "ticker", "stock", errors)
    start_raw = _extract(stock_raw, "start", "stock", errors)
    end_raw = _extract(stock_raw, "end", "stock", errors)

    if ticker is not None and (not isinstance(ticker, str) or not str(ticker).strip()):
        errors.append(f"stock.ticker must be a non-empty string, got {ticker!r}")

    today = dt.date.today()
    start = _coerce_date(start_raw, "stock.start", errors)
    end = _coerce_date(end_raw, "stock.end", errors)

    if start is not None and start > today:
        errors.append(
            f"stock.start {start.isoformat()} is in the future "
            f"(today is {today.isoformat()})"
        )
    if end is not None and end > today:
        errors.append(
            f"stock.end {end.isoformat()} is in the future "
            f"(today is {today.isoformat()})"
        )
    if start is not None and end is not None and start > end:
        errors.append(
            f"stock.start ({start.isoformat()}) must be <= "
            f"stock.end ({end.isoformat()})"
        )

    # ---- logging section ----
    logging_raw = raw.get("logging")
    if not isinstance(logging_raw, dict):
        errors.append("missing required section: 'logging'")
        logging_raw = {}

    log_dir = _extract(logging_raw, "dir", "logging", errors)
    log_level = _extract(logging_raw, "level", "logging", errors)

    if log_level is not None and str(log_level).upper() not in VALID_LOG_LEVELS:
        errors.append(
            f"logging.level {log_level!r} is not valid; "
            f"choose one of {sorted(VALID_LOG_LEVELS)}"
        )
    output_raw = raw.get("output")
    if not isinstance(output_raw, dict):
        errors.append("missing required section: 'output'")
        output_raw = {}

    output_dir = _extract(output_raw, "dir", "output", errors)
    # ---- raise everything at once ----
    if errors:
        bullet = "\n  - ".join(errors)
        raise Exception(
            f"invalid config in {path}:\n  - {bullet}"
        )

    return Config(
        setup=SetupConfig(
            num_rand=num_rand,
            class_min=class_min,
            class_max=class_max,
            type=cov_type,
            n_iter=n_iter,
        ),
        stock=StockConfig(
            ticker=str(ticker).strip(),
            start=start,
            end=end,
        ),
        logging=LoggingConfig(
            dir=_resolve_dir(log_dir),
            level=str(log_level).upper(),
        ),
        output=OutputConfig(dir=_resolve_dir(output_dir)),
    )


if __name__ == "__main__":
    import sys

    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
    print(cfg)