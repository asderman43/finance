import datetime as dt
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


@dataclass
class RunHandle:
    """Returned by ``new_run`` — points at the run folder and its logger."""
    dir: Path
    run_id: str
    logger: logging.Logger


class RegimeRunArchive:
    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)

    # ── creating a run ──────────────────────────────────────────────────────

    def new_run(self, cfg, *, log_level: str = "INFO") -> RunHandle:
        """Create a fresh timestamped run folder and a logger writing into it.

        Pass ``handle.logger`` to HMMTrainer so the training logs are captured
        in the same folder as the model and metrics.
        """
        ticker = cfg.stock.ticker
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

        # guard against two runs in the same second
        run_dir = self.base_dir / ticker / stamp
        suffix = 1
        while run_dir.exists():
            run_dir = self.base_dir / ticker / f"{stamp}_{suffix}"
            suffix += 1
        run_dir.mkdir(parents=True, exist_ok=True)

        run_id = run_dir.name
        logger = logging.getLogger(f"regime_run.{ticker}.{run_id}")
        logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        # fresh file handler for this run only
        fh = logging.FileHandler(run_dir / "run.log")
        fh.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(fh)
        # also echo to console
        if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
            logger.addHandler(logging.StreamHandler())

        logger.info("Run %s started for %s", run_id, ticker)
        return RunHandle(dir=run_dir, run_id=run_id, logger=logger)

    # ── saving a run ────────────────────────────────────────────────────────

    def save(
        self,
        handle: RunHandle,
        results: list[dict],
        evaluations: list[dict],
        *,
        features: pd.DataFrame,
        scaler=None,
        cfg=None,
        extra: dict | None = None,
    ) -> Path:
        """Save many models 1:1 with their evals.

        Each model goes in ``<run_dir>/seed_<seed>_<n_components>/`` with its own
        model.pkl + metrics.json. A run-level config.yaml and an index.json
        summarising all sub-runs are written at the run root. Returns run_dir.
        """
        if len(results) != len(evaluations):
            raise ValueError(
                f"{len(results)} results but {len(evaluations)} evaluations "
                "— they must be 1:1"
            )

        run_dir = handle.dir
        feature_names = list(features.columns)
        index_rows = []

        for result, evaluation in zip(results, evaluations):
            model = result["model"]
            seed = result.get("seed")
            n_components = int(model.n_components)

            sub_name = f"seed_{seed}_{n_components}" if seed is not None else f"nc_{n_components}"
            sub_dir = run_dir / sub_name
            suffix = 1
            while sub_dir.exists():
                sub_dir = run_dir / f"{sub_name}_{suffix}"
                suffix += 1
            sub_dir.mkdir(parents=True, exist_ok=True)

            converged = None
            if "converged" in result:
                converged = bool(result["converged"])
            elif hasattr(model, "monitor_"):
                converged = bool(model.monitor_.converged)

            metrics: dict[str, Any] = {
                "run_id": handle.run_id,
                "sub_run": sub_dir.name,
                "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                "ticker": cfg.stock.ticker if cfg else None,
                "start": cfg.stock.start.isoformat() if cfg else None,
                "end": cfg.stock.end.isoformat() if cfg else None,
                "n_components": n_components,
                "covariance_type": getattr(model, "covariance_type", None),
                "n_iter": cfg.setup.n_iter if cfg else None,
                "seed": int(seed) if seed is not None else None,
                "feature_names": feature_names,
                "converged": converged,
                **evaluation,
            }
            if extra:
                metrics.update(extra)

            joblib.dump(
                {"model": model, "scaler": scaler, "feature_names": feature_names},
                sub_dir / "model.pkl",
            )
            with (sub_dir / "metrics.json").open("w") as f:
                json.dump(metrics, f, indent=2)

            handle.logger.info(
                "Saved %s: n_components=%d, bic=%.2f, switch_rate=%.1f%%",
                sub_dir.name, n_components, evaluation["bic"], evaluation["switch_rate"] * 100,
            )
            index_rows.append({
                "sub_run": sub_dir.name,
                "n_components": n_components,
                "seed": int(seed) if seed is not None else None,
                "bic": evaluation.get("bic"),
                "aic": evaluation.get("aic"),
                "score": evaluation.get("score"),
                "switch_rate": evaluation.get("switch_rate"),
                "converged": converged,
            })

        # run-level config + index over all sub-runs
        if cfg is not None:
            import yaml
            raw = json.loads(json.dumps(cfg.to_dict(), default=str))
            with (run_dir / "config.yaml").open("w") as f:
                yaml.safe_dump(raw, f, default_flow_style=False, sort_keys=False)

        index_rows.sort(key=lambda r: (r["bic"] is None, r["bic"]))
        with (run_dir / "index.json").open("w") as f:
            json.dump(index_rows, f, indent=2)

        for h in list(handle.logger.handlers):
            if isinstance(h, logging.FileHandler):
                h.flush()
        return run_dir

    # ── comparing / loading runs ────────────────────────────────────────────

    def list_runs(self, ticker: str | None = None) -> pd.DataFrame:
        """Read every sub-run's metrics.json into one comparison table."""
        pattern = f"{ticker}/*/*/metrics.json" if ticker else "*/*/*/metrics.json"
        rows = []
        for mpath in sorted(self.base_dir.glob(pattern)):
            try:
                m = json.loads(mpath.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            rows.append({
                "run_id": m.get("run_id"),
                "sub_run": m.get("sub_run"),
                "ticker": m.get("ticker"),
                "n_components": m.get("n_components"),
                "cov_type": m.get("covariance_type"),
                "bic": m.get("bic"),
                "aic": m.get("aic"),
                "score": m.get("score"),
                "switch_rate": m.get("switch_rate"),
                "min_dwell_days": m.get("min_dwell_days"),
                "converged": m.get("converged"),
                "seed": m.get("seed"),
                "start": m.get("start"),
                "end": m.get("end"),
            })
        df = pd.DataFrame(rows)
        return df.sort_values("bic").reset_index(drop=True) if not df.empty else df
    
    def load_run(self, ticker: str, run_id: str) -> dict:
        """Load the best (lowest-bic) sub-run of a run."""
        run_dir = self.base_dir / ticker / run_id
        index_path = run_dir / "index.json"
        if not index_path.is_file():
            raise FileNotFoundError(f"no index at {index_path}")
        index = json.loads(index_path.read_text())
        if not index:
            raise ValueError(f"empty index for run {run_id}")
        best = index[0]  # already sorted by bic ascending in save()
        return self.load_sub_run(ticker, run_id, best["sub_run"])
    
    
    def load_sub_run(self, ticker: str, run_id: str, sub_run: str) -> dict:
        sub_dir = self.base_dir / ticker / run_id / sub_run
        if not sub_dir.is_dir():
            raise FileNotFoundError(f"no sub-run at {sub_dir}")
        bundle = joblib.load(sub_dir / "model.pkl")
        bundle["metrics"] = json.loads((sub_dir / "metrics.json").read_text())
        return bundle