import random
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from src.regimes.config import Config as RegimeConfig
from hmmlearn.hmm import GaussianHMM


class HMMTrainer:
    def __init__(
        self,
        cfg = RegimeConfig,
        logger = None
    ):
        self.covariance_type = cfg.setup.type
        self.n_iter = cfg.setup.n_iter
        self.num_rand = cfg.setup.num_rand
        self.best_model = None
        self.models = []
        self.logger = logger if logger is not None else self._build_logger(cfg.logging.dir, cfg.logging.level)

    def _build_logger(self, log_dir, log_level):
        logger = logging.getLogger(f"{__name__}.HMMTrainer")
        logger.setLevel(getattr(logging, log_level.upper(), logging.ERROR))

        # avoid duplicate handlers if instantiated more than once
        if not logger.handlers:
            fmt = logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
            )
            sh = logging.StreamHandler()
            sh.setFormatter(fmt)
            logger.addHandler(sh)

            if log_dir:
                Path(log_dir).mkdir(parents=True, exist_ok=True)
                fh = logging.FileHandler(Path(log_dir) / "training.log")
                fh.setFormatter(fmt)
                logger.addHandler(fh)

        return logger

    def train_one(self, X, num_class, seed):
        self.logger.info(
            "Fitting HMM: n_components=%d, cov=%s, seed=%d, n_iter=%d",
            num_class, self.covariance_type, seed, self.n_iter,
        )
        model = GaussianHMM(
            n_components=num_class,
            covariance_type=self.covariance_type,
            random_state=int(seed),
            n_iter=self.n_iter,
            min_covar=1e-2
        )

        try:
            model.fit(X)
        except Exception:
            self.logger.exception(
                "Fit failed: n_components=%d, seed=%d", num_class, seed
            )
            raise

        if not model.monitor_.converged:
            self.logger.warning(
                "Did not converge: n_components=%d, seed=%d (%d iters)",
                num_class, seed, self.n_iter,
            )

        result = {
            "model": model,
            "n_components": num_class,
            "seed": seed,
            "bic": model.bic(X),
            "aic": model.aic(X),
            "score": model.score(X),
            "converged": model.monitor_.converged,
        }
        self.logger.info(
            "Done: n_components=%d, seed=%d -> score=%.4f, bic=%.4f, aic=%.4f",
            num_class, seed, result["score"], result["bic"], result["aic"],
        )
        return result

    def train_multiple(self, X, class_min, class_max):
        self.models = []
        self.best_model = None

        for num_class in range(class_min, class_max + 1):
            seeds = random.sample(range(1_000_000), k=self.num_rand)
            for seed in seeds:
                new_model = self.train_one(X, num_class, seed)
                self.models.append(new_model)
                if self.best_model is None or new_model["bic"] < self.best_model["bic"]:
                    self.logger.info(
                        "New best: n_components=%d, seed=%d, bic=%.4f",
                        num_class, seed, new_model["bic"],
                    )
                    self.best_model = new_model

        self.logger.info(
            "Search complete. Best: n_components=%d, bic=%.4f",
            self.best_model["n_components"], self.best_model["bic"],
        )
        return {"models": self.models}
    
    
    
    def evaluate(self, model, Y, forward_returns):
        preds = np.asarray(model.predict(Y))
        if len(preds) != len(Y):
            raise ValueError(
                f"predict() returned {len(preds)} labels but features has "
                f"{len(Y)} rows — Y and features are misaligned"
            )

        transmat = np.asarray(model.transmat_)
        diag = np.diag(transmat)
        # expected dwell time in each state = 1 / (1 - p_ii)
        with np.errstate(divide="ignore"):
            dwell = np.where(diag < 1, 1.0 / (1.0 - diag), np.inf)
        switch_rate = float((np.diff(preds) != 0).mean()) if len(preds) > 1 else 0.0
        
        
        score = model.score(Y)
        bic = model.bic(Y)
        aic = model.aic(Y)
        
        
        regime_stats: dict[str, dict] = {}
        if forward_returns is not None:
            if len(forward_returns) != len(preds):
                raise ValueError(
                    f"predict() returned {len(preds)} labels but forward_returns has "
                    f"{len(forward_returns)} rows — Y and forward_returns are misaligned"
                )
            
            
         
            dfv = pd.DataFrame({"regime": preds, "ret": forward_returns})
            agg = dfv.groupby("regime")["ret"].agg(["mean", "std", "count", "min", "max"])
            for r, row in agg.iterrows():
                regime_stats[str(int(r))] = {
                    "mean": float(row["mean"]),
                    "std": float(row["std"]),
                    "count": int(row["count"]),
                    "min": float(row["min"]),
                    "max": float(row["max"]),
                }
        
        # occupancy = share of days in each regime
        occ = pd.Series(preds).value_counts(normalize=True).sort_index()
        occupancy = {str(int(k)): float(v) for k, v in occ.items()}
        
        self.logger.info(
            "HMM eval: n_components=%d, bic=%.2f, switch_rate=%.1f%%",
            model.n_components, bic, switch_rate * 100,
        )
        
        return {
            "score": score,
            "bic": bic,
            "aic": aic,
            "switch_rate": switch_rate,
            "mean_dwell_days": [float(d) for d in dwell],
            "min_dwell_days": float(np.min(dwell)),
            "transition_matrix": transmat.round(4).tolist(),
            "occupancy": occupancy,
            "regime_stats": regime_stats,
        }