from src.regimes.config import load_config
from src.regimes.trainer import HMMTrainer
from src.regimes.archive import RegimeRunArchive


def run_regime_search(cfg, X, features, Y, scaler, forward_returns=None):
    archive = RegimeRunArchive(cfg.output.dir)
    run = archive.new_run(cfg)

    trainer = HMMTrainer(cfg, logger=run.logger)
    out = trainer.train_multiple(X, cfg.setup.class_min, cfg.setup.class_max)

    results = out["models"]                 # list of result dicts
    evals = [
        trainer.evaluate(r["model"], Y, forward_returns)   # pass the model, not the dict
        for r in results
    ]

    archive.save(
        run,
        results=results,
        evaluations=evals,
        scaler=scaler,
        features=features,
        cfg=cfg,
    )
    return out, run.dir