"""
Two open questions, answered on a panel whose ground truth I control.

Q1. Does a lag window earn its keep? The target is BUILT to depend on
    f[t], f[t-1] and f[t-2]. A model given only f[t] cannot reach the
    achievable R2 no matter how flexible it is; one given the window can.
    If lags do not win here they cannot win anywhere, because here the
    history is the signal by construction.

Q2. Does a SHARED-parameter multi-output model beat independent per-horizon
    fits? This is the only version of multi-output that could add anything:
    sklearn's MultiOutputRegressor fits one estimator per output, so it is
    arithmetically identical to running N experiments -- which this library
    already supports through TargetSpec.horizons. A neural net with N output
    heads genuinely shares a representation, so it is the case worth
    measuring before changing the OOS schema, the manifest and the
    portfolio pivot to accommodate it.
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd

os.environ.setdefault("SQT_RUNS_DIR", tempfile.mkdtemp(prefix="spike_runs_"))

from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402

from standard_quant_tools.modeling.agent.models import (  # noqa: E402
    RegisterExternalPanelInput,
    RunModelExperimentInput,
)
from standard_quant_tools.modeling.agent.tools import (  # noqa: E402
    register_external_panel,
    run_model_experiment,
)
from standard_quant_tools.modeling.specs import (  # noqa: E402
    EstimatorSpec,
    ModelSpec,
    ValidationSpec,
)

ENTITIES = ["E0", "E1", "E2", "E3", "E4", "E5"]
N_BARS = 500
rng = np.random.default_rng(20260903)


def build_panel() -> pd.DataFrame:
    """A panel whose label depends on the recent PATH of a feature."""
    dates = pd.bdate_range("2021-01-04", periods=N_BARS)
    rows = []
    for entity in ENTITIES:
        # An autocorrelated driver, so lags carry information the current
        # value does not fully contain.
        shocks = rng.normal(0, 1, N_BARS)
        f = pd.Series(shocks).ewm(alpha=0.4).mean().to_numpy()
        noise_feature = rng.normal(0, 1, N_BARS)

        f1 = pd.Series(f).shift(1).to_numpy()
        f2 = pd.Series(f).shift(2).to_numpy()
        # THE GROUND TRUTH. Three quarters of the signal lives in bars the
        # current value does not reveal.
        latent = 0.25 * f + 0.45 * np.nan_to_num(f1) + 0.30 * np.nan_to_num(f2)
        h1 = latent + rng.normal(0, 0.35, N_BARS)
        # Longer horizons driven by the SAME latent plus their own noise --
        # the structure a shared representation could exploit.
        h5 = latent + rng.normal(0, 0.55, N_BARS)
        h20 = latent + rng.normal(0, 0.85, N_BARS)

        rows.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "entity": entity,
                    "f": f,
                    "f_lag1": f1,
                    "f_lag2": f2,
                    "noise": noise_feature,
                    "y_h1": h1,
                    "y_h5": h5,
                    "y_h20": h20,
                }
            )
        )
    panel = pd.concat(rows, ignore_index=True).dropna().reset_index(drop=True)
    return panel


def register(panel: pd.DataFrame, path: str, feature_columns):
    panel.to_parquet(path, index=False)
    result = register_external_panel(
        RegisterExternalPanelInput(
            path=path,
            targets=[
                {"name": "h1", "column": "y_h1", "horizon": 1},
                {"name": "h5", "column": "y_h5", "horizon": 5},
                {"name": "h20", "column": "y_h20", "horizon": 20},
            ],
            feature_columns=list(feature_columns),
        )
    )
    return result.dataset_id


def experiment(dataset_id, estimator_type, params, target="h1"):
    return run_model_experiment(
        RunModelExperimentInput(
            dataset_id=dataset_id,
            target=target,
            spec=ModelSpec(
                task="regression",
                estimator=EstimatorSpec(type=estimator_type, params=params),
                validation=ValidationSpec(train_window=250, test_window=60, embargo=5),
                random_seed=11,
            ),
        )
    )


def headline(result):
    metrics = result.oos_metrics or {}
    return (
        float(metrics.get("r2", float("nan"))),
        float(metrics.get("ic", metrics.get("spearman_ic", float("nan")))),
        int(result.n_folds),
    )


def main() -> None:
    panel = build_panel()
    tmp = tempfile.mkdtemp(prefix="spike_")
    print(f"panel: {len(panel):,} rows, {len(ENTITIES)} entities\n")

    narrow = register(panel, os.path.join(tmp, "narrow.parquet"), ["f", "noise"])
    wide = register(
        panel,
        os.path.join(tmp, "wide.parquet"),
        ["f", "f_lag1", "f_lag2", "noise"],
    )

    print("Q1  DOES THE LAG WINDOW EARN ITS KEEP")
    print(f"{'model':34} {'OOS r2':>9} {'IC':>8} {'folds':>6}")
    trials = [
        ("ridge, f only", narrow, "ridge", {"alpha": 1.0}),
        ("ridge, f + 2 lags", wide, "ridge", {"alpha": 1.0}),
        (
            "mlp,   f only",
            narrow,
            "mlp",
            {"n_hidden_units": 32, "max_iter": 600, "random_state": 0},
        ),
        (
            "mlp,   f + 2 lags",
            wide,
            "mlp",
            {"n_hidden_units": 32, "max_iter": 600, "random_state": 0},
        ),
        (
            "sgd,   f + 2 lags",
            wide,
            "sgd",
            {"loss": "huber", "alpha": 1e-4, "max_iter": 2000, "random_state": 0},
        ),
    ]
    for label, dataset_id, estimator, params in trials:
        r2, ic, folds = headline(experiment(dataset_id, estimator, params))
        print(f"{label:34} {r2:9.4f} {ic:8.4f} {folds:6d}")

    print("\nQ2  SHARED MULTI-OUTPUT vs INDEPENDENT PER-HORIZON FITS")
    features = ["f", "f_lag1", "f_lag2", "noise"]
    targets = ["y_h1", "y_h5", "y_h20"]
    ordered = panel.sort_values(["date", "entity"]).reset_index(drop=True)
    split = int(len(ordered) * 0.7)
    train, test = ordered.iloc[:split], ordered.iloc[split:]
    X_train = train[features].to_numpy()
    X_test = test[features].to_numpy()
    Y_train = train[targets].to_numpy()
    Y_test = test[targets].to_numpy()

    # Standardized the way the engine does, fit on train only.
    mean, std = X_train.mean(0), X_train.std(0)
    std[std == 0] = 1.0
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    print(f"{'model':34} " + " ".join(f"{t:>9}" for t in targets))

    independent = []
    for index in range(3):
        model = MLPRegressor(
            hidden_layer_sizes=(32,), max_iter=600, random_state=0
        ).fit(X_train, Y_train[:, index])
        independent.append(r2_score(Y_test[:, index], model.predict(X_test)))
    print("3 independent MLPs" + " " * 16 + " ".join(f"{v:9.4f}" for v in independent))

    shared = MLPRegressor(hidden_layer_sizes=(32,), max_iter=600, random_state=0).fit(
        X_train, Y_train
    )
    shared_scores = [
        r2_score(Y_test[:, i], shared.predict(X_test)[:, i]) for i in range(3)
    ]
    print(
        "1 shared-head MLP (3 outputs)"
        + " " * 5
        + " ".join(f"{v:9.4f}" for v in shared_scores)
    )

    ridge_scores = []
    for index in range(3):
        model = Ridge(alpha=1.0).fit(X_train, Y_train[:, index])
        ridge_scores.append(r2_score(Y_test[:, index], model.predict(X_test)))
    print(
        "3 independent ridges" + " " * 14 + " ".join(f"{v:9.4f}" for v in ridge_scores)
    )

    delta = np.mean(shared_scores) - np.mean(independent)
    print(f"\nshared minus independent, mean over horizons: {delta:+.4f}")


if __name__ == "__main__":
    sys.exit(main())
