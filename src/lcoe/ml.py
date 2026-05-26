from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

NUMERIC_FEATURES = (
    "installed_capacity_MW",
    "distance_from_shore_km",
    "mean_hub_wind_speed",
    "capacity_factor",
    "project_lifetime_years",
    "commissioning_year",
)
CATEGORICAL_FEATURES = ("country", "foundation_type", "grid_connection_model")
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _feature_importance(pipeline: Any, top_n: int = 8) -> list[dict[str, Any]]:
    regressor = pipeline.named_steps["regressor"]
    preprocessor = pipeline.named_steps["preprocessor"]
    importances = getattr(regressor, "feature_importances_", None)
    if importances is None:
        return []
    try:
        names = preprocessor.get_feature_names_out()
    except (AttributeError, ValueError):
        names = np.array(MODEL_FEATURES, dtype=object)
    pairs = sorted(zip(names, importances, strict=False), key=lambda item: item[1], reverse=True)
    return [
        {"feature": str(name).replace("num__", "").replace("cat__", ""), "importance": round(float(value), 4)}
        for name, value in pairs[:top_n]
    ]


def _mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = np.isfinite(actual) & np.isfinite(predicted) & (actual > 0)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100.0)


def _group_errors(frame: pd.DataFrame, group_col: str) -> list[dict[str, Any]]:
    if group_col not in frame.columns:
        return []
    rows: list[dict[str, Any]] = []
    for group, group_df in frame.dropna(subset=[group_col]).groupby(group_col):
        if len(group_df) < 2:
            continue
        rows.append(
            {
                "group": str(group),
                "rows": int(len(group_df)),
                "MAE_EUR_per_MW": round(float(np.mean(np.abs(group_df["actual"] - group_df["predicted"]))), 0),
                "MAPE_percent": round(_mape(group_df["actual"].to_numpy(), group_df["predicted"].to_numpy()), 2),
            }
        )
    return sorted(rows, key=lambda row: row["MAE_EUR_per_MW"], reverse=True)[:8]


def build_capex_ml_report(
    df: pd.DataFrame,
    project_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.dummy import DummyRegressor
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import mean_absolute_error, r2_score
        from sklearn.model_selection import cross_val_score, train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.preprocessing import StandardScaler
    except ModuleNotFoundError as exc:
        return {
            "status": "unavailable",
            "reason": f"{exc.name} is not installed; run dependency sync before training ML baseline",
        }

    data = df.copy()
    if "budget_quality_status" in data.columns:
        data = data[data["budget_quality_status"].eq("valid")].copy()
    if "budget_verification_level" in data.columns:
        data = data[data["budget_verification_level"].isin(["A", "B"])].copy()

    if "budget_EUR_per_MW" in data.columns:
        target = pd.to_numeric(data["budget_EUR_per_MW"], errors="coerce")
    else:
        target = (
            pd.to_numeric(data.get("total_budget_EUR_2026"), errors="coerce")
            / pd.to_numeric(data.get("installed_capacity_MW"), errors="coerce")
        )

    data = data.assign(
        budget_EUR_per_MW=target,
        target_log_capex_per_mw=np.log(target.where(target > 0)),
    )
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["target_log_capex_per_mw"])
    data = data[data["budget_EUR_per_MW"].between(500_000.0, 35_000_000.0)]

    available_features = [feature for feature in MODEL_FEATURES if feature in data.columns]
    if len(data) < 20 or not available_features:
        return {
            "status": "insufficient_data",
            "training_rows": int(len(data)),
            "reason": "at least 20 validated rows are required for the ML baseline",
        }

    numeric_features = [feature for feature in NUMERIC_FEATURES if feature in available_features]
    categorical_features = [feature for feature in CATEGORICAL_FEATURES if feature in available_features]
    x = data[available_features]
    y = data["target_log_capex_per_mw"]

    tree_preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )
    linear_preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
    actual = np.exp(y_test)
    cv_folds = min(5, max(2, len(data) // 12))

    candidates = {
        "baseline_median": Pipeline(
            steps=[
                ("preprocessor", tree_preprocessor),
                ("regressor", DummyRegressor(strategy="median")),
            ]
        ),
        "ridge_regression": Pipeline(
            steps=[
                ("preprocessor", linear_preprocessor),
                ("regressor", Ridge(alpha=1.0)),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", tree_preprocessor),
                (
                    "regressor",
                    RandomForestRegressor(
                        n_estimators=300,
                        min_samples_leaf=2,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            steps=[
                ("preprocessor", tree_preprocessor),
                (
                    "regressor",
                    HistGradientBoostingRegressor(
                        max_iter=200,
                        min_samples_leaf=6,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }

    model_rows: list[dict[str, Any]] = []
    fitted_models: dict[str, Any] = {}
    test_predictions: dict[str, np.ndarray] = {}
    for name, candidate in candidates.items():
        candidate.fit(x_train, y_train)
        fitted_models[name] = candidate
        predictions_log = candidate.predict(x_test)
        predictions = np.exp(predictions_log)
        test_predictions[name] = predictions
        cv_scores = cross_val_score(
            candidate,
            x,
            y,
            scoring="neg_mean_absolute_error",
            cv=cv_folds,
        )
        model_rows.append(
            {
                "model": name,
                "test_MAE_EUR_per_MW": round(float(mean_absolute_error(actual, predictions)), 0),
                "test_MAPE_percent": round(_mape(actual.to_numpy(), predictions), 2),
                "test_R2_log_target": round(float(r2_score(y_test, predictions_log)), 3),
                "cv_MAE_log_target": round(float(-cv_scores.mean()), 3),
            }
        )

    comparison = sorted(model_rows, key=lambda row: row["test_MAE_EUR_per_MW"])
    baseline_row = next(row for row in model_rows if row["model"] == "baseline_median")
    best_row = comparison[0]
    best_name = str(best_row["model"])
    best_model = fitted_models[best_name]
    best_predictions = test_predictions[best_name]
    error_frame = x_test.copy()
    error_frame["actual"] = actual.to_numpy()
    error_frame["predicted"] = best_predictions

    feature_importance = _feature_importance(best_model)
    if not feature_importance and "random_forest" in fitted_models:
        feature_importance = _feature_importance(fitted_models["random_forest"])

    report: dict[str, Any] = {
        "status": "trained",
        "model": best_name,
        "target": "log(budget_EUR_per_MW)",
        "training_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "features": available_features,
        "model_comparison": comparison,
        "best_model": best_name,
        "baseline_MAE_EUR_per_MW": baseline_row["test_MAE_EUR_per_MW"],
        "best_test_MAE_EUR_per_MW": best_row["test_MAE_EUR_per_MW"],
        "best_test_MAPE_percent": best_row["test_MAPE_percent"],
        "best_test_R2_log_target": best_row["test_R2_log_target"],
        "best_cv_MAE_log_target": best_row["cv_MAE_log_target"],
        "beats_baseline": bool(best_row["test_MAE_EUR_per_MW"] < baseline_row["test_MAE_EUR_per_MW"]),
        "conclusion": "ML model outperforms the median baseline on the holdout split"
        if best_row["test_MAE_EUR_per_MW"] < baseline_row["test_MAE_EUR_per_MW"]
        else "ML model does not outperform the median baseline; use robust estimator as primary",
        "error_by_country": _group_errors(error_frame, "country"),
        "error_by_foundation": _group_errors(error_frame, "foundation_type"),
        "feature_importance": feature_importance,
    }

    if project_data:
        project_frame = pd.DataFrame([{feature: project_data.get(feature) for feature in available_features}])
        predicted_unit = float(np.exp(best_model.predict(project_frame)[0]))
        capacity = _safe_float(project_data.get("installed_capacity_MW"))
        report["prediction_for_new_project"] = {
            "budget_EUR_per_MW": round(predicted_unit, 0),
            "total_budget_EUR_2026": round(predicted_unit * capacity, 0) if capacity else None,
        }

    return report
