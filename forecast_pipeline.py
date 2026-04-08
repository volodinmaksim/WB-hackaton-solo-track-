from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


TARGET_COL = "target_1h"
TIME_COL = "timestamp"
ROUTE_COL = "route_id"
TEST_ID_COL = "id"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Honest time-series pipeline for WB hackathon.")
    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path(r"D:\ml\WB-hackathon\vscode\train_solo_track.parquet"),
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=Path(r"D:\ml\WB-hackathon\vscode\test_solo_track.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"D:\ml\WB-hackathon\vscode\outputs"),
    )
    parser.add_argument(
        "--val-horizon",
        type=int,
        default=8,
        help="How many latest global timestamps to hold out for validation.",
    )
    return parser.parse_args()


def make_time_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    ts = pd.to_datetime(result[TIME_COL])

    result["hour"] = ts.dt.hour.astype("int8")
    result["minute"] = ts.dt.minute.astype("int8")
    result["hour_slot"] = (ts.dt.hour * 2 + (ts.dt.minute // 30)).astype("int8")
    result["dayofweek"] = ts.dt.dayofweek.astype("int8")
    result["day"] = ts.dt.day.astype("int8")
    result["month"] = ts.dt.month.astype("int8")
    result["is_weekend"] = (ts.dt.dayofweek >= 5).astype("int8")
    result["is_month_start"] = ts.dt.is_month_start.astype("int8")
    result["is_month_end"] = ts.dt.is_month_end.astype("int8")
    return result


def build_target_aggregates(train_df: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], float]:
    global_mean = float(train_df[TARGET_COL].mean())

    agg_route = (
        train_df.groupby([ROUTE_COL], as_index=False)[TARGET_COL]
        .mean()
        .rename(columns={TARGET_COL: "target_mean_route"})
    )
    agg_route_hour = (
        train_df.groupby([ROUTE_COL, "hour_slot"], as_index=False)[TARGET_COL]
        .mean()
        .rename(columns={TARGET_COL: "target_mean_route_hour"})
    )
    agg_route_dow = (
        train_df.groupby([ROUTE_COL, "dayofweek"], as_index=False)[TARGET_COL]
        .mean()
        .rename(columns={TARGET_COL: "target_mean_route_dow"})
    )
    agg_route_hour_dow = (
        train_df.groupby([ROUTE_COL, "hour_slot", "dayofweek"], as_index=False)[TARGET_COL]
        .mean()
        .rename(columns={TARGET_COL: "target_mean_route_hour_dow"})
    )

    agg_global_hour = (
        train_df.groupby(["hour_slot"], as_index=False)[TARGET_COL]
        .mean()
        .rename(columns={TARGET_COL: "target_mean_global_hour"})
    )
    agg_global_dow = (
        train_df.groupby(["dayofweek"], as_index=False)[TARGET_COL]
        .mean()
        .rename(columns={TARGET_COL: "target_mean_global_dow"})
    )

    return {
        "route": agg_route,
        "route_hour": agg_route_hour,
        "route_dow": agg_route_dow,
        "route_hour_dow": agg_route_hour_dow,
        "global_hour": agg_global_hour,
        "global_dow": agg_global_dow,
    }, global_mean


def apply_target_aggregates(
    df: pd.DataFrame,
    aggregate_tables: dict[str, pd.DataFrame],
    global_mean: float,
) -> pd.DataFrame:
    result = df.copy()
    result = result.merge(aggregate_tables["route"], on=[ROUTE_COL], how="left")
    result = result.merge(aggregate_tables["route_hour"], on=[ROUTE_COL, "hour_slot"], how="left")
    result = result.merge(aggregate_tables["route_dow"], on=[ROUTE_COL, "dayofweek"], how="left")
    result = result.merge(
        aggregate_tables["route_hour_dow"],
        on=[ROUTE_COL, "hour_slot", "dayofweek"],
        how="left",
    )
    result = result.merge(aggregate_tables["global_hour"], on=["hour_slot"], how="left")
    result = result.merge(aggregate_tables["global_dow"], on=["dayofweek"], how="left")

    result["target_mean_route"] = result["target_mean_route"].fillna(global_mean)
    result["target_mean_global_hour"] = result["target_mean_global_hour"].fillna(global_mean)
    result["target_mean_global_dow"] = result["target_mean_global_dow"].fillna(global_mean)
    result["target_mean_route_hour"] = result["target_mean_route_hour"].fillna(result["target_mean_route"])
    result["target_mean_route_dow"] = result["target_mean_route_dow"].fillna(result["target_mean_route"])
    result["target_mean_route_hour_dow"] = result["target_mean_route_hour_dow"].fillna(
        result["target_mean_route_hour"]
    )

    return result


def make_validation_split(train_df: pd.DataFrame, val_horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_timestamps = np.sort(train_df[TIME_COL].unique())
    if val_horizon >= len(unique_timestamps):
        raise ValueError("Validation horizon is too large for available timestamps.")

    val_timestamps = unique_timestamps[-val_horizon:]
    fit_mask = ~train_df[TIME_COL].isin(val_timestamps)
    val_mask = train_df[TIME_COL].isin(val_timestamps)

    fit_df = train_df.loc[fit_mask].copy()
    val_df = train_df.loc[val_mask].copy()
    return fit_df, val_df


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.abs(y_true).sum()
    if denominator == 0:
        return 0.0
    return float(np.abs(y_true - y_pred).sum() / denominator)


def relative_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.abs(y_true).sum()
    if denominator == 0:
        return 0.0
    return float(np.abs(y_pred.sum() - y_true.sum()) / denominator)


def competition_metric(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    metric_wape = wape(y_true, y_pred)
    metric_bias = relative_bias(y_true, y_pred)
    return {
        "wape": metric_wape,
        "relative_bias": metric_bias,
        "score": metric_wape + metric_bias,
    }


def make_model() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="l1",
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )


def fit_and_predict(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[lgb.LGBMRegressor, np.ndarray]:
    model = make_model()

    X_train = train_df[feature_cols].copy()
    X_valid = valid_df[feature_cols].copy()
    y_train = train_df[TARGET_COL].to_numpy()
    y_valid = valid_df[TARGET_COL].to_numpy()

    for frame in (X_train, X_valid):
        frame[ROUTE_COL] = frame[ROUTE_COL].astype("category")

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="l1",
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=100),
        ],
        categorical_feature=[ROUTE_COL],
    )
    predictions = model.predict(X_valid, num_iteration=model.best_iteration_)
    return model, predictions


def train_full_and_predict_test(
    full_train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[lgb.LGBMRegressor, np.ndarray]:
    model = make_model()
    X_train = full_train_df[feature_cols].copy()
    y_train = full_train_df[TARGET_COL].to_numpy()
    X_test = test_df[feature_cols].copy()

    X_train[ROUTE_COL] = X_train[ROUTE_COL].astype("category")
    X_test[ROUTE_COL] = pd.Categorical(X_test[ROUTE_COL], categories=X_train[ROUTE_COL].cat.categories)

    model.fit(X_train, y_train, categorical_feature=[ROUTE_COL])
    predictions = model.predict(X_test, num_iteration=model.best_iteration_)
    return model, predictions


def save_feature_importance(
    model: lgb.LGBMRegressor,
    feature_cols: list[str],
    output_path: Path,
) -> None:
    importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance_gain": model.booster_.feature_importance(importance_type="gain"),
            "importance_split": model.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values("importance_gain", ascending=False)
    importance.to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(args.train_path)
    test = pd.read_parquet(args.test_path)

    train = make_time_features(train)
    test = make_time_features(test)

    fit_df, val_df = make_validation_split(train, val_horizon=args.val_horizon)

    aggregate_tables, global_mean = build_target_aggregates(fit_df)
    fit_with_aggs = apply_target_aggregates(fit_df, aggregate_tables, global_mean)
    val_with_aggs = apply_target_aggregates(val_df, aggregate_tables, global_mean)

    base_features = [
        ROUTE_COL,
        "hour",
        "minute",
        "hour_slot",
        "dayofweek",
        "day",
        "month",
        "is_weekend",
        "is_month_start",
        "is_month_end",
    ]
    agg_features = [
        "target_mean_route",
        "target_mean_route_hour",
        "target_mean_route_dow",
        "target_mean_route_hour_dow",
        "target_mean_global_hour",
        "target_mean_global_dow",
    ]

    baseline_model, baseline_pred = fit_and_predict(fit_df, val_df, feature_cols=base_features)
    agg_model, agg_pred = fit_and_predict(fit_with_aggs, val_with_aggs, feature_cols=base_features + agg_features)

    y_val = val_df[TARGET_COL].to_numpy()
    metrics = {
        "baseline": competition_metric(y_val, baseline_pred),
        "with_aggregates": competition_metric(y_val, agg_pred),
        "validation": {
            "train_rows": int(len(fit_df)),
            "valid_rows": int(len(val_df)),
            "valid_timestamps": [str(ts) for ts in sorted(val_df[TIME_COL].unique())],
        },
    }

    full_aggregate_tables, full_global_mean = build_target_aggregates(train)
    train_full = apply_target_aggregates(train, full_aggregate_tables, full_global_mean)
    test_full = apply_target_aggregates(test, full_aggregate_tables, full_global_mean)

    final_model, test_pred = train_full_and_predict_test(
        train_full,
        test_full,
        feature_cols=base_features + agg_features,
    )

    submission = test[[TEST_ID_COL]].copy()
    submission[TARGET_COL] = np.clip(test_pred, a_min=0, a_max=None)

    metrics_path = args.output_dir / "metrics.json"
    submission_path = args.output_dir / "submission.csv"
    importance_path = args.output_dir / "feature_importance.csv"

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    submission.to_csv(submission_path, index=False)
    save_feature_importance(final_model, base_features + agg_features, importance_path)

    print(json.dumps(metrics, indent=2))
    print(f"Saved submission to: {submission_path}")
    print(f"Saved feature importance to: {importance_path}")


def run_validation_experiment(
    train_path: str | Path = r"D:\ml\WB-hackathon\vscode\train_solo_track.parquet",
    val_horizon: int = 8,
) -> dict[str, float]:
    train = pd.read_parquet(train_path)
    train = make_time_features(train)

    fit_df, val_df = make_validation_split(train, val_horizon=val_horizon)
    aggregate_tables, global_mean = build_target_aggregates(fit_df)

    fit_with_aggs = apply_target_aggregates(fit_df, aggregate_tables, global_mean)
    val_with_aggs = apply_target_aggregates(val_df, aggregate_tables, global_mean)

    base_features = [
        ROUTE_COL,
        "hour",
        "minute",
        "hour_slot",
        "dayofweek",
        "day",
        "month",
        "is_weekend",
        "is_month_start",
        "is_month_end",
    ]
    agg_features = [
        "target_mean_route",
        "target_mean_route_hour",
        "target_mean_route_dow",
        "target_mean_route_hour_dow",
        "target_mean_global_hour",
        "target_mean_global_dow",
    ]

    _, baseline_pred = fit_and_predict(fit_df, val_df, feature_cols=base_features)
    _, agg_pred = fit_and_predict(fit_with_aggs, val_with_aggs, feature_cols=base_features + agg_features)
    y_val = val_df[TARGET_COL].to_numpy()

    baseline_metrics = competition_metric(y_val, baseline_pred)
    agg_metrics = competition_metric(y_val, agg_pred)

    return {
        "baseline_score": baseline_metrics["score"],
        "agg_score": agg_metrics["score"],
        "baseline_wape": baseline_metrics["wape"],
        "agg_wape": agg_metrics["wape"],
        "baseline_relative_bias": baseline_metrics["relative_bias"],
        "agg_relative_bias": agg_metrics["relative_bias"],
    }


if __name__ == "__main__":
    main()
