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
    parser = argparse.ArgumentParser(description="Финальное воспроизводимое решение для WB hackathon solo track.")
    parser.add_argument("--train-path", type=Path, default=Path("data/train_solo_track.parquet"))
    parser.add_argument("--test-path", type=Path, default=Path("data/test_solo_track.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/final_solution"))
    parser.add_argument("--val-horizon", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=20.0)
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


def make_validation_split(train_df: pd.DataFrame, val_horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_timestamps = np.sort(train_df[TIME_COL].unique())
    if val_horizon >= len(unique_timestamps):
        raise ValueError("Validation horizon is too large for available timestamps.")
    val_timestamps = unique_timestamps[-val_horizon:]
    fit_df = train_df.loc[~train_df[TIME_COL].isin(val_timestamps)].copy()
    val_df = train_df.loc[train_df[TIME_COL].isin(val_timestamps)].copy()
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


def build_smoothing_tables(train_df: pd.DataFrame, alpha: float) -> tuple[dict[str, pd.DataFrame], float]:
    global_mean = float(train_df[TARGET_COL].mean())

    route = train_df.groupby([ROUTE_COL])[TARGET_COL].agg(["mean", "size"]).reset_index()
    route = route.rename(columns={"mean": "route_mean", "size": "route_count"})
    route["target_smooth_route"] = (route["route_mean"] * route["route_count"] + global_mean * alpha) / (
        route["route_count"] + alpha
    )

    route_hour = train_df.groupby([ROUTE_COL, "hour_slot"])[TARGET_COL].agg(["mean", "size"]).reset_index()
    route_hour = route_hour.rename(columns={"mean": "route_hour_mean", "size": "route_hour_count"})
    route_hour["target_smooth_route_hour"] = (
        route_hour["route_hour_mean"] * route_hour["route_hour_count"] + global_mean * alpha
    ) / (route_hour["route_hour_count"] + alpha)

    route_dow = train_df.groupby([ROUTE_COL, "dayofweek"])[TARGET_COL].agg(["mean", "size"]).reset_index()
    route_dow = route_dow.rename(columns={"mean": "route_dow_mean", "size": "route_dow_count"})
    route_dow["target_smooth_route_dow"] = (
        route_dow["route_dow_mean"] * route_dow["route_dow_count"] + global_mean * alpha
    ) / (route_dow["route_dow_count"] + alpha)

    route_hour_dow = train_df.groupby([ROUTE_COL, "hour_slot", "dayofweek"])[TARGET_COL].agg(["mean", "size"]).reset_index()
    route_hour_dow = route_hour_dow.rename(columns={"mean": "route_hour_dow_mean", "size": "route_hour_dow_count"})
    route_hour_dow["target_smooth_route_hour_dow"] = (
        route_hour_dow["route_hour_dow_mean"] * route_hour_dow["route_hour_dow_count"] + global_mean * alpha
    ) / (route_hour_dow["route_hour_dow_count"] + alpha)

    global_hour = train_df.groupby(["hour_slot"])[TARGET_COL].agg(["mean", "size"]).reset_index()
    global_hour = global_hour.rename(columns={"mean": "global_hour_mean", "size": "global_hour_count"})
    global_hour["target_smooth_global_hour"] = (
        global_hour["global_hour_mean"] * global_hour["global_hour_count"] + global_mean * alpha
    ) / (global_hour["global_hour_count"] + alpha)

    global_dow = train_df.groupby(["dayofweek"])[TARGET_COL].agg(["mean", "size"]).reset_index()
    global_dow = global_dow.rename(columns={"mean": "global_dow_mean", "size": "global_dow_count"})
    global_dow["target_smooth_global_dow"] = (
        global_dow["global_dow_mean"] * global_dow["global_dow_count"] + global_mean * alpha
    ) / (global_dow["global_dow_count"] + alpha)

    return {
        "route": route[[ROUTE_COL, "target_smooth_route"]],
        "route_hour": route_hour[[ROUTE_COL, "hour_slot", "target_smooth_route_hour"]],
        "route_dow": route_dow[[ROUTE_COL, "dayofweek", "target_smooth_route_dow"]],
        "route_hour_dow": route_hour_dow[[ROUTE_COL, "hour_slot", "dayofweek", "target_smooth_route_hour_dow"]],
        "global_hour": global_hour[["hour_slot", "target_smooth_global_hour"]],
        "global_dow": global_dow[["dayofweek", "target_smooth_global_dow"]],
    }, global_mean


def apply_smoothing_tables(df: pd.DataFrame, tables: dict[str, pd.DataFrame], global_mean: float) -> pd.DataFrame:
    result = df.copy()
    result = result.merge(tables["route"], on=[ROUTE_COL], how="left")
    result = result.merge(tables["route_hour"], on=[ROUTE_COL, "hour_slot"], how="left")
    result = result.merge(tables["route_dow"], on=[ROUTE_COL, "dayofweek"], how="left")
    result = result.merge(tables["route_hour_dow"], on=[ROUTE_COL, "hour_slot", "dayofweek"], how="left")
    result = result.merge(tables["global_hour"], on=["hour_slot"], how="left")
    result = result.merge(tables["global_dow"], on=["dayofweek"], how="left")

    result["target_smooth_route"] = result["target_smooth_route"].fillna(global_mean)
    result["target_smooth_route_hour"] = result["target_smooth_route_hour"].fillna(result["target_smooth_route"])
    result["target_smooth_route_dow"] = result["target_smooth_route_dow"].fillna(result["target_smooth_route"])
    result["target_smooth_route_hour_dow"] = result["target_smooth_route_hour_dow"].fillna(
        result["target_smooth_route_hour"]
    )
    result["target_smooth_global_hour"] = result["target_smooth_global_hour"].fillna(global_mean)
    result["target_smooth_global_dow"] = result["target_smooth_global_dow"].fillna(global_mean)

    for col in [
        "target_smooth_route",
        "target_smooth_route_hour",
        "target_smooth_route_dow",
        "target_smooth_route_hour_dow",
        "target_smooth_global_hour",
        "target_smooth_global_dow",
    ]:
        result[col] = result[col].astype("float32")

    return result


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


def fit_and_predict(train_df: pd.DataFrame, valid_df: pd.DataFrame, feature_cols: list[str]) -> tuple[lgb.LGBMRegressor, np.ndarray]:
    model = make_model()
    x_train = train_df[feature_cols].copy()
    x_valid = valid_df[feature_cols].copy()
    y_train = train_df[TARGET_COL].to_numpy()
    y_valid = valid_df[TARGET_COL].to_numpy()
    for frame in (x_train, x_valid):
        frame[ROUTE_COL] = frame[ROUTE_COL].astype("category")
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False), lgb.log_evaluation(period=100)],
        categorical_feature=[ROUTE_COL],
    )
    return model, model.predict(x_valid, num_iteration=model.best_iteration_)


def train_full_and_predict_test(full_train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]) -> tuple[lgb.LGBMRegressor, np.ndarray]:
    model = make_model()
    x_train = full_train_df[feature_cols].copy()
    y_train = full_train_df[TARGET_COL].to_numpy()
    x_test = test_df[feature_cols].copy()
    x_train[ROUTE_COL] = x_train[ROUTE_COL].astype("category")
    x_test[ROUTE_COL] = pd.Categorical(x_test[ROUTE_COL], categories=x_train[ROUTE_COL].cat.categories)
    model.fit(x_train, y_train, categorical_feature=[ROUTE_COL])
    return model, model.predict(x_test, num_iteration=model.best_iteration_)


def save_feature_importance(model: lgb.LGBMRegressor, feature_cols: list[str], output_path: Path) -> None:
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

    train = make_time_features(pd.read_parquet(args.train_path))
    test = make_time_features(pd.read_parquet(args.test_path))

    fit_df, val_df = make_validation_split(train, val_horizon=args.val_horizon)
    tables, global_mean = build_smoothing_tables(fit_df, alpha=args.alpha)
    fit_features = apply_smoothing_tables(fit_df, tables, global_mean)
    val_features = apply_smoothing_tables(val_df, tables, global_mean)

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
    smooth_features = [
        "target_smooth_route",
        "target_smooth_route_hour",
        "target_smooth_route_dow",
        "target_smooth_route_hour_dow",
        "target_smooth_global_hour",
        "target_smooth_global_dow",
    ]

    _, valid_pred = fit_and_predict(fit_features, val_features, base_features + smooth_features)
    y_val = val_df[TARGET_COL].to_numpy()
    metrics = competition_metric(y_val, valid_pred)

    full_tables, full_global_mean = build_smoothing_tables(train, alpha=args.alpha)
    train_full = apply_smoothing_tables(train, full_tables, full_global_mean)
    test_full = apply_smoothing_tables(test, full_tables, full_global_mean)
    final_model, pred_test = train_full_and_predict_test(train_full, test_full, base_features + smooth_features)
    pred_test = np.clip(pred_test, a_min=0, a_max=None)

    submission = pd.DataFrame({TEST_ID_COL: test[TEST_ID_COL], "y_pred": pred_test})
    metrics_payload = {
        "model_name": "lightgbm_smoothing_alpha20",
        "alpha": args.alpha,
        "val_horizon": args.val_horizon,
        "local_validation": metrics,
        "known_public_score": 0.3608341250041473,
        "best_submission_file_from_hackathon": "submission_route_time_agg_smooth_v1.csv",
    }

    submission.to_csv(args.output_dir / "submission.csv", index=False)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    save_feature_importance(final_model, base_features + smooth_features, args.output_dir / "feature_importance.csv")

    print(json.dumps(metrics_payload, indent=2, ensure_ascii=False))
    print(f"Submission saved to: {args.output_dir / 'submission.csv'}")


if __name__ == "__main__":
    main()
