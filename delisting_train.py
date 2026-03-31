from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from delisting_data import PreparedTrainingData
from delisting_shared import (
    CODE_COLUMN,
    COMPANY_COLUMN,
    CORP_CODE_COLUMN,
    DEFAULT_THRESHOLDS,
    EVENT_YEAR_COLUMN,
    FEATURE_COLUMNS,
    META_OUTPUT_COLUMNS,
    PREDICTION_ELIGIBLE_COLUMN,
    PREDICTION_LABEL_COLUMN,
    PREDICTION_LABELS,
    PREDICTION_MODEL_COLUMN,
    PREDICTION_PROBABILITY_COLUMN,
    PREDICTION_SKIP_REASON_COLUMN,
    TARGET_COLUMN,
    YEAR_COLUMN,
    write_json,
)


def build_logistic_model(seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=seed,
                ),
            ),
        ]
    )


def safe_roc_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, probabilities))


def safe_pr_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    if int((y_true == 1).sum()) == 0:
        return float("nan")
    return float(average_precision_score(y_true, probabilities))


def compute_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": safe_roc_auc(y_true, probabilities),
        "pr_auc": safe_pr_auc(y_true, probabilities),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "positive_support": int((y_true == 1).sum()),
        "negative_support": int((y_true == 0).sum()),
    }


def build_threshold_comparison(valid_frame: pd.DataFrame, probabilities: np.ndarray, thresholds: list[float]) -> pd.DataFrame:
    rows = []
    y_true = valid_frame[TARGET_COLUMN].to_numpy()
    for threshold in thresholds or DEFAULT_THRESHOLDS:
        row = compute_metrics(y_true, probabilities, threshold)
        row["split"] = "valid"
        rows.append(row)
    return pd.DataFrame(rows).sort_values("threshold", kind="mergesort").reset_index(drop=True)


def select_threshold(threshold_frame: pd.DataFrame) -> float:
    ordered = threshold_frame.sort_values(
        by=["recall", "f1", "precision", "threshold"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return float(ordered.iloc[0]["threshold"])


def build_prediction_frame(frame: pd.DataFrame, probabilities: np.ndarray, threshold: float, split_name: str) -> pd.DataFrame:
    output = frame[[COMPANY_COLUMN, CODE_COLUMN, CORP_CODE_COLUMN, YEAR_COLUMN, EVENT_YEAR_COLUMN, TARGET_COLUMN]].copy()
    output["split"] = split_name
    output[PREDICTION_PROBABILITY_COLUMN] = probabilities
    predictions = (probabilities >= threshold).astype(int)
    output[PREDICTION_LABEL_COLUMN] = [PREDICTION_LABELS[int(value)] for value in predictions]
    output["predicted_target"] = predictions
    return output


def write_markdown_report(
    path: Path,
    training_data: PreparedTrainingData,
    threshold_frame: pd.DataFrame,
    split_metrics: pd.DataFrame,
    selected_threshold: float,
) -> None:
    lines = [
        "# 상장폐지 미래예측 1차 베이스라인 보고서",
        "",
        "## 데이터 요약",
        f'- 원본 행 수: {training_data.stats["raw_rows"]}',
        f'- 학습 가능 행 수: {training_data.stats["usable_rows"]}',
        f'- 양성(Y-2 미래상폐): {training_data.stats["positive_rows"]}',
        f'- 음성(기타 정상 연도): {training_data.stats["negative_rows"]}',
        f'- Y-1 제외 행 수: {training_data.stats["excluded_y_minus_1_rows"]}',
        f'- 학습 구간: {training_data.train_frame[YEAR_COLUMN].min()}~{training_data.train_frame[YEAR_COLUMN].max()}',
        f'- 검증 구간: {training_data.valid_frame[YEAR_COLUMN].min()}',
        f'- 테스트 구간: {training_data.test_frame[YEAR_COLUMN].min()}',
        "",
        "## 선택된 threshold",
        "- validation 기준 recall -> f1 -> precision 순으로 선택",
        f"- selected threshold: **{selected_threshold:.2f}**",
        "",
    ]

    table_columns = [
        "split",
        "threshold_type",
        "threshold",
        "recall",
        "precision",
        "f1",
        "roc_auc",
        "pr_auc",
        "tp",
        "fp",
        "fn",
        "tn",
    ]
    header = "| " + " | ".join(table_columns) + " |"
    separator = "| " + " | ".join(["---"] * len(table_columns)) + " |"
    lines.extend(["## split metrics", header, separator])
    for _, row in split_metrics.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["split"]),
                    str(row["threshold_type"]),
                    f'{row["threshold"]:.2f}',
                    f'{row["recall"]:.4f}',
                    f'{row["precision"]:.4f}',
                    f'{row["f1"]:.4f}',
                    "NaN" if pd.isna(row["roc_auc"]) else f'{row["roc_auc"]:.4f}',
                    "NaN" if pd.isna(row["pr_auc"]) else f'{row["pr_auc"]:.4f}',
                    str(int(row["tp"])),
                    str(int(row["fp"])),
                    str(int(row["fn"])),
                    str(int(row["tn"])),
                ]
            )
            + " |"
        )
    lines.extend(["", "## validation threshold sweep", header, separator])
    for _, row in threshold_frame.iterrows():
        roc_auc_text = "NaN" if pd.isna(row["roc_auc"]) else f'{row["roc_auc"]:.4f}'
        pr_auc_text = "NaN" if pd.isna(row["pr_auc"]) else f'{row["pr_auc"]:.4f}'
        lines.append(
            "| valid | candidate | "
            f'{row["threshold"]:.2f} | {row["recall"]:.4f} | {row["precision"]:.4f} | {row["f1"]:.4f} | '
            f"{roc_auc_text} | "
            f"{pr_auc_text} | "
            f'{int(row["tp"])} | {int(row["fp"])} | {int(row["fn"])} | {int(row["tn"])} |'
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fit_and_save_logistic(
    output_dir: Path,
    training_data: PreparedTrainingData,
    split_metrics: pd.DataFrame,
    threshold_frame: pd.DataFrame,
    selected_threshold: float,
    seed: int,
    split_config: dict[str, int],
) -> None:
    final_fit_frame = pd.concat([training_data.train_frame, training_data.valid_frame], ignore_index=True)
    final_model = build_logistic_model(seed)
    final_model.fit(final_fit_frame[FEATURE_COLUMNS], final_fit_frame[TARGET_COLUMN].astype(int))

    bundle = {
        "model_name": "logistic",
        "estimator": final_model,
        "feature_columns": FEATURE_COLUMNS,
        "prediction_labels": PREDICTION_LABELS,
        "threshold": selected_threshold,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "split_config": split_config,
        "threshold_selection": "validation recall -> f1 -> precision",
        "training_stats": training_data.stats,
        "metrics": split_metrics.to_dict(orient="records"),
    }
    joblib.dump(bundle, output_dir / "model_logistic.joblib")
    joblib.dump(bundle, output_dir / "best_model.joblib")

    metrics_summary = pd.DataFrame(
        [
            {
                "model": "logistic",
                "selected_threshold": selected_threshold,
                "valid_recall": float(
                    split_metrics.loc[
                        (split_metrics["split"] == "valid") & (split_metrics["threshold_type"] == "selected"),
                        "recall",
                    ].iloc[0]
                ),
                "valid_precision": float(
                    split_metrics.loc[
                        (split_metrics["split"] == "valid") & (split_metrics["threshold_type"] == "selected"),
                        "precision",
                    ].iloc[0]
                ),
                "valid_f1": float(
                    split_metrics.loc[
                        (split_metrics["split"] == "valid") & (split_metrics["threshold_type"] == "selected"),
                        "f1",
                    ].iloc[0]
                ),
                "test_recall": float(
                    split_metrics.loc[
                        (split_metrics["split"] == "test") & (split_metrics["threshold_type"] == "selected"),
                        "recall",
                    ].iloc[0]
                ),
                "test_precision": float(
                    split_metrics.loc[
                        (split_metrics["split"] == "test") & (split_metrics["threshold_type"] == "selected"),
                        "precision",
                    ].iloc[0]
                ),
                "test_f1": float(
                    split_metrics.loc[
                        (split_metrics["split"] == "test") & (split_metrics["threshold_type"] == "selected"),
                        "f1",
                    ].iloc[0]
                ),
            }
        ]
    )
    metrics_summary.to_csv(output_dir / "metrics_summary.csv", index=False, encoding="utf-8-sig")
    threshold_frame.to_csv(output_dir / "threshold_comparison.csv", index=False, encoding="utf-8-sig")
    split_metrics.to_csv(output_dir / "split_metrics.csv", index=False, encoding="utf-8-sig")
    write_json(output_dir / "feature_columns.json", FEATURE_COLUMNS)
    write_json(output_dir / "training_stats.json", training_data.stats)
    write_markdown_report(output_dir / "comparison_report.md", training_data, threshold_frame, split_metrics, selected_threshold)


def merge_prediction_columns(base_frame: pd.DataFrame, prediction_frame: pd.DataFrame) -> pd.DataFrame:
    merged = base_frame.merge(
        prediction_frame,
        on=[CODE_COLUMN, YEAR_COLUMN],
        how="left",
        suffixes=("", "__pred"),
    )
    for column in [
        PREDICTION_PROBABILITY_COLUMN,
        PREDICTION_LABEL_COLUMN,
        PREDICTION_MODEL_COLUMN,
        PREDICTION_ELIGIBLE_COLUMN,
        PREDICTION_SKIP_REASON_COLUMN,
    ]:
        pred_column = f"{column}__pred"
        if pred_column not in merged.columns:
            continue
        if column == PREDICTION_ELIGIBLE_COLUMN:
            base_bool = merged[column].astype("boolean")
            pred_bool = merged[pred_column].astype("boolean")
            merged[column] = pred_bool.where(pred_bool.notna(), base_bool).to_numpy(dtype=bool, na_value=False)
        else:
            merged[column] = merged[pred_column].combine_first(merged[column])
        merged = merged.drop(columns=[pred_column])

    output_columns = META_OUTPUT_COLUMNS + [
        PREDICTION_PROBABILITY_COLUMN,
        PREDICTION_LABEL_COLUMN,
        PREDICTION_MODEL_COLUMN,
        PREDICTION_ELIGIBLE_COLUMN,
        PREDICTION_SKIP_REASON_COLUMN,
    ]
    extra_columns = [column for column in merged.columns if column not in output_columns]
    return merged[output_columns + extra_columns]
