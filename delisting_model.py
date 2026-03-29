from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


WORKDIR = Path(r"C:\kwoss_C\model_test")
DEFAULT_TRAIN_INPUT = WORKDIR / "downloads" / "dart_financials" / "financial_ratios_2015_2025.csv"
DEFAULT_PREDICT_INPUT = WORKDIR / "downloads" / "dart_financials" / "financial_ratios_2024_test.csv"
DEFAULT_OUTPUT_DIR = WORKDIR / "artifacts" / "delisting"

STATUS_COLUMN = "기업상태"
COMPANY_COLUMN = "기업명"
CODE_COLUMN = "종목코드"
CORP_CODE_COLUMN = "corp_code"
YEAR_COLUMN = "연도"
FS_COLUMN = "selected_fs_div"
HAS_DATA_COLUMN = "has_data"
SOURCE_FILE_COLUMN = "source_file"

LABEL_MAPPING = {
    "상장기업": 0,
    "상폐기업": 1,
}
LABEL_NAME_MAPPING = {value: key for key, value in LABEL_MAPPING.items()}

FEATURE_COLUMNS = [
    "총자산증가율",
    "유동자산증가율",
    "매출액증가율",
    "순이익증가율",
    "영업이익증가율",
    "매출액순이익률",
    "매출총이익률",
    "자기자본순이익률",
    "매출채권회전율",
    "재고자산회전율",
    "총자본회전율",
    "유형자산회전율",
    "매출원가율",
    "부채비율",
    "유동비율",
    "자기자본비율",
    "당좌비율",
    "비유동자산장기적합률",
    "순운전자본비율",
    "차입금의존도",
    "현금비율",
    "유형자산",
    "무형자산",
    "무형자산상각비",
    "유형자산상각비",
    "감가상각비",
    "총자본영업이익률",
    "총자본순이익률",
    "유보액/납입자본비율",
    "총자본투자효율",
]

DIAGNOSTIC_COLUMNS = [
    "총자산",
    "유동자산",
    "매출액",
    "순이익",
    "영업이익",
    "유동부채",
    "자기자본",
]

META_OUTPUT_COLUMNS = [COMPANY_COLUMN, CODE_COLUMN, YEAR_COLUMN]
ALL_REQUIRED_COLUMNS = [
    STATUS_COLUMN,
    COMPANY_COLUMN,
    CODE_COLUMN,
    CORP_CODE_COLUMN,
    YEAR_COLUMN,
    FS_COLUMN,
    HAS_DATA_COLUMN,
    SOURCE_FILE_COLUMN,
    *FEATURE_COLUMNS,
    *DIAGNOSTIC_COLUMNS,
]


@dataclass
class PreparedTrainingData:
    features: pd.DataFrame
    labels: pd.Series
    groups: pd.Series
    frame: pd.DataFrame
    stats: dict[str, Any]


@dataclass
class PreparedPredictionData:
    working_frame: pd.DataFrame
    output_frame: pd.DataFrame
    stats: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="상장폐지 분류 모델 학습 및 예측 CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="3개 모델을 학습하고 성능을 비교합니다.")
    train_parser.add_argument("--input", type=Path, default=DEFAULT_TRAIN_INPUT, help="학습용 CSV 경로")
    train_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="산출물 저장 디렉터리")
    train_parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    train_parser.add_argument("--cv-folds", type=int, default=5, help="StratifiedGroupKFold 분할 수")

    predict_parser = subparsers.add_parser("predict", help="저장된 모델로 예측합니다.")
    predict_parser.add_argument("--model-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="모델 산출물 디렉터리")
    predict_parser.add_argument("--input", type=Path, default=DEFAULT_PREDICT_INPUT, help="예측용 CSV 경로")
    predict_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR / "predictions_2024.csv", help="예측 결과 CSV")
    predict_parser.add_argument(
        "--model",
        default="best",
        choices=["best", "logistic", "random_forest", "xgboost"],
        help="예측에 사용할 모델",
    )

    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {path}")
    return pd.read_csv(path)


def ensure_columns(df: pd.DataFrame) -> None:
    missing = [column for column in ALL_REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {missing}")


def deduplicate_company_year(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    if "__label" not in working.columns:
        working["__label"] = working[STATUS_COLUMN].map(LABEL_MAPPING)
    working["__label_priority"] = working["__label"].fillna(-1)
    working = working.sort_values(
        by=[CODE_COLUMN, YEAR_COLUMN, "__label_priority"],
        ascending=[True, True, False],
        kind="mergesort",
    )
    working = working.drop_duplicates(subset=[CODE_COLUMN, YEAR_COLUMN], keep="first")
    return working.drop(columns=["__label_priority"])


def prepare_training_data(df: pd.DataFrame) -> PreparedTrainingData:
    ensure_columns(df)
    stats: dict[str, Any] = {"raw_rows": int(len(df))}

    working = df.copy()
    working = working[working[HAS_DATA_COLUMN].fillna(False).astype(bool)].copy()
    stats["after_has_data"] = int(len(working))

    working["__label"] = working[STATUS_COLUMN].map(LABEL_MAPPING)
    unmapped = int(working["__label"].isna().sum())
    if unmapped:
        raise ValueError(f"알 수 없는 기업상태가 {unmapped}건 있습니다.")

    working = working[~working[FEATURE_COLUMNS].isna().all(axis=1)].copy()
    stats["after_all_null_feature_drop"] = int(len(working))

    before_dedup = len(working)
    working = deduplicate_company_year(working)
    stats["after_dedup"] = int(len(working))
    stats["dedup_removed"] = int(before_dedup - len(working))

    labels = working["__label"].astype(int)
    groups = working[CODE_COLUMN].astype(str)
    features = working[FEATURE_COLUMNS].copy()

    stats["positive_rows"] = int((labels == 1).sum())
    stats["negative_rows"] = int((labels == 0).sum())
    stats["company_count"] = int(groups.nunique())
    stats["year_min"] = int(working[YEAR_COLUMN].min())
    stats["year_max"] = int(working[YEAR_COLUMN].max())

    return PreparedTrainingData(
        features=features,
        labels=labels,
        groups=groups,
        frame=working,
        stats=stats,
    )


def prepare_prediction_data(df: pd.DataFrame) -> PreparedPredictionData:
    ensure_columns(df)
    stats: dict[str, Any] = {"input_rows": int(len(df))}

    output = df.copy()
    output["상폐확률"] = np.nan
    output["예측라벨"] = pd.Series([pd.NA] * len(output), dtype="string")
    output["사용모델"] = pd.Series([pd.NA] * len(output), dtype="string")
    output["예측가능여부"] = False
    output["스킵사유"] = pd.Series([""] * len(output), dtype="string")

    eligible_mask = df[HAS_DATA_COLUMN].fillna(False).astype(bool)
    output.loc[~eligible_mask, "스킵사유"] = "has_data=False"

    all_feature_null_mask = df[FEATURE_COLUMNS].isna().all(axis=1)
    newly_ineligible = eligible_mask & all_feature_null_mask
    output.loc[newly_ineligible, "스킵사유"] = "30개 재무비율 모두 결측"

    eligible_mask = eligible_mask & ~all_feature_null_mask
    eligible = df.loc[eligible_mask].copy()
    eligible["__row_key"] = eligible[CODE_COLUMN].astype(str) + "::" + eligible[YEAR_COLUMN].astype(str)

    before_dedup = len(eligible)
    eligible["__label"] = eligible[STATUS_COLUMN].map(LABEL_MAPPING)
    eligible = deduplicate_company_year(eligible)
    eligible["__row_key"] = eligible[CODE_COLUMN].astype(str) + "::" + eligible[YEAR_COLUMN].astype(str)

    stats["eligible_rows_before_dedup"] = int(before_dedup)
    stats["eligible_rows_after_dedup"] = int(len(eligible))
    stats["dedup_removed"] = int(before_dedup - len(eligible))
    stats["skipped_rows"] = int((~eligible_mask).sum())

    return PreparedPredictionData(
        working_frame=eligible,
        output_frame=output,
        stats=stats,
    )


def build_models(seed: int, positive_count: int, negative_count: int) -> dict[str, Any]:
    scale_pos_weight = max(negative_count / max(positive_count, 1), 1.0)

    logistic = Pipeline(
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

    random_forest = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=400,
                    class_weight="balanced_subsample",
                    min_samples_leaf=2,
                    n_jobs=-1,
                    random_state=seed,
                ),
            ),
        ]
    )

    xgboost = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                XGBClassifier(
                    n_estimators=400,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    random_state=seed,
                    n_jobs=-1,
                    scale_pos_weight=scale_pos_weight,
                    eval_metric="logloss",
                    tree_method="hist",
                ),
            ),
        ]
    )

    return {
        "logistic": logistic,
        "random_forest": random_forest,
        "xgboost": xgboost,
    }


def compute_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    predictions = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    metrics = {
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "positive_support": int((y_true == 1).sum()),
        "negative_support": int((y_true == 0).sum()),
    }
    return metrics


def evaluate_model(
    model_name: str,
    estimator: Any,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    cv_folds: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cv = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    oof_probabilities = np.full(len(X), np.nan, dtype=float)
    fold_summaries: list[dict[str, Any]] = []

    for fold_index, (train_idx, valid_idx) in enumerate(cv.split(X, y, groups), start=1):
        fold_estimator = clone(estimator)
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        y_valid = y.iloc[valid_idx]

        fold_estimator.fit(X_train, y_train)
        fold_probabilities = fold_estimator.predict_proba(X_valid)[:, 1]
        oof_probabilities[valid_idx] = fold_probabilities

        fold_metrics = compute_metrics(y_valid.to_numpy(), fold_probabilities)
        fold_metrics["model"] = model_name
        fold_metrics["fold"] = fold_index
        fold_metrics["train_rows"] = int(len(train_idx))
        fold_metrics["valid_rows"] = int(len(valid_idx))
        fold_summaries.append(fold_metrics)

    if np.isnan(oof_probabilities).any():
        raise RuntimeError(f"{model_name} 교차검증 예측에 누락이 있습니다.")

    overall_metrics = compute_metrics(y.to_numpy(), oof_probabilities)
    overall_metrics["model"] = model_name
    overall_metrics["fold"] = "overall"
    overall_metrics["train_rows"] = int(len(X))
    overall_metrics["valid_rows"] = int(len(X))
    return overall_metrics, fold_summaries


def select_best_model(metrics_frame: pd.DataFrame) -> str:
    sorted_frame = metrics_frame.sort_values(
        by=["recall", "f1", "pr_auc"],
        ascending=[False, False, False],
        kind="mergesort",
    )
    return str(sorted_frame.iloc[0]["model"])


def fit_and_save_models(
    models: dict[str, Any],
    training_data: PreparedTrainingData,
    metrics_frame: pd.DataFrame,
    output_dir: Path,
) -> str:
    best_model_name = select_best_model(metrics_frame)

    for model_name, estimator in models.items():
        fitted = clone(estimator)
        fitted.fit(training_data.features, training_data.labels)

        bundle = {
            "model_name": model_name,
            "estimator": fitted,
            "feature_columns": FEATURE_COLUMNS,
            "label_mapping": LABEL_MAPPING,
            "label_name_mapping": LABEL_NAME_MAPPING,
            "threshold": 0.5,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "training_stats": training_data.stats,
            "metrics": metrics_frame.set_index("model").loc[model_name].to_dict(),
        }

        model_path = output_dir / f"model_{model_name}.joblib"
        joblib.dump(bundle, model_path)

        if model_name == best_model_name:
            joblib.dump(bundle, output_dir / "best_model.joblib")

    return best_model_name


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_markdown_report(
    path: Path,
    training_data: PreparedTrainingData,
    metrics_frame: pd.DataFrame,
    fold_frame: pd.DataFrame,
    best_model_name: str,
) -> None:
    ordered_metrics = metrics_frame.sort_values(
        by=["recall", "f1", "pr_auc"],
        ascending=[False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)

    table_columns = ["model", "recall", "precision", "f1", "roc_auc", "pr_auc", "tp", "fp", "fn", "tn"]
    header = "| " + " | ".join(table_columns) + " |"
    separator = "| " + " | ".join(["---"] * len(table_columns)) + " |"
    rows = []
    for _, row in ordered_metrics.iterrows():
        rows.append(
            "| "
            + " | ".join(
                [
                    str(row["model"]),
                    f'{row["recall"]:.4f}',
                    f'{row["precision"]:.4f}',
                    f'{row["f1"]:.4f}',
                    f'{row["roc_auc"]:.4f}',
                    f'{row["pr_auc"]:.4f}',
                    str(int(row["tp"])),
                    str(int(row["fp"])),
                    str(int(row["fn"])),
                    str(int(row["tn"])),
                ]
            )
            + " |"
        )

    fold_lines = []
    fold_metrics = fold_frame.sort_values(by=["model", "fold"]).reset_index(drop=True)
    for _, row in fold_metrics.iterrows():
        fold_lines.append(
            f'- {row["model"]} fold {row["fold"]}: '
            f'recall={row["recall"]:.4f}, precision={row["precision"]:.4f}, '
            f'f1={row["f1"]:.4f}, roc_auc={row["roc_auc"]:.4f}, pr_auc={row["pr_auc"]:.4f}'
        )

    lines = [
        "# 상장폐지 분류 모델 비교 보고서",
        "",
        "## 데이터 요약",
        f'- 원본 행 수: {training_data.stats["raw_rows"]}',
        f'- has_data=True 적용 후: {training_data.stats["after_has_data"]}',
        f'- 30개 재무비율 모두 결측 제거 후: {training_data.stats["after_all_null_feature_drop"]}',
        f'- 종목코드+연도 중복 정리 후: {training_data.stats["after_dedup"]}',
        f'- 양성(상폐기업): {training_data.stats["positive_rows"]}',
        f'- 음성(상장기업): {training_data.stats["negative_rows"]}',
        f'- 기업 수: {training_data.stats["company_count"]}',
        f'- 사용 feature 수: {len(FEATURE_COLUMNS)}',
        "",
        "## 모델 비교",
        f'- 베스트 모델: **{best_model_name}**',
        f'- 선정 기준: recall 우선, 동률 시 f1 다음 pr_auc',
        "",
        header,
        separator,
        *rows,
        "",
        "## Fold 상세",
        *fold_lines,
        "",
        "## 실행 커맨드",
        "```powershell",
        "python -m pip install -r .\\requirements-ml.txt",
        "python .\\delisting_model.py train --input .\\downloads\\dart_financials\\financial_ratios_2015_2025.csv --output-dir .\\artifacts\\delisting",
        "python .\\delisting_model.py predict --model-dir .\\artifacts\\delisting --input .\\downloads\\dart_financials\\financial_ratios_2024_test.csv --output .\\artifacts\\delisting\\predictions_2024.csv",
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_train(args: argparse.Namespace) -> None:
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = read_csv(input_path)
    training_data = prepare_training_data(raw_df)

    models = build_models(
        seed=args.seed,
        positive_count=training_data.stats["positive_rows"],
        negative_count=training_data.stats["negative_rows"],
    )

    metric_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for model_name, estimator in models.items():
        overall_metrics, fold_metrics = evaluate_model(
            model_name=model_name,
            estimator=estimator,
            X=training_data.features,
            y=training_data.labels,
            groups=training_data.groups,
            cv_folds=args.cv_folds,
            seed=args.seed,
        )
        metric_rows.append(overall_metrics)
        fold_rows.extend(fold_metrics)

    metrics_frame = pd.DataFrame(metric_rows)
    fold_frame = pd.DataFrame(fold_rows)
    best_model_name = fit_and_save_models(models, training_data, metrics_frame, output_dir)

    metrics_frame = metrics_frame.sort_values(
        by=["recall", "f1", "pr_auc"],
        ascending=[False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)

    metrics_frame.to_csv(output_dir / "metrics_summary.csv", index=False, encoding="utf-8-sig")
    fold_frame.to_csv(output_dir / "cv_fold_metrics.csv", index=False, encoding="utf-8-sig")
    write_json(output_dir / "label_mapping.json", LABEL_MAPPING)
    write_json(output_dir / "feature_columns.json", FEATURE_COLUMNS)
    write_json(output_dir / "training_stats.json", training_data.stats)
    write_markdown_report(
        output_dir / "comparison_report.md",
        training_data=training_data,
        metrics_frame=metrics_frame,
        fold_frame=fold_frame,
        best_model_name=best_model_name,
    )

    print(f"학습 완료: {output_dir}")
    print(metrics_frame.to_string(index=False))
    print(f"베스트 모델: {best_model_name}")


def load_model_bundle(model_dir: Path, model_name: str) -> dict[str, Any]:
    model_path_map = {
        "best": model_dir / "best_model.joblib",
        "logistic": model_dir / "model_logistic.joblib",
        "random_forest": model_dir / "model_random_forest.joblib",
        "xgboost": model_dir / "model_xgboost.joblib",
    }
    model_path = model_path_map[model_name]
    if not model_path.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    bundle = joblib.load(model_path)
    return bundle


def run_predict(args: argparse.Namespace) -> None:
    model_dir = args.model_dir.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = load_model_bundle(model_dir, args.model)
    input_df = read_csv(args.input.resolve())
    prepared = prepare_prediction_data(input_df)

    if prepared.working_frame.empty:
        prepared.output_frame.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"예측 가능한 행이 없어 빈 결과만 저장했습니다: {output_path}")
        return

    working_features = prepared.working_frame[bundle["feature_columns"]]
    probabilities = bundle["estimator"].predict_proba(working_features)[:, 1]
    predictions = (probabilities >= bundle.get("threshold", 0.5)).astype(int)

    prediction_frame = prepared.working_frame[[CODE_COLUMN, YEAR_COLUMN]].copy()
    prediction_frame["상폐확률"] = probabilities
    prediction_frame["예측라벨"] = [LABEL_NAME_MAPPING[int(value)] for value in predictions]
    prediction_frame["사용모델"] = bundle["model_name"]
    prediction_frame["예측가능여부"] = True
    prediction_frame["스킵사유"] = ""

    merged = prepared.output_frame.merge(
        prediction_frame,
        on=[CODE_COLUMN, YEAR_COLUMN],
        how="left",
        suffixes=("", "__pred"),
    )

    for column in ["상폐확률", "예측라벨", "사용모델", "예측가능여부", "스킵사유"]:
        pred_column = f"{column}__pred"
        if pred_column in merged.columns:
            if column == "예측가능여부":
                base_bool = merged[column].astype("boolean")
                pred_bool = merged[pred_column].astype("boolean")
                combined = pred_bool.where(pred_bool.notna(), base_bool)
                merged[column] = combined.to_numpy(dtype=bool, na_value=False)
            else:
                merged[column] = merged[pred_column].combine_first(merged[column])
            merged = merged.drop(columns=[pred_column])

    output_columns = META_OUTPUT_COLUMNS + [
        "상폐확률",
        "예측라벨",
        "사용모델",
        "예측가능여부",
        "스킵사유",
    ]
    extra_columns = [column for column in merged.columns if column not in output_columns]
    merged = merged[output_columns + extra_columns]
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"예측 완료: {output_path}")
    print(f"입력 행 수: {len(input_df)}")
    print(f"예측 가능 행 수(중복 정리 후): {len(prepared.working_frame)}")
    print(f"출력 행 수: {len(merged)}")


def main() -> None:
    args = parse_args()
    if args.command == "train":
        run_train(args)
    elif args.command == "predict":
        run_predict(args)
    else:
        raise ValueError(f"지원하지 않는 명령입니다: {args.command}")


if __name__ == "__main__":
    main()
