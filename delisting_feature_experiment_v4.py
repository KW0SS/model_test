from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from delisting_data import build_labeled_dataset, load_or_build_events
from delisting_shared import (
    CODE_COLUMN,
    COMPANY_COLUMN,
    CORP_CODE_COLUMN,
    DEFAULT_COMPANY_MASTER,
    DEFAULT_TRAIN_INPUT,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    YEAR_COLUMN,
    read_csv,
    write_json,
)


WORKDIR = Path(r"C:\kwoss_C\model_test")
DEFAULT_EVENTS = WORKDIR / "artifacts" / "delisting_improved_v3" / "filtered_delisted" / "filtered_delisted_kept.csv"
DEFAULT_OUTPUT_DIR = WORKDIR / "artifacts" / "delisting_feature_experiment_v4"
DEFAULT_THRESHOLDS = [round(value / 100, 2) for value in range(5, 100)]


@dataclass
class DatasetBundle:
    name: str
    frame: pd.DataFrame
    feature_columns: list[str]
    train_start_year: int
    train_end_year: int
    valid_year: int
    test_year: int
    use_winsorization: bool


def build_model(seed: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=seed,
                ),
            ),
        ]
    )


def add_year_standardized_columns(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    output = frame.copy()
    for column in feature_columns:
        grouped = output.groupby(YEAR_COLUMN)[column]
        mean = grouped.transform("mean")
        std = grouped.transform("std").replace(0, np.nan)
        output[f"{column}__year_z"] = ((output[column] - mean) / std).replace([np.inf, -np.inf], np.nan)
    return output


def build_single_year_baseline(labeled_df: pd.DataFrame) -> DatasetBundle:
    usable = labeled_df[labeled_df["include_for_training"]].copy()
    usable[TARGET_COLUMN] = usable[TARGET_COLUMN].astype(int)
    return DatasetBundle(
        name="baseline_single_year",
        frame=usable,
        feature_columns=FEATURE_COLUMNS.copy(),
        train_start_year=2015,
        train_end_year=2020,
        valid_year=2021,
        test_year=2022,
        use_winsorization=False,
    )


def build_single_year_standardized(labeled_df: pd.DataFrame) -> DatasetBundle:
    usable = labeled_df[labeled_df["include_for_training"]].copy()
    usable[TARGET_COLUMN] = usable[TARGET_COLUMN].astype(int)
    usable = add_year_standardized_columns(usable, FEATURE_COLUMNS)
    feature_columns = FEATURE_COLUMNS + [f"{column}__year_z" for column in FEATURE_COLUMNS]
    return DatasetBundle(
        name="single_year_year_standardized",
        frame=usable,
        feature_columns=feature_columns,
        train_start_year=2015,
        train_end_year=2020,
        valid_year=2021,
        test_year=2022,
        use_winsorization=True,
    )


def build_three_year_trend(labeled_df: pd.DataFrame) -> DatasetBundle:
    full = add_year_standardized_columns(labeled_df.copy(), FEATURE_COLUMNS)
    lookup = full.set_index([CODE_COLUMN, YEAR_COLUMN])
    rows: list[dict[str, object]] = []

    anchor_candidates = full[full["include_for_training"]].copy()
    anchor_candidates[TARGET_COLUMN] = anchor_candidates[TARGET_COLUMN].astype(int)

    for anchor in anchor_candidates.itertuples(index=False):
        code = getattr(anchor, CODE_COLUMN)
        anchor_year = int(getattr(anchor, YEAR_COLUMN))
        history_years = [anchor_year - 2, anchor_year - 1, anchor_year]
        if any((code, history_year) not in lookup.index for history_year in history_years):
            continue

        history_frames = [lookup.loc[(code, history_year)] for history_year in history_years]
        if any(
            (not bool(frame["__has_data_bool"])) or bool(frame["all_feature_null"]) or bool(frame["too_many_missing_features"])
            for frame in history_frames
        ):
            continue

        sample = {
            CODE_COLUMN: code,
            CORP_CODE_COLUMN: history_frames[-1][CORP_CODE_COLUMN],
            COMPANY_COLUMN: history_frames[-1][COMPANY_COLUMN],
            YEAR_COLUMN: anchor_year,
            TARGET_COLUMN: int(history_frames[-1][TARGET_COLUMN]),
        }
        oldest, middle, current = history_frames
        for column in FEATURE_COLUMNS:
            sample[f"{column}__anchor"] = current[column]
            sample[f"{column}__anchor_year_z"] = current[f"{column}__year_z"]
            sample[f"{column}__delta_1y"] = (
                current[column] - middle[column] if pd.notna(current[column]) and pd.notna(middle[column]) else np.nan
            )
            sample[f"{column}__delta_2y"] = (
                current[column] - oldest[column] if pd.notna(current[column]) and pd.notna(oldest[column]) else np.nan
            )
            values = [oldest[column], middle[column], current[column]]
            sample[f"{column}__mean_3y"] = np.nan if all(pd.isna(values)) else float(np.nanmean(values))
        rows.append(sample)

    frame = pd.DataFrame(rows)
    feature_columns = [column for column in frame.columns if column not in {CODE_COLUMN, CORP_CODE_COLUMN, COMPANY_COLUMN, YEAR_COLUMN, TARGET_COLUMN}]
    return DatasetBundle(
        name="trend_3year_year_standardized",
        frame=frame,
        feature_columns=feature_columns,
        train_start_year=2017,
        train_end_year=2020,
        valid_year=2021,
        test_year=2022,
        use_winsorization=True,
    )


def split_bundle(bundle: DatasetBundle) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = bundle.frame.copy()
    train = frame[frame[YEAR_COLUMN].between(bundle.train_start_year, bundle.train_end_year)].copy()
    valid = frame[frame[YEAR_COLUMN] == bundle.valid_year].copy()
    test = frame[frame[YEAR_COLUMN] == bundle.test_year].copy()
    if train.empty or valid.empty or test.empty:
        raise ValueError(f"{bundle.name}: split result is empty")
    return train, valid, test


def apply_winsorization(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_columns: list[str],
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    low = train_frame[feature_columns].quantile(lower_quantile)
    high = train_frame[feature_columns].quantile(upper_quantile)
    outputs = []
    for frame in [train_frame.copy(), valid_frame.copy(), test_frame.copy()]:
        frame[feature_columns] = frame[feature_columns].clip(low, high, axis=1)
        outputs.append(frame)
    return outputs[0], outputs[1], outputs[2]


def select_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    best_score: tuple[float, float, float, float] | None = None
    best_threshold = 0.5
    for threshold in DEFAULT_THRESHOLDS:
        predictions = (probabilities >= threshold).astype(int)
        precision = precision_score(y_true, predictions, zero_division=0)
        recall = recall_score(y_true, predictions, zero_division=0)
        f1 = f1_score(y_true, predictions, zero_division=0)
        score = (f1, precision, recall, -threshold)
        if best_score is None or score > best_score:
            best_score = score
            best_threshold = threshold
    return float(best_threshold)


def compute_split_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "tp": int(((predictions == 1) & (y_true == 1)).sum()),
        "fp": int(((predictions == 1) & (y_true == 0)).sum()),
        "fn": int(((predictions == 0) & (y_true == 1)).sum()),
        "tn": int(((predictions == 0) & (y_true == 0)).sum()),
    }


def run_experiment(bundle: DatasetBundle) -> tuple[dict[str, object], pd.DataFrame]:
    train_frame, valid_frame, test_frame = split_bundle(bundle)
    if bundle.use_winsorization:
        train_frame, valid_frame, test_frame = apply_winsorization(train_frame, valid_frame, test_frame, bundle.feature_columns)

    model = build_model()
    model.fit(train_frame[bundle.feature_columns], train_frame[TARGET_COLUMN].astype(int))
    valid_probabilities = model.predict_proba(valid_frame[bundle.feature_columns])[:, 1]
    test_probabilities = model.predict_proba(test_frame[bundle.feature_columns])[:, 1]
    selected_threshold = select_threshold(valid_frame[TARGET_COLUMN].to_numpy(), valid_probabilities)

    threshold_rows = []
    for threshold in DEFAULT_THRESHOLDS:
        metrics = compute_split_metrics(valid_frame[TARGET_COLUMN].to_numpy(), valid_probabilities, threshold)
        threshold_rows.append(
            {
                "experiment": bundle.name,
                "threshold": threshold,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "tn": metrics["tn"],
            }
        )

    valid_metrics = compute_split_metrics(valid_frame[TARGET_COLUMN].to_numpy(), valid_probabilities, selected_threshold)
    test_metrics = compute_split_metrics(test_frame[TARGET_COLUMN].to_numpy(), test_probabilities, selected_threshold)
    summary = {
        "experiment": bundle.name,
        "train_start_year": bundle.train_start_year,
        "train_end_year": bundle.train_end_year,
        "valid_year": bundle.valid_year,
        "test_year": bundle.test_year,
        "feature_count": len(bundle.feature_columns),
        "train_rows": int(len(train_frame)),
        "valid_rows": int(len(valid_frame)),
        "test_rows": int(len(test_frame)),
        "train_positive_rows": int((train_frame[TARGET_COLUMN] == 1).sum()),
        "valid_positive_rows": int((valid_frame[TARGET_COLUMN] == 1).sum()),
        "test_positive_rows": int((test_frame[TARGET_COLUMN] == 1).sum()),
        "selected_threshold": selected_threshold,
        "valid_precision": valid_metrics["precision"],
        "valid_recall": valid_metrics["recall"],
        "valid_f1": valid_metrics["f1"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "test_tp": test_metrics["tp"],
        "test_fp": test_metrics["fp"],
        "test_fn": test_metrics["fn"],
        "test_tn": test_metrics["tn"],
    }
    return summary, pd.DataFrame(threshold_rows)


def write_summary_markdown(path: Path, summary_frame: pd.DataFrame) -> None:
    lines = [
        "# Delisting Feature Experiment V4",
        "",
        "업종 컬럼이 없어 `업종/연도 표준화` 대신 `연도 표준화`로 실험했습니다.",
        "",
        "## Experiment Results",
        "| experiment | feature_count | selected_threshold | valid_precision | valid_recall | valid_f1 | test_precision | test_recall | test_f1 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in summary_frame.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["experiment"]),
                    str(int(row["feature_count"])),
                    f'{row["selected_threshold"]:.2f}',
                    f'{row["valid_precision"]:.4f}',
                    f'{row["valid_recall"]:.4f}',
                    f'{row["valid_f1"]:.4f}',
                    f'{row["test_precision"]:.4f}',
                    f'{row["test_recall"]:.4f}',
                    f'{row["test_f1"]:.4f}',
                ]
            )
            + " |"
        )
    best = summary_frame.sort_values(["test_f1", "test_precision", "test_recall"], ascending=[False, False, False]).iloc[0]
    lines.extend(
        [
            "",
            "## Notes",
            f'- Best test F1 experiment: `{best["experiment"]}`',
            "- `baseline_single_year`: V3와 비슷한 단일 연도 재무비율 구조",
            "- `single_year_year_standardized`: 단일 연도 재무비율 + 연도 표준화 + 이상치 처리",
            "- `trend_3year_year_standardized`: 최근 3개년 anchor 값 + 변화량 + 3년 평균 + 연도 표준화 + 이상치 처리",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = read_csv(DEFAULT_TRAIN_INPUT)
    events_df = load_or_build_events(output_dir, DEFAULT_EVENTS, DEFAULT_COMPANY_MASTER)
    labeled_df, _, _ = build_labeled_dataset(raw_df, events_df)

    bundles = [
        build_single_year_baseline(labeled_df),
        build_single_year_standardized(labeled_df),
        build_three_year_trend(labeled_df),
    ]

    summaries = []
    threshold_frames = []
    for bundle in bundles:
        summary, threshold_frame = run_experiment(bundle)
        summaries.append(summary)
        threshold_frames.append(threshold_frame)

    summary_frame = pd.DataFrame(summaries).sort_values(
        ["test_f1", "test_precision", "test_recall"],
        ascending=[False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    threshold_frame = pd.concat(threshold_frames, ignore_index=True)

    summary_frame.to_csv(output_dir / "experiment_summary.csv", index=False, encoding="utf-8-sig")
    threshold_frame.to_csv(output_dir / "experiment_thresholds.csv", index=False, encoding="utf-8-sig")
    write_summary_markdown(output_dir / "experiment_summary.md", summary_frame)
    write_json(output_dir / "best_experiment.json", summary_frame.iloc[0].to_dict())

    print(summary_frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
