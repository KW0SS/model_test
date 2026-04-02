from __future__ import annotations

from pathlib import Path

import pandas as pd

from delisting_data import (
    attach_events,
    coerce_financial_frame,
    deduplicate_company_year,
    load_or_build_events,
    prepare_training_data,
)
from delisting_shared import (
    CODE_COLUMN,
    COMPANY_COLUMN,
    CORP_CODE_COLUMN,
    DEFAULT_COMPANY_MASTER,
    DEFAULT_THRESHOLDS,
    DEFAULT_TRAIN_INPUT,
    EVENT_DATE_COLUMN,
    EVENT_SOURCE_COLUMN,
    EVENT_YEAR_COLUMN,
    EXCLUDE_REASON_COLUMN,
    FEATURE_COLUMNS,
    FS_COLUMN,
    HAS_DATA_COLUMN,
    INCLUDE_COLUMN,
    MAX_ALLOWED_MISSING_FEATURES,
    SOURCE_FILE_COLUMN,
    STATUS_COLUMN,
    TARGET_COLUMN,
    YEAR_COLUMN,
    Y_MINUS_1_EXCLUDED_COLUMN,
    read_csv,
    write_json,
)
from delisting_train import build_logistic_model, build_threshold_comparison, compute_metrics, select_threshold


WORKDIR = Path(r"C:\kwoss_C\model_test")
DEFAULT_EVENTS = WORKDIR / "artifacts" / "delisting_improved_v3" / "filtered_delisted" / "filtered_delisted_kept.csv"
DEFAULT_OUTPUT_DIR = WORKDIR / "artifacts" / "delisting_label_window_experiment"


def build_labeled_dataset_with_offsets(
    raw_df: pd.DataFrame,
    events_df: pd.DataFrame,
    *,
    positive_offsets: set[int],
    excluded_offsets: set[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    coerced = coerce_financial_frame(raw_df)
    group_stats = (
        coerced.groupby([CODE_COLUMN, YEAR_COLUMN], dropna=False)
        .agg(
            raw_row_count=(CODE_COLUMN, "size"),
            status_values=(STATUS_COLUMN, lambda s: "|".join(sorted({str(v) for v in s.dropna()}))),
        )
        .reset_index()
    )
    group_stats["status_conflict"] = group_stats["status_values"].str.contains(r"\|", regex=True)

    deduped = deduplicate_company_year(coerced)
    deduped = deduped.merge(group_stats, on=[CODE_COLUMN, YEAR_COLUMN], how="left")
    labeled = attach_events(deduped, events_df)
    labeled["missing_feature_count"] = labeled[FEATURE_COLUMNS].isna().sum(axis=1).astype(int)
    labeled["all_feature_null"] = labeled[FEATURE_COLUMNS].isna().all(axis=1)
    labeled["too_many_missing_features"] = labeled["missing_feature_count"] > MAX_ALLOWED_MISSING_FEATURES

    event_year = labeled[EVENT_YEAR_COLUMN].astype("Int64")
    positive_mask = pd.Series(False, index=labeled.index)
    for offset in sorted(positive_offsets):
        positive_mask = positive_mask | (labeled[EVENT_YEAR_COLUMN].notna() & (labeled[YEAR_COLUMN] == (event_year - offset)))

    excluded_window_mask = pd.Series(False, index=labeled.index)
    for offset in sorted(excluded_offsets):
        offset_mask = labeled[EVENT_YEAR_COLUMN].notna() & (labeled[YEAR_COLUMN] == (event_year - offset))
        if offset == 1:
            offset_mask = offset_mask & labeled[Y_MINUS_1_EXCLUDED_COLUMN]
        excluded_window_mask = excluded_window_mask | offset_mask

    excluded_event_or_after_mask = labeled[EVENT_YEAR_COLUMN].notna() & (labeled[YEAR_COLUMN] >= event_year)

    labeled["is_positive_candidate"] = positive_mask
    labeled["is_excluded_window"] = excluded_window_mask
    labeled["is_excluded_event_or_after"] = excluded_event_or_after_mask
    labeled["event_offset"] = (event_year - labeled[YEAR_COLUMN]).astype("Int64")

    labeled[EXCLUDE_REASON_COLUMN] = ""
    labeled.loc[labeled["__has_data_bool"].eq(False), EXCLUDE_REASON_COLUMN] = "has_data_false"
    labeled.loc[labeled["all_feature_null"], EXCLUDE_REASON_COLUMN] = "all_features_null"
    labeled.loc[labeled["too_many_missing_features"], EXCLUDE_REASON_COLUMN] = "too_many_missing_features"
    labeled.loc[labeled["status_conflict"].fillna(False), EXCLUDE_REASON_COLUMN] = "status_conflict_duplicate"
    labeled.loc[excluded_event_or_after_mask, EXCLUDE_REASON_COLUMN] = "event_year_or_later"
    labeled.loc[excluded_window_mask, EXCLUDE_REASON_COLUMN] = "excluded_window"

    labeled[INCLUDE_COLUMN] = labeled[EXCLUDE_REASON_COLUMN].eq("")
    labeled[TARGET_COLUMN] = pd.Series([pd.NA] * len(labeled), dtype="Int64")
    labeled.loc[labeled[INCLUDE_COLUMN], TARGET_COLUMN] = 0
    labeled.loc[labeled[INCLUDE_COLUMN] & positive_mask, TARGET_COLUMN] = 1

    quality_columns = [
        COMPANY_COLUMN,
        CODE_COLUMN,
        CORP_CODE_COLUMN,
        YEAR_COLUMN,
        STATUS_COLUMN,
        FS_COLUMN,
        HAS_DATA_COLUMN,
        "raw_row_count",
        "status_values",
        "status_conflict",
        "missing_feature_count",
        "all_feature_null",
        "too_many_missing_features",
        EVENT_YEAR_COLUMN,
        EVENT_DATE_COLUMN,
        EVENT_SOURCE_COLUMN,
        Y_MINUS_1_EXCLUDED_COLUMN,
        "event_offset",
        "is_positive_candidate",
        "is_excluded_window",
        "is_excluded_event_or_after",
        TARGET_COLUMN,
        INCLUDE_COLUMN,
        EXCLUDE_REASON_COLUMN,
        SOURCE_FILE_COLUMN,
    ]
    return labeled, labeled[quality_columns].copy()


def run_experiment(
    *,
    name: str,
    positive_offsets: set[int],
    excluded_offsets: set[int],
    raw_df: pd.DataFrame,
    events_df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, object]:
    labeled_df, quality_report = build_labeled_dataset_with_offsets(
        raw_df,
        events_df,
        positive_offsets=positive_offsets,
        excluded_offsets=excluded_offsets,
    )
    experiment_dir = output_dir / name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    labeled_df.to_csv(experiment_dir / "future_delist_labeled.csv", index=False, encoding="utf-8-sig")
    quality_report.to_csv(experiment_dir / "data_quality_report.csv", index=False, encoding="utf-8-sig")

    training_data = prepare_training_data(
        labeled_df=labeled_df,
        train_start_year=2015,
        train_end_year=2020,
        valid_year=2021,
        test_year=2022,
    )

    model = build_logistic_model(42)
    X_train = training_data.train_frame[FEATURE_COLUMNS]
    y_train = training_data.train_frame[TARGET_COLUMN].astype(int)
    X_valid = training_data.valid_frame[FEATURE_COLUMNS]
    y_valid = training_data.valid_frame[TARGET_COLUMN].astype(int)
    X_test = training_data.test_frame[FEATURE_COLUMNS]
    y_test = training_data.test_frame[TARGET_COLUMN].astype(int)

    model.fit(X_train, y_train)
    valid_probabilities = model.predict_proba(X_valid)[:, 1]
    test_probabilities = model.predict_proba(X_test)[:, 1]

    threshold_frame = build_threshold_comparison(training_data.valid_frame, valid_probabilities, DEFAULT_THRESHOLDS)
    selected_threshold = select_threshold(threshold_frame)
    threshold_frame.to_csv(experiment_dir / "threshold_comparison.csv", index=False, encoding="utf-8-sig")

    valid_metrics = compute_metrics(y_valid.to_numpy(), valid_probabilities, selected_threshold)
    test_metrics = compute_metrics(y_test.to_numpy(), test_probabilities, selected_threshold)
    split_metrics = pd.DataFrame(
        [
            {"split": "valid", "threshold_type": "selected", **valid_metrics},
            {"split": "test", "threshold_type": "selected", **test_metrics},
        ]
    )
    split_metrics.to_csv(experiment_dir / "split_metrics.csv", index=False, encoding="utf-8-sig")

    positive_breakdown = (
        training_data.frame[training_data.frame[TARGET_COLUMN] == 1]["event_offset"]
        .value_counts(dropna=False)
        .sort_index()
        .to_dict()
    )

    summary = {
        "experiment": name,
        "positive_offsets": sorted(positive_offsets),
        "excluded_offsets": sorted(excluded_offsets),
        "selected_threshold": float(selected_threshold),
        "usable_rows": int(training_data.stats["usable_rows"]),
        "positive_rows": int(training_data.stats["positive_rows"]),
        "train_positive_rows": int(training_data.stats["train_positive_rows"]),
        "valid_positive_rows": int(training_data.stats["valid_positive_rows"]),
        "test_positive_rows": int(training_data.stats["test_positive_rows"]),
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
        "excluded_window_rows": int((labeled_df[EXCLUDE_REASON_COLUMN] == "excluded_window").sum()),
        "excluded_event_or_after_rows": int((labeled_df[EXCLUDE_REASON_COLUMN] == "event_year_or_later").sum()),
        "positive_offset_breakdown": {str(key): int(value) for key, value in positive_breakdown.items()},
    }
    write_json(experiment_dir / "summary.json", summary)
    return summary


def write_markdown(path: Path, summary_frame: pd.DataFrame) -> None:
    lines = [
        "# Delisting Label Window Experiment",
        "",
        "기존 `Y-2 only` 기준과 `Y-1 + Y-2` 기준을 같은 데이터/같은 Logistic 모델로 비교했습니다.",
        "",
        "| experiment | positive_offsets | excluded_offsets | threshold | valid_precision | valid_recall | valid_f1 | test_precision | test_recall | test_f1 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in summary_frame.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["experiment"]),
                    str(row["positive_offsets"]),
                    str(row["excluded_offsets"]),
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
    lines.extend(
        [
            "",
            "## Interpretation",
            "- `baseline_y2_only`: 기존처럼 `Y-2`만 positive, `Y-1`은 제외",
            "- `experiment_y1_y2`: `Y-1`과 `Y-2`를 모두 positive로 사용, `Y-1` 제외 없음",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = read_csv(DEFAULT_TRAIN_INPUT)
    events_df = load_or_build_events(output_dir, DEFAULT_EVENTS, DEFAULT_COMPANY_MASTER)

    experiments = [
        ("baseline_y2_only", {2}, {1}),
        ("experiment_y1_y2", {1, 2}, set()),
    ]

    summaries = []
    for name, positive_offsets, excluded_offsets in experiments:
        summaries.append(
            run_experiment(
                name=name,
                positive_offsets=positive_offsets,
                excluded_offsets=excluded_offsets,
                raw_df=raw_df,
                events_df=events_df,
                output_dir=output_dir,
            )
        )

    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(output_dir / "experiment_summary.csv", index=False, encoding="utf-8-sig")
    write_markdown(output_dir / "experiment_summary.md", summary_frame)
    write_json(output_dir / "experiment_summary.json", summary_frame.to_dict(orient="records"))
    print(summary_frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
