from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from delisting_shared import (
    ALL_REQUIRED_COLUMNS,
    CODE_COLUMN,
    COMPANY_COLUMN,
    CORP_CODE_COLUMN,
    EVENT_DATE_COLUMN,
    EVENT_SOURCE_COLUMN,
    EVENT_YEAR_COLUMN,
    EXCLUDE_REASON_COLUMN,
    FEATURE_COLUMNS,
    FS_COLUMN,
    HAS_DATA_COLUMN,
    INCLUDE_COLUMN,
    SOURCE_FILE_COLUMN,
    STATUS_COLUMN,
    STATUS_PRIORITY,
    TARGET_COLUMN,
    YEAR_COLUMN,
    Y_MINUS_1_EXCLUDED_COLUMN,
    ensure_columns,
    normalize_corp_code,
    normalize_stock_code,
    parse_bool_series,
    read_company_master,
    read_csv,
)


@dataclass
class PreparedTrainingData:
    frame: pd.DataFrame
    train_frame: pd.DataFrame
    valid_frame: pd.DataFrame
    test_frame: pd.DataFrame
    stats: dict[str, int]


@dataclass
class PreparedPredictionData:
    working_frame: pd.DataFrame
    output_frame: pd.DataFrame
    stats: dict[str, int]


def coerce_financial_frame(df: pd.DataFrame) -> pd.DataFrame:
    ensure_columns(df)
    working = df.copy()
    working[CODE_COLUMN] = normalize_stock_code(working[CODE_COLUMN])
    working[CORP_CODE_COLUMN] = normalize_corp_code(working[CORP_CODE_COLUMN])
    working[YEAR_COLUMN] = pd.to_numeric(working[YEAR_COLUMN], errors="raise").astype(int)
    working["__has_data_bool"] = parse_bool_series(working[HAS_DATA_COLUMN])
    working["__fs_priority"] = working[FS_COLUMN].astype("string").fillna("").str.upper().eq("CFS").astype(int)
    working["__status_priority"] = working[STATUS_COLUMN].astype("string").map(STATUS_PRIORITY).fillna(-1).astype(int)
    return working


def build_delist_events_from_master(company_master_path: Path) -> pd.DataFrame:
    company_master = read_company_master(company_master_path)
    required = {"corp_code", "corp_name", "stock_code", "modify_date", "status", "source"}
    missing = sorted(required - set(company_master.columns))
    if missing:
        raise ValueError(f"company_master.json에 필요한 컬럼이 없습니다: {missing}")

    working = company_master.copy()
    working = working[working["status"].astype("string").str.lower() == "delisted"].copy()
    working["stock_code"] = normalize_stock_code(working["stock_code"])
    working["corp_code"] = normalize_corp_code(working["corp_code"])
    working["modify_date"] = working["modify_date"].astype("string").str.strip()
    working = working[working["stock_code"].ne("") & working["corp_code"].ne("")]
    working[EVENT_YEAR_COLUMN] = pd.to_numeric(
        working["modify_date"].str.extract(r"^(\d{4})", expand=False),
        errors="coerce",
    ).astype("Int64")
    working = working[working[EVENT_YEAR_COLUMN].notna()].copy()
    working[EVENT_DATE_COLUMN] = working["modify_date"]
    working[EVENT_SOURCE_COLUMN] = (
        working["source"].astype("string").fillna("")
        + " | event_year derived from company_master.modify_date"
    )
    working[Y_MINUS_1_EXCLUDED_COLUMN] = True
    working = working.rename(
        columns={
            "stock_code": CODE_COLUMN,
            "corp_code": CORP_CODE_COLUMN,
            "corp_name": COMPANY_COLUMN,
        }
    )
    columns = [
        CODE_COLUMN,
        CORP_CODE_COLUMN,
        COMPANY_COLUMN,
        EVENT_YEAR_COLUMN,
        EVENT_DATE_COLUMN,
        EVENT_SOURCE_COLUMN,
        Y_MINUS_1_EXCLUDED_COLUMN,
    ]
    working = working[columns].copy()
    working = working.sort_values(
        by=[CORP_CODE_COLUMN, CODE_COLUMN, EVENT_YEAR_COLUMN, EVENT_DATE_COLUMN],
        ascending=[True, True, True, True],
        kind="mergesort",
    )
    return working.drop_duplicates(subset=[CORP_CODE_COLUMN, CODE_COLUMN], keep="first").reset_index(drop=True)


def normalize_event_frame_schema(events_df: pd.DataFrame) -> pd.DataFrame:
    working = events_df.copy()

    if EVENT_YEAR_COLUMN not in working.columns:
        if "상폐일" in working.columns:
            parsed_date = pd.to_datetime(working["상폐일"], errors="coerce")
            working[EVENT_YEAR_COLUMN] = parsed_date.dt.year.astype("Int64")
            working[EVENT_DATE_COLUMN] = parsed_date.dt.strftime("%Y-%m-%d")
        else:
            raise ValueError("상폐 사건 CSV에 event_year_Y 또는 상폐일 컬럼이 필요합니다.")

    if CODE_COLUMN not in working.columns:
        raise ValueError("상폐 사건 CSV에 종목코드 컬럼이 필요합니다.")

    if COMPANY_COLUMN not in working.columns:
        working[COMPANY_COLUMN] = pd.Series([pd.NA] * len(working), dtype="string")

    if CORP_CODE_COLUMN not in working.columns:
        working[CORP_CODE_COLUMN] = pd.Series([""] * len(working), dtype="string")

    if EVENT_DATE_COLUMN not in working.columns:
        if "상폐일" in working.columns:
            working[EVENT_DATE_COLUMN] = working["상폐일"].astype("string")
        else:
            working[EVENT_DATE_COLUMN] = pd.Series([pd.NA] * len(working), dtype="string")

    if EVENT_SOURCE_COLUMN not in working.columns:
        if "폐지사유" in working.columns:
            working[EVENT_SOURCE_COLUMN] = "폐지사유: " + working["폐지사유"].astype("string").fillna("")
        else:
            working[EVENT_SOURCE_COLUMN] = pd.Series(["user_event_csv"] * len(working), dtype="string")

    if Y_MINUS_1_EXCLUDED_COLUMN not in working.columns:
        working[Y_MINUS_1_EXCLUDED_COLUMN] = True

    columns = [
        CODE_COLUMN,
        CORP_CODE_COLUMN,
        COMPANY_COLUMN,
        EVENT_YEAR_COLUMN,
        EVENT_DATE_COLUMN,
        EVENT_SOURCE_COLUMN,
        Y_MINUS_1_EXCLUDED_COLUMN,
    ]
    return working[columns].copy()


def load_or_build_events(output_dir: Path, events_path: Path | None, company_master_path: Path) -> pd.DataFrame:
    if events_path is not None:
        events_df = normalize_event_frame_schema(read_csv(events_path))
    else:
        events_df = build_delist_events_from_master(company_master_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        events_df.to_csv(output_dir / "delist_events.csv", index=False, encoding="utf-8-sig")

    required = {
        CODE_COLUMN,
        CORP_CODE_COLUMN,
        EVENT_YEAR_COLUMN,
        EVENT_SOURCE_COLUMN,
        Y_MINUS_1_EXCLUDED_COLUMN,
    }
    missing = sorted(required - set(events_df.columns))
    if missing:
        raise ValueError(f"상폐 사건 CSV에 필요한 컬럼이 없습니다: {missing}")

    working = events_df.copy()
    working[CODE_COLUMN] = normalize_stock_code(working[CODE_COLUMN])
    working[CORP_CODE_COLUMN] = normalize_corp_code(working[CORP_CODE_COLUMN])
    working[EVENT_YEAR_COLUMN] = pd.to_numeric(working[EVENT_YEAR_COLUMN], errors="coerce").astype("Int64")
    working[Y_MINUS_1_EXCLUDED_COLUMN] = parse_bool_series(working[Y_MINUS_1_EXCLUDED_COLUMN])
    if COMPANY_COLUMN not in working.columns:
        working[COMPANY_COLUMN] = pd.Series([pd.NA] * len(working), dtype="string")
    if EVENT_DATE_COLUMN not in working.columns:
        working[EVENT_DATE_COLUMN] = pd.Series([pd.NA] * len(working), dtype="string")
    working = working.dropna(subset=[EVENT_YEAR_COLUMN]).copy()
    working = working.sort_values(
        by=[CORP_CODE_COLUMN, CODE_COLUMN, EVENT_YEAR_COLUMN],
        ascending=[True, True, True],
        kind="mergesort",
    )
    return working.drop_duplicates(subset=[CORP_CODE_COLUMN, CODE_COLUMN], keep="first").reset_index(drop=True)


def build_conflict_report(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby([CODE_COLUMN, YEAR_COLUMN], dropna=False)
        .agg(
            raw_row_count=(CODE_COLUMN, "size"),
            company_names=(COMPANY_COLUMN, lambda s: "|".join(sorted({str(v) for v in s.dropna()}))),
            corp_codes=(CORP_CODE_COLUMN, lambda s: "|".join(sorted({str(v) for v in s.dropna()}))),
            status_values=(STATUS_COLUMN, lambda s: "|".join(sorted({str(v) for v in s.dropna()}))),
            source_files=(SOURCE_FILE_COLUMN, lambda s: "|".join(sorted({str(v) for v in s.dropna()}))),
        )
        .reset_index()
    )
    grouped["distinct_status_count"] = grouped["status_values"].str.split("|").apply(
        lambda values: len([value for value in values if value])
    )
    grouped["status_conflict"] = grouped["distinct_status_count"] > 1
    conflicts = grouped[(grouped["raw_row_count"] > 1) | grouped["status_conflict"]].copy()
    return conflicts.sort_values([CODE_COLUMN, YEAR_COLUMN], kind="mergesort").reset_index(drop=True)


def deduplicate_company_year(df: pd.DataFrame) -> pd.DataFrame:
    ordered = df.sort_values(
        by=[CODE_COLUMN, YEAR_COLUMN, "__has_data_bool", "__fs_priority", "__status_priority", SOURCE_FILE_COLUMN],
        ascending=[True, True, False, False, False, True],
        kind="mergesort",
    )
    return ordered.drop_duplicates(subset=[CODE_COLUMN, YEAR_COLUMN], keep="first").reset_index(drop=True)


def attach_events(frame: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    columns = [
        CORP_CODE_COLUMN,
        CODE_COLUMN,
        EVENT_YEAR_COLUMN,
        EVENT_DATE_COLUMN,
        EVENT_SOURCE_COLUMN,
        Y_MINUS_1_EXCLUDED_COLUMN,
    ]
    merged = frame.merge(events[columns], on=[CORP_CODE_COLUMN, CODE_COLUMN], how="left")
    missing_event_mask = merged[EVENT_YEAR_COLUMN].isna()
    if missing_event_mask.any():
        fallback = frame.loc[missing_event_mask].merge(
            events[[CODE_COLUMN, EVENT_YEAR_COLUMN, EVENT_DATE_COLUMN, EVENT_SOURCE_COLUMN, Y_MINUS_1_EXCLUDED_COLUMN]],
            on=CODE_COLUMN,
            how="left",
        )
        for column in [EVENT_YEAR_COLUMN, EVENT_DATE_COLUMN, EVENT_SOURCE_COLUMN, Y_MINUS_1_EXCLUDED_COLUMN]:
            merged.loc[missing_event_mask, column] = fallback[column].values
    merged[Y_MINUS_1_EXCLUDED_COLUMN] = parse_bool_series(
        merged[Y_MINUS_1_EXCLUDED_COLUMN].astype("string").fillna("False")
    )
    merged[EVENT_YEAR_COLUMN] = pd.to_numeric(merged[EVENT_YEAR_COLUMN], errors="coerce").astype("Int64")
    return merged


def build_labeled_dataset(raw_df: pd.DataFrame, events_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    labeled["all_feature_null"] = labeled[FEATURE_COLUMNS].isna().all(axis=1)

    event_year = labeled[EVENT_YEAR_COLUMN].astype("Int64")
    positive_mask = labeled[EVENT_YEAR_COLUMN].notna() & (labeled[YEAR_COLUMN] == (event_year - 2))
    excluded_y_minus_1_mask = (
        labeled[EVENT_YEAR_COLUMN].notna()
        & labeled[Y_MINUS_1_EXCLUDED_COLUMN]
        & (labeled[YEAR_COLUMN] == (event_year - 1))
    )
    excluded_event_or_after_mask = labeled[EVENT_YEAR_COLUMN].notna() & (labeled[YEAR_COLUMN] >= event_year)

    labeled["is_positive_candidate"] = positive_mask
    labeled["is_excluded_y_minus_1"] = excluded_y_minus_1_mask
    labeled["is_excluded_event_or_after"] = excluded_event_or_after_mask
    labeled[EXCLUDE_REASON_COLUMN] = ""
    labeled.loc[labeled["__has_data_bool"].eq(False), EXCLUDE_REASON_COLUMN] = "has_data_false"
    labeled.loc[labeled["all_feature_null"], EXCLUDE_REASON_COLUMN] = "all_features_null"
    labeled.loc[excluded_event_or_after_mask, EXCLUDE_REASON_COLUMN] = "event_year_or_later"
    labeled.loc[excluded_y_minus_1_mask, EXCLUDE_REASON_COLUMN] = "excluded_y_minus_1"

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
        "all_feature_null",
        EVENT_YEAR_COLUMN,
        EVENT_DATE_COLUMN,
        EVENT_SOURCE_COLUMN,
        Y_MINUS_1_EXCLUDED_COLUMN,
        "is_positive_candidate",
        "is_excluded_y_minus_1",
        "is_excluded_event_or_after",
        TARGET_COLUMN,
        INCLUDE_COLUMN,
        EXCLUDE_REASON_COLUMN,
        SOURCE_FILE_COLUMN,
    ]
    quality_report = labeled[quality_columns].copy()
    excluded_y_minus_1 = quality_report[quality_report["is_excluded_y_minus_1"]].copy()
    return labeled, quality_report, excluded_y_minus_1


def prepare_training_data(
    labeled_df: pd.DataFrame,
    train_start_year: int,
    train_end_year: int,
    valid_year: int,
    test_year: int,
) -> PreparedTrainingData:
    usable = labeled_df[labeled_df[INCLUDE_COLUMN]].copy()
    usable[TARGET_COLUMN] = usable[TARGET_COLUMN].astype(int)

    train_frame = usable[usable[YEAR_COLUMN].between(train_start_year, train_end_year)].copy()
    valid_frame = usable[usable[YEAR_COLUMN] == valid_year].copy()
    test_frame = usable[usable[YEAR_COLUMN] == test_year].copy()

    if train_frame.empty or valid_frame.empty or test_frame.empty:
        raise ValueError("시간 기준 split 결과가 비어 있습니다. 연도 설정을 확인해주세요.")
    if train_frame[TARGET_COLUMN].nunique() < 2:
        raise ValueError("학습 구간에 양성과 음성이 모두 필요합니다.")

    stats = {
        "raw_rows": int(len(labeled_df)),
        "usable_rows": int(len(usable)),
        "positive_rows": int((usable[TARGET_COLUMN] == 1).sum()),
        "negative_rows": int((usable[TARGET_COLUMN] == 0).sum()),
        "excluded_rows": int((~labeled_df[INCLUDE_COLUMN]).sum()),
        "excluded_y_minus_1_rows": int((labeled_df[EXCLUDE_REASON_COLUMN] == "excluded_y_minus_1").sum()),
        "excluded_event_or_after_rows": int((labeled_df[EXCLUDE_REASON_COLUMN] == "event_year_or_later").sum()),
        "excluded_has_data_false_rows": int((labeled_df[EXCLUDE_REASON_COLUMN] == "has_data_false").sum()),
        "excluded_all_features_null_rows": int((labeled_df[EXCLUDE_REASON_COLUMN] == "all_features_null").sum()),
        "train_rows": int(len(train_frame)),
        "train_positive_rows": int((train_frame[TARGET_COLUMN] == 1).sum()),
        "train_negative_rows": int((train_frame[TARGET_COLUMN] == 0).sum()),
        "valid_rows": int(len(valid_frame)),
        "valid_positive_rows": int((valid_frame[TARGET_COLUMN] == 1).sum()),
        "valid_negative_rows": int((valid_frame[TARGET_COLUMN] == 0).sum()),
        "test_rows": int(len(test_frame)),
        "test_positive_rows": int((test_frame[TARGET_COLUMN] == 1).sum()),
        "test_negative_rows": int((test_frame[TARGET_COLUMN] == 0).sum()),
        "year_min": int(usable[YEAR_COLUMN].min()),
        "year_max": int(usable[YEAR_COLUMN].max()),
        "company_count": int(usable[CODE_COLUMN].nunique()),
    }
    return PreparedTrainingData(usable, train_frame, valid_frame, test_frame, stats)


def prepare_prediction_data(df: pd.DataFrame) -> PreparedPredictionData:
    working = coerce_financial_frame(df)
    output = df.copy()
    output["상폐확률"] = pd.Series([pd.NA] * len(output), dtype="Float64")
    output["예측라벨"] = pd.Series([pd.NA] * len(output), dtype="string")
    output["사용모델"] = pd.Series([pd.NA] * len(output), dtype="string")
    output["예측가능여부"] = False
    output["제외사유"] = pd.Series([""] * len(output), dtype="string")

    deduped = deduplicate_company_year(working)
    deduped["all_feature_null"] = deduped[FEATURE_COLUMNS].isna().all(axis=1)
    eligible = deduped[deduped["__has_data_bool"] & ~deduped["all_feature_null"]].copy()

    skip_frame = deduped[[CODE_COLUMN, YEAR_COLUMN, "__has_data_bool", "all_feature_null"]].copy()
    skip_frame["제외사유"] = ""
    skip_frame.loc[~skip_frame["__has_data_bool"], "제외사유"] = "has_data_false"
    skip_frame.loc[skip_frame["all_feature_null"], "제외사유"] = "all_features_null"
    output = output.merge(
        skip_frame[[CODE_COLUMN, YEAR_COLUMN, "제외사유"]],
        on=[CODE_COLUMN, YEAR_COLUMN],
        how="left",
        suffixes=("", "__derived"),
    )
    output["제외사유"] = output["제외사유__derived"].combine_first(output["제외사유"])
    output = output.drop(columns=["제외사유__derived"])

    stats = {
        "input_rows": int(len(df)),
        "deduped_rows": int(len(deduped)),
        "eligible_rows": int(len(eligible)),
        "skipped_has_data_false_rows": int((~deduped["__has_data_bool"]).sum()),
        "skipped_all_feature_null_rows": int(deduped["all_feature_null"].sum()),
    }
    return PreparedPredictionData(eligible, output, stats)


def save_preparation_outputs(
    output_dir: Path,
    events_df: pd.DataFrame,
    labeled_df: pd.DataFrame,
    quality_report: pd.DataFrame,
    conflicts: pd.DataFrame,
    excluded_y_minus_1: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    events_df.to_csv(output_dir / "delist_events.csv", index=False, encoding="utf-8-sig")
    labeled_df.to_csv(output_dir / "future_delist_labeled.csv", index=False, encoding="utf-8-sig")
    quality_report.to_csv(output_dir / "data_quality_report.csv", index=False, encoding="utf-8-sig")
    conflicts.to_csv(output_dir / "label_conflicts.csv", index=False, encoding="utf-8-sig")
    excluded_y_minus_1.to_csv(output_dir / "excluded_y_minus_1.csv", index=False, encoding="utf-8-sig")
