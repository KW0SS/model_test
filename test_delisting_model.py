from __future__ import annotations

import unittest

import pandas as pd

from delisting_data import build_labeled_dataset, deduplicate_company_year
from delisting_shared import (
    CODE_COLUMN,
    COMPANY_COLUMN,
    CORP_CODE_COLUMN,
    DIAGNOSTIC_COLUMNS,
    EVENT_SOURCE_COLUMN,
    EVENT_YEAR_COLUMN,
    EXCLUDE_REASON_COLUMN,
    FEATURE_COLUMNS,
    FS_COLUMN,
    HAS_DATA_COLUMN,
    INCLUDE_COLUMN,
    SOURCE_FILE_COLUMN,
    STATUS_COLUMN,
    TARGET_COLUMN,
    YEAR_COLUMN,
    Y_MINUS_1_EXCLUDED_COLUMN,
)


def build_base_row(
    *,
    status: str,
    company: str,
    code: str,
    corp_code: str,
    year: int,
    has_data: str = "True",
    source_file: str = "dummy.json",
    feature_value: float = 1.0,
) -> dict[str, object]:
    row = {
        STATUS_COLUMN: status,
        COMPANY_COLUMN: company,
        CODE_COLUMN: code,
        CORP_CODE_COLUMN: corp_code,
        YEAR_COLUMN: year,
        FS_COLUMN: "CFS",
        HAS_DATA_COLUMN: has_data,
        SOURCE_FILE_COLUMN: source_file,
    }
    for column in FEATURE_COLUMNS:
        row[column] = feature_value
    for column in DIAGNOSTIC_COLUMNS:
        row[column] = 10.0
    return row


class DelistingModelTests(unittest.TestCase):
    def test_deduplicate_prefers_listed_after_has_data_and_cfs(self) -> None:
        rows = [
            build_base_row(
                status="상폐기업",
                company="테스트",
                code="000001",
                corp_code="00000001",
                year=2020,
                has_data="False",
                source_file="a.json",
            ),
            build_base_row(
                status="상폐기업",
                company="테스트",
                code="000001",
                corp_code="00000001",
                year=2020,
                has_data="True",
                source_file="b.json",
            ),
            build_base_row(
                status="상장기업",
                company="테스트",
                code="000001",
                corp_code="00000001",
                year=2020,
                has_data="True",
                source_file="c.json",
            ),
        ]
        frame = pd.DataFrame(rows)
        frame["__has_data_bool"] = frame[HAS_DATA_COLUMN].eq("True")
        frame["__fs_priority"] = 1
        frame["__status_priority"] = frame[STATUS_COLUMN].map({"상장기업": 2, "상폐기업": 1})

        deduped = deduplicate_company_year(frame)

        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped.iloc[0][STATUS_COLUMN], "상장기업")
        self.assertEqual(deduped.iloc[0][SOURCE_FILE_COLUMN], "c.json")

    def test_future_delist_labeling_uses_y_minus_2_and_excludes_y_minus_1(self) -> None:
        rows = [
            build_base_row(status="상장기업", company="A", code="000001", corp_code="00000001", year=2020),
            build_base_row(status="상장기업", company="A", code="000001", corp_code="00000001", year=2021),
            build_base_row(status="상장기업", company="A", code="000001", corp_code="00000001", year=2022),
            build_base_row(status="상장기업", company="A", code="000001", corp_code="00000001", year=2023),
            build_base_row(status="상장기업", company="B", code="000002", corp_code="00000002", year=2020),
        ]
        events = pd.DataFrame(
            [
                {
                    CODE_COLUMN: "000001",
                    CORP_CODE_COLUMN: "00000001",
                    COMPANY_COLUMN: "A",
                    EVENT_YEAR_COLUMN: 2022,
                    "event_date": "20220331",
                    EVENT_SOURCE_COLUMN: "unit-test",
                    Y_MINUS_1_EXCLUDED_COLUMN: True,
                }
            ]
        )

        labeled, _, excluded_y_minus_1 = build_labeled_dataset(pd.DataFrame(rows), events)
        row_2020 = labeled[(labeled[CODE_COLUMN] == "000001") & (labeled[YEAR_COLUMN] == 2020)].iloc[0]
        row_2021 = labeled[(labeled[CODE_COLUMN] == "000001") & (labeled[YEAR_COLUMN] == 2021)].iloc[0]
        row_2022 = labeled[(labeled[CODE_COLUMN] == "000001") & (labeled[YEAR_COLUMN] == 2022)].iloc[0]
        row_b = labeled[(labeled[CODE_COLUMN] == "000002") & (labeled[YEAR_COLUMN] == 2020)].iloc[0]

        self.assertEqual(int(row_2020[TARGET_COLUMN]), 1)
        self.assertTrue(bool(row_2020[INCLUDE_COLUMN]))
        self.assertTrue(pd.isna(row_2021[TARGET_COLUMN]))
        self.assertEqual(row_2021[EXCLUDE_REASON_COLUMN], "excluded_y_minus_1")
        self.assertTrue(pd.isna(row_2022[TARGET_COLUMN]))
        self.assertEqual(row_2022[EXCLUDE_REASON_COLUMN], "event_year_or_later")
        self.assertEqual(int(row_b[TARGET_COLUMN]), 0)
        self.assertTrue(bool(row_b[INCLUDE_COLUMN]))
        self.assertEqual(len(excluded_y_minus_1), 1)

    def test_deduplicate_prefers_more_complete_row_when_priority_matches(self) -> None:
        rows = [
            build_base_row(status="?곸옣湲곗뾽", company="A", code="000010", corp_code="00000010", year=2020, source_file="a.json"),
            build_base_row(status="?곸옣湲곗뾽", company="A", code="000010", corp_code="00000010", year=2020, source_file="b.json"),
        ]
        frame = pd.DataFrame(rows)
        frame.loc[0, FEATURE_COLUMNS[:3]] = pd.NA
        frame["__has_data_bool"] = True
        frame["__fs_priority"] = 1
        frame["__status_priority"] = 2
        frame["__feature_missing_count"] = frame[FEATURE_COLUMNS].isna().sum(axis=1)

        deduped = deduplicate_company_year(frame)

        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped.iloc[0][SOURCE_FILE_COLUMN], "b.json")

    def test_future_delist_labeling_excludes_status_conflict_and_sparse_rows(self) -> None:
        rows = [
            build_base_row(status="?곸옣湲곗뾽", company="A", code="000001", corp_code="00000001", year=2020, source_file="a.json"),
            build_base_row(status="?곹룓湲곗뾽", company="A", code="000001", corp_code="00000001", year=2020, source_file="b.json"),
            build_base_row(status="?곸옣湲곗뾽", company="B", code="000002", corp_code="00000002", year=2020),
        ]
        for column in FEATURE_COLUMNS[:12]:
            rows[-1][column] = pd.NA

        events = pd.DataFrame(
            [
                {
                    CODE_COLUMN: "000001",
                    CORP_CODE_COLUMN: "00000001",
                    COMPANY_COLUMN: "A",
                    EVENT_YEAR_COLUMN: 2022,
                    "event_date": "20220331",
                    EVENT_SOURCE_COLUMN: "unit-test",
                    Y_MINUS_1_EXCLUDED_COLUMN: True,
                }
            ]
        )

        labeled, _, _ = build_labeled_dataset(pd.DataFrame(rows), events)
        conflict_row = labeled[(labeled[CODE_COLUMN] == "000001") & (labeled[YEAR_COLUMN] == 2020)].iloc[0]
        sparse_row = labeled[(labeled[CODE_COLUMN] == "000002") & (labeled[YEAR_COLUMN] == 2020)].iloc[0]

        self.assertEqual(conflict_row[EXCLUDE_REASON_COLUMN], "status_conflict_duplicate")
        self.assertFalse(bool(conflict_row[INCLUDE_COLUMN]))
        self.assertEqual(sparse_row[EXCLUDE_REASON_COLUMN], "too_many_missing_features")
        self.assertFalse(bool(sparse_row[INCLUDE_COLUMN]))


if __name__ == "__main__":
    unittest.main()
