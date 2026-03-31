from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


WORKDIR = Path(r"C:\kwoss_C\model_test")
DEFAULT_TRAIN_INPUT = WORKDIR / "downloads" / "dart_financials" / "financial_ratios_2015_2025.csv"
DEFAULT_PREDICT_INPUT = WORKDIR / "downloads" / "dart_financials" / "financial_ratios_2024_test.csv"
DEFAULT_COMPANY_MASTER = WORKDIR / "downloads" / "dart_financials" / "_meta" / "company_master.json"
DEFAULT_OUTPUT_DIR = WORKDIR / "artifacts" / "delisting"

STATUS_COLUMN = "기업상태"
COMPANY_COLUMN = "기업명"
CODE_COLUMN = "종목코드"
CORP_CODE_COLUMN = "corp_code"
YEAR_COLUMN = "연도"
FS_COLUMN = "selected_fs_div"
HAS_DATA_COLUMN = "has_data"
SOURCE_FILE_COLUMN = "source_file"

EVENT_YEAR_COLUMN = "event_year_Y"
EVENT_DATE_COLUMN = "event_date"
EVENT_SOURCE_COLUMN = "event_source"
Y_MINUS_1_EXCLUDED_COLUMN = "y_minus_1_excluded"
TARGET_COLUMN = "target_future_delist"
INCLUDE_COLUMN = "include_for_training"
EXCLUDE_REASON_COLUMN = "exclude_reason"

PREDICTION_PROBABILITY_COLUMN = "상폐확률"
PREDICTION_LABEL_COLUMN = "예측라벨"
PREDICTION_MODEL_COLUMN = "사용모델"
PREDICTION_ELIGIBLE_COLUMN = "예측가능여부"
PREDICTION_SKIP_REASON_COLUMN = "제외사유"

LISTED_STATUS = "상장기업"
DELISTED_STATUS = "상폐기업"
UNLISTED_STATUS = "비상장기업"
PREDICTION_LABELS = {0: "정상", 1: "미래상폐위험"}

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

STATUS_PRIORITY = {
    LISTED_STATUS: 2,
    DELISTED_STATUS: 1,
    UNLISTED_STATUS: 0,
}

DEFAULT_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def read_company_master(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"company_master.json 파일을 찾을 수 없습니다: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("company_master.json 형식이 예상과 다릅니다.")
    return pd.DataFrame(payload)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_thresholds(value: str) -> list[float]:
    tokens = [token.strip() for token in str(value).split(",") if token.strip()]
    if not tokens:
        return DEFAULT_THRESHOLDS
    thresholds = sorted({float(token) for token in tokens})
    for threshold in thresholds:
        if threshold <= 0 or threshold >= 1:
            raise ValueError(f"threshold는 0과 1 사이여야 합니다: {threshold}")
    return thresholds


def normalize_stock_code(series: pd.Series) -> pd.Series:
    text = series.astype("string").fillna("").str.strip()
    text = text.str.replace(r"\.0$", "", regex=True)
    return text.str.zfill(6)


def normalize_corp_code(series: pd.Series) -> pd.Series:
    text = series.astype("string").fillna("").str.strip()
    text = text.str.replace(r"\.0$", "", regex=True)
    return text.str.zfill(8)


def parse_bool_series(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.isin({"true", "1", "y", "yes"})


def ensure_columns(df: pd.DataFrame) -> None:
    missing = [column for column in ALL_REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {missing}")
