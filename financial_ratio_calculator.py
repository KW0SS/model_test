from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


WORKDIR = Path(r"C:\kwoss_C\model_test")
DEFAULT_INPUT_ROOT = WORKDIR / "downloads" / "dart_financials"
DEFAULT_OUTPUT_PATH = DEFAULT_INPUT_ROOT / "financial_ratios_2015_2025.csv"
TARGET_STATUS_DIRS = ("상장기업", "상폐기업")
ANNUAL_REPORT_CODE = "11011"

META_COLUMNS = [
    "기업상태",
    "기업명",
    "종목코드",
    "corp_code",
    "연도",
    "selected_fs_div",
    "has_data",
    "source_file",
]

RATIO_COLUMNS = [
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

OUTPUT_COLUMNS = META_COLUMNS + RATIO_COLUMNS + DIAGNOSTIC_COLUMNS


@dataclass(frozen=True)
class FieldSpec:
    key: str
    sections: tuple[str, ...]
    account_ids: tuple[str, ...]
    name_keywords: tuple[str, ...] = ()


@dataclass(frozen=True)
class FieldValue:
    current: int | float | None
    previous: int | float | None


FIELD_SPECS: dict[str, FieldSpec] = {
    "assets": FieldSpec(
        key="assets",
        sections=("BS",),
        account_ids=("ifrs-full_Assets", "ifrs_Assets"),
        name_keywords=("자산총계",),
    ),
    "current_assets": FieldSpec(
        key="current_assets",
        sections=("BS",),
        account_ids=("ifrs-full_CurrentAssets", "ifrs_CurrentAssets"),
        name_keywords=("유동자산",),
    ),
    "noncurrent_assets": FieldSpec(
        key="noncurrent_assets",
        sections=("BS",),
        account_ids=("ifrs-full_NoncurrentAssets", "ifrs_NoncurrentAssets"),
        name_keywords=("비유동자산",),
    ),
    "liabilities": FieldSpec(
        key="liabilities",
        sections=("BS",),
        account_ids=("ifrs-full_Liabilities", "ifrs_Liabilities"),
        name_keywords=("부채총계",),
    ),
    "equity": FieldSpec(
        key="equity",
        sections=("BS",),
        account_ids=("ifrs-full_Equity", "ifrs_Equity"),
        name_keywords=("자본총계", "기말자본"),
    ),
    "current_liabilities": FieldSpec(
        key="current_liabilities",
        sections=("BS",),
        account_ids=("ifrs-full_CurrentLiabilities", "ifrs_CurrentLiabilities"),
        name_keywords=("유동부채",),
    ),
    "revenue": FieldSpec(
        key="revenue",
        sections=("CIS", "IS"),
        account_ids=(
            "ifrs-full_Revenue",
            "ifrs_Revenue",
            "ifrs-full_RevenueFromContractsWithCustomers",
        ),
        name_keywords=("매출액", "수익"),
    ),
    "net_income": FieldSpec(
        key="net_income",
        sections=("CIS", "IS"),
        account_ids=("ifrs-full_ProfitLoss", "ifrs_ProfitLoss"),
        name_keywords=("당기순이익", "당기순손익"),
    ),
    "operating_income": FieldSpec(
        key="operating_income",
        sections=("CIS", "IS"),
        account_ids=("dart_OperatingIncomeLoss",),
        name_keywords=("영업이익", "영업손실"),
    ),
    "gross_profit": FieldSpec(
        key="gross_profit",
        sections=("CIS", "IS"),
        account_ids=("ifrs-full_GrossProfit", "ifrs_GrossProfit"),
        name_keywords=("매출총이익",),
    ),
    "cost_of_sales": FieldSpec(
        key="cost_of_sales",
        sections=("CIS", "IS"),
        account_ids=("ifrs-full_CostOfSales", "ifrs_CostOfSales", "dart_CostOfSales"),
        name_keywords=("매출원가",),
    ),
    "trade_receivables": FieldSpec(
        key="trade_receivables",
        sections=("BS",),
        account_ids=(
            "ifrs-full_CurrentTradeReceivables",
            "ifrs_CurrentTradeReceivables",
            "ifrs-full_TradeAndOtherCurrentReceivables",
            "ifrs_TradeAndOtherCurrentReceivables",
            "dart_ShortTermTradeReceivablesGross",
            "dart_ShortTermTradeAndOtherCurrentReceivablesGross",
        ),
        name_keywords=("매출채권", "매출채권및기타채권", "매출및기타채권"),
    ),
    "inventories": FieldSpec(
        key="inventories",
        sections=("BS",),
        account_ids=("ifrs-full_Inventories", "ifrs_Inventories", "ifrs-full_InventoriesTotal"),
        name_keywords=("재고자산", "유동재고자산"),
    ),
    "property_plant_equipment": FieldSpec(
        key="property_plant_equipment",
        sections=("BS",),
        account_ids=("ifrs-full_PropertyPlantAndEquipment", "ifrs_PropertyPlantAndEquipment"),
        name_keywords=("유형자산",),
    ),
    "intangible_assets": FieldSpec(
        key="intangible_assets",
        sections=("BS",),
        account_ids=(
            "ifrs-full_IntangibleAssetsOtherThanGoodwill",
            "ifrs_IntangibleAssetsOtherThanGoodwill",
            "ifrs-full_IntangibleAssetsAndGoodwill",
            "ifrs_IntangibleAssetsAndGoodwill",
        ),
        name_keywords=("무형자산",),
    ),
    "long_term_borrowings": FieldSpec(
        key="long_term_borrowings",
        sections=("BS",),
        account_ids=(
            "dart_LongTermBorrowingsGross",
            "ifrs-full_LongtermBorrowings",
            "ifrs_LongtermBorrowings",
            "ifrs-full_NoncurrentBorrowings",
            "ifrs_NoncurrentBorrowings",
            "ifrs-full_NoncurrentPortionOfNoncurrentBorrowings",
            "ifrs-full_NoncurrentPortionOfOtherNoncurrentBorrowings",
        ),
        name_keywords=("장기차입금",),
    ),
    "short_term_borrowings": FieldSpec(
        key="short_term_borrowings",
        sections=("BS",),
        account_ids=(
            "ifrs-full_ShorttermBorrowings",
            "ifrs_ShorttermBorrowings",
            "dart_ShortTermBorrowings",
            "ifrs-full_CurrentBorrowings",
            "ifrs_CurrentBorrowings",
            "ifrs-full_CurrentBorrowingsAndCurrentPortionOfNoncurrentBorrowings",
            "ifrs_CurrentPortionOfLongtermBorrowings",
            "ifrs-full_CurrentPortionOfLongtermBorrowings",
            "ifrs-full_OtherCurrentBorrowingsAndCurrentPortionOfOtherNoncurrentBorrowings",
        ),
        name_keywords=("단기차입금", "유동성장기차입금"),
    ),
    "bonds_issued": FieldSpec(
        key="bonds_issued",
        sections=("BS",),
        account_ids=(
            "dart_BondsIssued",
            "ifrs-full_BondsIssued",
            "ifrs_BondsIssued",
            "ifrs-full_CurrentBondsIssuedAndCurrentPortionOfNoncurrentBondsIssued",
            "ifrs-full_NoncurrentPortionOfNoncurrentBondsIssued",
        ),
        name_keywords=("사채", "회사채"),
    ),
    "cash_and_cash_equivalents": FieldSpec(
        key="cash_and_cash_equivalents",
        sections=("BS",),
        account_ids=("ifrs-full_CashAndCashEquivalents", "ifrs_CashAndCashEquivalents"),
        name_keywords=("현금및현금성자산", "현금예금"),
    ),
    "interest_expense": FieldSpec(
        key="interest_expense",
        sections=("CF", "CIS", "IS"),
        account_ids=(
            "dart_AdjustmentsForInterestExpenses",
            "dart_InterestExpenseFinanceExpense",
            "ifrs-full_InterestExpense",
            "ifrs_InterestExpense",
            "ifrs-full_AdjustmentsForInterestExpense",
        ),
        name_keywords=("이자비용",),
    ),
    "depreciation_expense": FieldSpec(
        key="depreciation_expense",
        sections=("CF", "CIS", "IS"),
        account_ids=(
            "dart_AdjustmentsForDepreciationExpense",
            "ifrs-full_AdjustmentsForDepreciationExpense",
            "ifrs_AdjustmentsForDepreciationExpense",
            "dart_DepreciationExpense",
            "dart_DepreciationExpenseSellingGeneralAdministrativeExpenses",
        ),
        name_keywords=("감가상각비",),
    ),
    "amortisation_expense": FieldSpec(
        key="amortisation_expense",
        sections=("CF", "CIS", "IS"),
        account_ids=(
            "dart_AdjustmentsForAmortisationExpense",
            "dart_AdjustmentsForAmortizationExpense",
            "ifrs-full_AdjustmentsForAmortisationExpense",
            "ifrs-full_AdjustmentsForAmortizationExpense",
            "ifrs_AdjustmentsForAmortisationExpense",
            "ifrs_AdjustmentsForAmortizationExpense",
            "dart_AmortisationExpense",
            "dart_AmortizationExpense",
            "dart_AmortisationExpenseSellingGeneralAdministrativeExpenses",
            "dart_AmortizationExpenseSellingGeneralAdministrativeExpenses",
        ),
        name_keywords=("무형자산상각비",),
    ),
    "retained_earnings": FieldSpec(
        key="retained_earnings",
        sections=("BS",),
        account_ids=("ifrs-full_RetainedEarnings", "ifrs_RetainedEarnings"),
        name_keywords=("이익잉여금",),
    ),
    "capital_surplus": FieldSpec(
        key="capital_surplus",
        sections=("BS",),
        account_ids=("dart_CapitalSurplus", "ifrs-full_SharePremium", "ifrs_SharePremium"),
        name_keywords=("자본잉여금", "주식발행초과금"),
    ),
    "issued_capital": FieldSpec(
        key="issued_capital",
        sections=("BS",),
        account_ids=(
            "ifrs-full_IssuedCapital",
            "ifrs_IssuedCapital",
            "ifrs-full_CapitalStock",
            "ifrs_CapitalStock",
            "dart_IssuedCapitalOfCommonStock",
        ),
        name_keywords=("자본금", "납입자본금"),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate annual financial ratios from downloaded DART JSON files."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Root directory that contains DART JSON downloads",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Destination CSV path",
    )
    parser.add_argument("--start-year", type=int, default=2015, help="First business year")
    parser.add_argument("--end-year", type=int, default=2025, help="Last business year")
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    return "".join(str(value or "").strip().split()).lower()


def parse_amount(value: Any) -> int | float | None:
    text = str(value or "").strip()
    if not text:
        return None
    text = text.replace(",", "")
    if text in {"-", "--", "N/A", "nan", "None"}:
        return None
    if text.startswith("(") and text.endswith(")"):
        text = f"-{text[1:-1]}"
    try:
        number = float(text)
    except ValueError:
        return None
    if number.is_integer():
        return int(number)
    return number


def parse_order(value: Any) -> tuple[int, str]:
    text = str(value or "").strip()
    if text.isdigit():
        return int(text), text
    return 10**9, text


def is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def nan() -> float:
    return float("nan")


def safe_div(numerator: Any, denominator: Any, multiplier: float = 1.0) -> float:
    if is_missing(numerator) or is_missing(denominator):
        return nan()
    denominator_value = float(denominator)
    if denominator_value == 0:
        return nan()
    return float(numerator) / denominator_value * multiplier


def sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (parse_order(row.get("ord")), str(row.get("account_id") or "")))


def find_matching_row(rows: list[dict[str, Any]], spec: FieldSpec) -> dict[str, Any] | None:
    normalized_keywords = tuple(normalize_text(keyword) for keyword in spec.name_keywords)
    exact_id_map = {account_id: index for index, account_id in enumerate(spec.account_ids)}

    for section in spec.sections:
        section_rows = [row for row in rows if row.get("sj_div") == section]
        if not section_rows:
            continue

        exact_matches = [
            row for row in section_rows if str(row.get("account_id") or "") in exact_id_map
        ]
        if exact_matches:
            return min(
                exact_matches,
                key=lambda row: (
                    exact_id_map[str(row.get("account_id") or "")],
                    parse_order(row.get("ord")),
                ),
            )

        for keyword in normalized_keywords:
            keyword_matches = [
                row
                for row in section_rows
                if keyword and keyword in normalize_text(row.get("account_nm"))
            ]
            if keyword_matches:
                return sort_rows(keyword_matches)[0]

    return None


def extract_fields(rows: list[dict[str, Any]]) -> dict[str, FieldValue]:
    extracted: dict[str, FieldValue] = {}
    sorted_rows = sort_rows(rows)
    for key, spec in FIELD_SPECS.items():
        row = find_matching_row(sorted_rows, spec)
        if row is None:
            extracted[key] = FieldValue(current=None, previous=None)
            continue
        extracted[key] = FieldValue(
            current=parse_amount(row.get("thstrm_amount")),
            previous=parse_amount(row.get("frmtrm_amount")),
        )
    return extracted


def empty_ratio_payload() -> dict[str, float]:
    return {column: nan() for column in RATIO_COLUMNS}


def calculate_ratios(fields: dict[str, FieldValue]) -> dict[str, float | int]:
    current = {key: value.current for key, value in fields.items()}
    previous = {key: value.previous for key, value in fields.items()}

    depreciation_total = (
        (0 if is_missing(current["depreciation_expense"]) else current["depreciation_expense"])
        + (0 if is_missing(current["amortisation_expense"]) else current["amortisation_expense"])
    )
    if is_missing(current["depreciation_expense"]) and is_missing(current["amortisation_expense"]):
        depreciation_total = nan()

    borrowings_total = sum(
        0 if is_missing(current[field]) else float(current[field])
        for field in ("long_term_borrowings", "short_term_borrowings", "bonds_issued")
    )
    if all(is_missing(current[field]) for field in ("long_term_borrowings", "short_term_borrowings", "bonds_issued")):
        borrowings_total = nan()

    retained_plus_surplus: float | int
    if is_missing(current["retained_earnings"]) and is_missing(current["capital_surplus"]):
        retained_plus_surplus = nan()
    else:
        retained_plus_surplus = (
            (0 if is_missing(current["retained_earnings"]) else current["retained_earnings"])
            + (0 if is_missing(current["capital_surplus"]) else current["capital_surplus"])
        )

    total_capital = current["assets"]

    return {
        "총자산증가율": safe_div(
            None if is_missing(current["assets"]) or is_missing(previous["assets"]) else current["assets"] - previous["assets"],
            previous["assets"],
            100.0,
        ),
        "유동자산증가율": safe_div(
            None
            if is_missing(current["current_assets"]) or is_missing(previous["current_assets"])
            else current["current_assets"] - previous["current_assets"],
            previous["current_assets"],
            100.0,
        ),
        "매출액증가율": safe_div(
            None if is_missing(current["revenue"]) or is_missing(previous["revenue"]) else current["revenue"] - previous["revenue"],
            previous["revenue"],
            100.0,
        ),
        "순이익증가율": safe_div(
            None if is_missing(current["net_income"]) or is_missing(previous["net_income"]) else current["net_income"] - previous["net_income"],
            previous["net_income"],
            100.0,
        ),
        "영업이익증가율": safe_div(
            None
            if is_missing(current["operating_income"]) or is_missing(previous["operating_income"])
            else current["operating_income"] - previous["operating_income"],
            previous["operating_income"],
            100.0,
        ),
        "매출액순이익률": safe_div(current["net_income"], current["revenue"], 100.0),
        "매출총이익률": safe_div(current["gross_profit"], current["revenue"], 100.0),
        "자기자본순이익률": safe_div(current["net_income"], current["equity"], 100.0),
        "매출채권회전율": safe_div(current["revenue"], current["trade_receivables"]),
        "재고자산회전율": safe_div(current["cost_of_sales"], current["inventories"]),
        "총자본회전율": safe_div(current["revenue"], total_capital),
        "유형자산회전율": safe_div(current["revenue"], current["property_plant_equipment"]),
        "매출원가율": safe_div(current["cost_of_sales"], current["revenue"], 100.0),
        "부채비율": safe_div(current["liabilities"], current["equity"], 100.0),
        "유동비율": safe_div(current["current_assets"], current["current_liabilities"], 100.0),
        "자기자본비율": safe_div(current["equity"], current["assets"], 100.0),
        "당좌비율": safe_div(
            None
            if is_missing(current["current_assets"]) or is_missing(current["inventories"])
            else current["current_assets"] - current["inventories"],
            current["current_liabilities"],
            100.0,
        ),
        "비유동자산장기적합률": safe_div(current["noncurrent_assets"], current["long_term_borrowings"]),
        "순운전자본비율": safe_div(
            None
            if is_missing(current["current_assets"]) or is_missing(current["current_liabilities"])
            else current["current_assets"] - current["current_liabilities"],
            total_capital,
            100.0,
        ),
        "차입금의존도": safe_div(borrowings_total, total_capital, 100.0),
        "현금비율": safe_div(current["cash_and_cash_equivalents"], current["current_liabilities"], 100.0),
        "유형자산": current["property_plant_equipment"] if not is_missing(current["property_plant_equipment"]) else nan(),
        "무형자산": current["intangible_assets"] if not is_missing(current["intangible_assets"]) else nan(),
        "무형자산상각비": current["amortisation_expense"] if not is_missing(current["amortisation_expense"]) else nan(),
        "유형자산상각비": current["depreciation_expense"] if not is_missing(current["depreciation_expense"]) else nan(),
        "감가상각비": depreciation_total,
        "총자본영업이익률": safe_div(current["operating_income"], total_capital, 100.0),
        "총자본순이익률": safe_div(current["net_income"], total_capital, 100.0),
        "유보액/납입자본비율": safe_div(retained_plus_surplus, current["issued_capital"], 100.0),
        "총자본투자효율": safe_div(
            None
            if is_missing(current["net_income"]) and is_missing(current["interest_expense"])
            else (0 if is_missing(current["net_income"]) else current["net_income"])
            + (0 if is_missing(current["interest_expense"]) else current["interest_expense"]),
            total_capital,
        ),
    }


def iter_annual_files(input_root: Path, start_year: int, end_year: int) -> list[Path]:
    files: list[Path] = []
    for status_dir in TARGET_STATUS_DIRS:
        base_dir = input_root / status_dir
        if not base_dir.exists():
            continue
        for year in range(start_year, end_year + 1):
            files.extend(base_dir.glob(f"*/{year}/*__{ANNUAL_REPORT_CODE}.json"))
    return sorted(files, key=lambda path: (path.parts[-4], path.parts[-3], path.parts[-2], path.name))


def build_base_record(payload: dict[str, Any], source_file: Path) -> dict[str, Any]:
    company = payload.get("company", {})
    classification = payload.get("classification", {})
    request = payload.get("request", {})
    result = payload.get("result", {})

    return {
        "기업상태": classification.get("status_label") or classification.get("status"),
        "기업명": company.get("corp_name"),
        "종목코드": company.get("stock_code"),
        "corp_code": company.get("corp_code"),
        "연도": request.get("bsns_year"),
        "selected_fs_div": request.get("selected_fs_div"),
        "has_data": bool(result.get("has_data")),
        "source_file": str(source_file),
    }


def build_diagnostic_values(fields: dict[str, FieldValue]) -> dict[str, Any]:
    current = {key: value.current for key, value in fields.items()}
    return {
        "총자산": current["assets"] if not is_missing(current["assets"]) else nan(),
        "유동자산": current["current_assets"] if not is_missing(current["current_assets"]) else nan(),
        "매출액": current["revenue"] if not is_missing(current["revenue"]) else nan(),
        "순이익": current["net_income"] if not is_missing(current["net_income"]) else nan(),
        "영업이익": current["operating_income"] if not is_missing(current["operating_income"]) else nan(),
        "유동부채": current["current_liabilities"] if not is_missing(current["current_liabilities"]) else nan(),
        "자기자본": current["equity"] if not is_missing(current["equity"]) else nan(),
    }


def build_record(source_file: Path) -> dict[str, Any]:
    payload = json.loads(source_file.read_text(encoding="utf-8"))
    record = build_base_record(payload, source_file.resolve())

    if not payload.get("result", {}).get("has_data") or not payload.get("financial_statements"):
        record.update(empty_ratio_payload())
        record.update({column: nan() for column in DIAGNOSTIC_COLUMNS})
        return record

    fields = extract_fields(payload["financial_statements"])
    record.update(calculate_ratios(fields))
    record.update(build_diagnostic_values(fields))
    return record


def format_output_value(value: Any) -> str | int:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        return f"{value:.6f}"
    if value is None:
        return "NaN"
    return str(value)


def write_csv(output_path: Path, records: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow({column: format_output_value(record.get(column)) for column in OUTPUT_COLUMNS})


def main() -> int:
    args = parse_args()
    if args.start_year > args.end_year:
        raise ValueError("--start-year must be less than or equal to --end-year")

    files = iter_annual_files(args.input_root, args.start_year, args.end_year)
    if not files:
        raise FileNotFoundError(
            f"No annual DART JSON files were found in {args.input_root} for {args.start_year}-{args.end_year}."
        )

    records = [build_record(path) for path in files]
    write_csv(args.output_path, records)

    print(f"Processed annual files: {len(files)}")
    print(f"Saved CSV: {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
