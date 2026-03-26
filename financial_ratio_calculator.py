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

RATIO_LABELS = {
    "asset_growth_rate": "총자산증가율",
    "current_asset_growth_rate": "유동자산증가율",
    "revenue_growth_rate": "매출액증가율",
    "net_income_growth_rate": "순이익증가율",
    "operating_income_growth_rate": "영업이익증가율",
    "net_margin": "매출액순이익률",
    "gross_margin": "매출총이익률",
    "roe": "자기자본순이익률",
    "receivables_turnover": "매출채권회전율",
    "inventory_turnover": "재고자산회전율",
    "total_capital_turnover": "총자본회전율",
    "fixed_asset_turnover": "유형자산회전율",
    "cost_of_sales_ratio": "매출원가율",
    "debt_ratio": "부채비율",
    "current_ratio": "유동비율",
    "equity_ratio": "자기자본비율",
    "quick_ratio": "당좌비율",
    "noncurrent_assets_to_long_term_borrowings": "비유동자산장기적합률",
    "net_working_capital_ratio": "순운전자본비율",
    "borrowings_dependency": "차입금의존도",
    "cash_ratio": "현금비율",
    "property_plant_equipment_value": "유형자산",
    "intangible_assets_value": "무형자산",
    "amortisation_expense_value": "무형자산상각비",
    "depreciation_expense_value": "유형자산상각비",
    "total_depreciation_and_amortisation": "감가상각비",
    "operating_income_on_total_capital": "총자본영업이익률",
    "net_income_on_total_capital": "총자본순이익률",
    "retained_earnings_to_paid_in_capital": "유보액/납입자본비율",
    "total_capital_investment_efficiency": "총자본투자효율",
}

DIAGNOSTIC_LABELS = {
    "assets": "총자산",
    "current_assets": "유동자산",
    "revenue": "매출액",
    "net_income": "순이익",
    "operating_income": "영업이익",
    "current_liabilities": "유동부채",
    "equity": "자기자본",
}


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
        description="Calculate annual financial ratios from downloaded DART JSON files.",
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
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Ignore existing CSV rows and rebuild the output from source JSON files.",
    )
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


def parse_year(value: Any) -> int:
    return int(str(value).strip())


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


def diff_or_none(current: Any, previous: Any) -> float | int | None:
    if is_missing(current) or is_missing(previous):
        return None
    return float(current) - float(previous)


def sum_or_nan(*values: Any) -> float:
    present = [float(value) for value in values if not is_missing(value)]
    if not present:
        return nan()
    return sum(present)


def sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (parse_order(row.get("ord")), str(row.get("account_id") or "")))


def find_matching_row(rows: list[dict[str, Any]], spec: FieldSpec) -> dict[str, Any] | None:
    normalized_keywords = tuple(normalize_text(keyword) for keyword in spec.name_keywords)
    exact_id_map = {account_id: index for index, account_id in enumerate(spec.account_ids)}

    for section in spec.sections:
        section_rows = [row for row in rows if row.get("sj_div") == section]
        if not section_rows:
            continue

        exact_matches = [row for row in section_rows if str(row.get("account_id") or "") in exact_id_map]
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
    return {label: nan() for label in RATIO_COLUMNS}


def empty_diagnostic_payload() -> dict[str, float]:
    return {label: nan() for label in DIAGNOSTIC_COLUMNS}


def calculate_ratios(fields: dict[str, FieldValue]) -> dict[str, float]:
    current = {key: value.current for key, value in fields.items()}
    previous = {key: value.previous for key, value in fields.items()}

    total_capital = current["assets"]
    depreciation_total = sum_or_nan(current["depreciation_expense"], current["amortisation_expense"])
    borrowings_total = sum_or_nan(
        current["long_term_borrowings"],
        current["short_term_borrowings"],
        current["bonds_issued"],
    )
    retained_plus_surplus = sum_or_nan(current["retained_earnings"], current["capital_surplus"])

    ratios_by_key = {
        "asset_growth_rate": safe_div(diff_or_none(current["assets"], previous["assets"]), previous["assets"], 100.0),
        "current_asset_growth_rate": safe_div(
            diff_or_none(current["current_assets"], previous["current_assets"]),
            previous["current_assets"],
            100.0,
        ),
        "revenue_growth_rate": safe_div(
            diff_or_none(current["revenue"], previous["revenue"]),
            previous["revenue"],
            100.0,
        ),
        "net_income_growth_rate": safe_div(
            diff_or_none(current["net_income"], previous["net_income"]),
            previous["net_income"],
            100.0,
        ),
        "operating_income_growth_rate": safe_div(
            diff_or_none(current["operating_income"], previous["operating_income"]),
            previous["operating_income"],
            100.0,
        ),
        "net_margin": safe_div(current["net_income"], current["revenue"], 100.0),
        "gross_margin": safe_div(current["gross_profit"], current["revenue"], 100.0),
        "roe": safe_div(current["net_income"], current["equity"], 100.0),
        "receivables_turnover": safe_div(current["revenue"], current["trade_receivables"]),
        "inventory_turnover": safe_div(current["cost_of_sales"], current["inventories"]),
        "total_capital_turnover": safe_div(current["revenue"], total_capital),
        "fixed_asset_turnover": safe_div(current["revenue"], current["property_plant_equipment"]),
        "cost_of_sales_ratio": safe_div(current["cost_of_sales"], current["revenue"], 100.0),
        "debt_ratio": safe_div(current["liabilities"], current["equity"], 100.0),
        "current_ratio": safe_div(current["current_assets"], current["current_liabilities"], 100.0),
        "equity_ratio": safe_div(current["equity"], current["assets"], 100.0),
        "quick_ratio": safe_div(
            diff_or_none(current["current_assets"], current["inventories"]),
            current["current_liabilities"],
            100.0,
        ),
        "noncurrent_assets_to_long_term_borrowings": safe_div(
            current["noncurrent_assets"],
            current["long_term_borrowings"],
        ),
        "net_working_capital_ratio": safe_div(
            diff_or_none(current["current_assets"], current["current_liabilities"]),
            total_capital,
            100.0,
        ),
        "borrowings_dependency": safe_div(borrowings_total, total_capital, 100.0),
        "cash_ratio": safe_div(current["cash_and_cash_equivalents"], current["current_liabilities"], 100.0),
        "property_plant_equipment_value": (
            float(current["property_plant_equipment"])
            if not is_missing(current["property_plant_equipment"])
            else nan()
        ),
        "intangible_assets_value": (
            float(current["intangible_assets"]) if not is_missing(current["intangible_assets"]) else nan()
        ),
        "amortisation_expense_value": (
            float(current["amortisation_expense"]) if not is_missing(current["amortisation_expense"]) else nan()
        ),
        "depreciation_expense_value": (
            float(current["depreciation_expense"]) if not is_missing(current["depreciation_expense"]) else nan()
        ),
        "total_depreciation_and_amortisation": depreciation_total,
        "operating_income_on_total_capital": safe_div(current["operating_income"], total_capital, 100.0),
        "net_income_on_total_capital": safe_div(current["net_income"], total_capital, 100.0),
        "retained_earnings_to_paid_in_capital": safe_div(retained_plus_surplus, current["issued_capital"], 100.0),
        "total_capital_investment_efficiency": safe_div(
            sum_or_nan(current["net_income"], current["interest_expense"]),
            total_capital,
        ),
    }
    return {RATIO_LABELS[key]: value for key, value in ratios_by_key.items()}


def build_diagnostic_values(fields: dict[str, FieldValue]) -> dict[str, float]:
    current = {key: value.current for key, value in fields.items()}
    diagnostics_by_key = {
        "assets": current["assets"],
        "current_assets": current["current_assets"],
        "revenue": current["revenue"],
        "net_income": current["net_income"],
        "operating_income": current["operating_income"],
        "current_liabilities": current["current_liabilities"],
        "equity": current["equity"],
    }
    return {
        DIAGNOSTIC_LABELS[key]: float(value) if not is_missing(value) else nan()
        for key, value in diagnostics_by_key.items()
    }


def annual_file_sort_key(path: Path, input_root: Path) -> tuple[str, str, int, str]:
    relative = path.relative_to(input_root)
    status, company, year, filename = relative.parts[-4:]
    return status, company, parse_year(year), filename


def iter_annual_files(input_root: Path, start_year: int, end_year: int) -> list[Path]:
    files: list[Path] = []
    for status_dir_name in TARGET_STATUS_DIRS:
        base_dir = input_root / status_dir_name
        if not base_dir.exists():
            continue
        for year in range(start_year, end_year + 1):
            files.extend(base_dir.glob(f"*/{year}/*__{ANNUAL_REPORT_CODE}.json"))
    return sorted(files, key=lambda path: annual_file_sort_key(path, input_root))


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
        "source_file": str(source_file.resolve()),
    }


def build_record(source_file: Path) -> dict[str, Any]:
    payload = json.loads(source_file.read_text(encoding="utf-8"))
    record = build_base_record(payload, source_file)
    rows = payload.get("financial_statements") or payload.get("list") or []
    if not payload.get("result", {}).get("has_data") or not rows:
        record.update(empty_ratio_payload())
        record.update(empty_diagnostic_payload())
        return record

    fields = extract_fields(rows)
    record.update(calculate_ratios(fields))
    record.update(build_diagnostic_values(fields))
    return record


def record_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(record.get("기업상태") or ""),
        str(record.get("corp_code") or ""),
        str(record.get("연도") or ""),
    )


def load_existing_records(output_path: Path) -> tuple[dict[tuple[str, str, str], dict[str, Any]], int]:
    if not output_path.exists():
        return {}, 0

    records: dict[tuple[str, str, str], dict[str, Any]] = {}
    duplicate_count = 0
    with output_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            normalized_row = {column: row.get(column, "") for column in OUTPUT_COLUMNS}
            key = record_key(normalized_row)
            if key in records:
                duplicate_count += 1
            records[key] = normalized_row
    return records, duplicate_count


def merge_records(
    existing_records: dict[tuple[str, str, str], dict[str, Any]],
    source_files: list[Path],
) -> tuple[list[dict[str, Any]], int]:
    merged = dict(existing_records)
    added_count = 0
    for source_file in source_files:
        record = build_record(source_file)
        key = record_key(record)
        if key in merged:
            continue
        merged[key] = record
        added_count += 1
    records = sorted(
        merged.values(),
        key=lambda record: (
            str(record.get("기업상태") or ""),
            str(record.get("기업명") or ""),
            parse_year(record.get("연도") or 0),
            str(record.get("corp_code") or ""),
        ),
    )
    return records, added_count


def format_output_value(value: Any) -> str | int:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        return f"{value:.6f}"
    if value is None or value == "":
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

    source_files = iter_annual_files(args.input_root, args.start_year, args.end_year)
    if not source_files:
        raise FileNotFoundError(
            f"No annual DART JSON files were found in {args.input_root} for {args.start_year}-{args.end_year}."
        )

    existing_records: dict[tuple[str, str, str], dict[str, Any]] = {}
    duplicate_count = 0
    if not args.rebuild:
        existing_records, duplicate_count = load_existing_records(args.output_path)

    records, added_count = merge_records(existing_records, source_files)
    write_csv(args.output_path, records)

    print(f"Processed annual files: {len(source_files)}")
    print(f"Existing records loaded: {len(existing_records)}")
    print(f"Duplicate existing keys skipped: {duplicate_count}")
    print(f"New records added: {added_count}")
    print(f"Saved CSV: {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
