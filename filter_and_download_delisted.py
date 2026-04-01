from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from dart_financial_downloader import (
    CompanyRecord,
    DartQuotaExceeded,
    REPORT_CODES,
    STATUS_LABELS,
    build_file_payload,
    build_output_path,
    fetch_financial_statement,
    get_dart_api_key,
    make_session,
    write_json,
)
from delisting_shared import (
    CODE_COLUMN,
    COMPANY_COLUMN,
    CORP_CODE_COLUMN,
    DEFAULT_OUTPUT_DIR,
    EVENT_DATE_COLUMN,
    EVENT_SOURCE_COLUMN,
    EVENT_YEAR_COLUMN,
    HAS_DATA_COLUMN,
    SOURCE_FILE_COLUMN,
    YEAR_COLUMN,
    normalize_corp_code,
    normalize_stock_code,
    read_csv,
)


WORKDIR = Path(r"C:\kwoss_C\model_test")
DEFAULT_EVENTS_INPUT = WORKDIR / "2015_2025_delisted.csv"
DEFAULT_RATIOS_INPUT = WORKDIR / "downloads" / "dart_financials" / "financial_ratios_2015_2025.csv"
DEFAULT_COMPANY_MASTER = WORKDIR / "downloads" / "dart_financials" / "_meta" / "company_master.json"
DEFAULT_DOWNLOAD_ROOT = WORKDIR / "downloads" / "dart_financials"
DEFAULT_ARTIFACT_DIR = DEFAULT_OUTPUT_DIR / "filtered_delisted"
ANNUAL_REPORT_CODE = "11011"


@dataclass(frozen=True)
class ExcludeRule:
    name: str
    patterns: tuple[str, ...]


EXCLUDE_RULES: tuple[ExcludeRule, ...] = (
    ExcludeRule("merge_or_spac", ("합병", "피흡수합병", "소멸합병", "스팩", "SPAC", "기업인수목적")),
    ExcludeRule("market_transfer", ("이전상장", "유가증권시장 상장", "코스닥시장 상장")),
    ExcludeRule("privatization", ("완전자회사", "완전자회사화", "완전자회사로 편입")),
)

EXCLUDE_RULES = EXCLUDE_RULES + (
    ExcludeRule("voluntary_delisting", ("신청에 의한 상장폐지", "상장폐지 신청")),
    ExcludeRule("expiry_delisting", ("존속기간 만료",)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter delisted events and download missing annual DART statements for kept companies.",
    )
    parser.add_argument("--events-input", type=Path, default=DEFAULT_EVENTS_INPUT, help="상폐 이벤트 CSV 경로")
    parser.add_argument("--ratios-input", type=Path, default=DEFAULT_RATIOS_INPUT, help="기존 재무비율 CSV 경로")
    parser.add_argument("--company-master", type=Path, default=DEFAULT_COMPANY_MASTER, help="company_master.json 경로")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR, help="필터링 결과 저장 디렉터리")
    parser.add_argument("--download-root", type=Path, default=DEFAULT_DOWNLOAD_ROOT, help="DART JSON 저장 루트")
    parser.add_argument("--start-year", type=int, default=2015, help="다운로드 시작 사업연도")
    parser.add_argument("--end-year", type=int, default=2025, help="다운로드 종료 사업연도")
    parser.add_argument("--sleep-seconds", type=float, default=0.05, help="요청 사이 sleep")
    parser.add_argument("--request-timeout", type=int, default=60, help="HTTP timeout")
    parser.add_argument("--retries", type=int, default=3, help="재시도 횟수")
    parser.add_argument("--overwrite", action="store_true", help="기존 JSON을 덮어씀")
    parser.add_argument("--skip-download", action="store_true", help="필터링/목록 생성만 수행")
    parser.add_argument("--skip-ratio-refresh", action="store_true", help="다운로드 후 재무비율 CSV 갱신 생략")
    return parser.parse_args()


def read_events_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"기업명", "종목코드", "상폐일", "폐지사유"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"상폐 CSV에 필요한 컬럼이 없습니다: {missing}")
    df = df.copy()
    df["종목코드"] = normalize_stock_code(df["종목코드"])
    df["상폐일"] = df["상폐일"].astype("string").str.strip()
    df["폐지사유"] = df["폐지사유"].astype("string").fillna("").str.strip()
    df["event_year_Y"] = pd.to_datetime(df["상폐일"], errors="coerce").dt.year.astype("Int64")
    return df


def read_company_master(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    df = pd.DataFrame(payload)
    df = df.copy()
    df["stock_code"] = normalize_stock_code(df["stock_code"])
    df["corp_code"] = normalize_corp_code(df["corp_code"])
    return df


def assign_exclude_rule(reason: str) -> str:
    text = str(reason or "")
    for rule in EXCLUDE_RULES:
        if any(pattern in text for pattern in rule.patterns):
            return rule.name
    return ""


def build_filtered_frames(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = events.copy()
    working["exclude_rule"] = working["폐지사유"].map(assign_exclude_rule)
    excluded = working[working["exclude_rule"].ne("")].copy()
    kept = working[working["exclude_rule"].eq("")].copy()
    return kept.reset_index(drop=True), excluded.reset_index(drop=True)


def build_missing_manifest(
    kept_events: pd.DataFrame,
    ratios: pd.DataFrame,
    company_master: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    ratio_code_col = ratios.columns[2]
    ratios = ratios.copy()
    ratios[ratio_code_col] = normalize_stock_code(ratios[ratio_code_col])
    ratio_codes = set(ratios[ratio_code_col].dropna().astype(str))

    kept = kept_events.copy()
    kept["financial_data_exists"] = kept["종목코드"].isin(ratio_codes)
    missing = kept[~kept["financial_data_exists"]].copy()

    master_columns = ["stock_code", "corp_code", "corp_name", "modify_date", "status"]
    master_lookup = company_master[master_columns].drop_duplicates(subset=["stock_code"], keep="first")
    missing = missing.merge(master_lookup, left_on="종목코드", right_on="stock_code", how="left")
    missing["corp_code"] = normalize_corp_code(missing["corp_code"])
    missing["resolved_corp_code"] = missing["corp_code"].astype("string").fillna("").ne("00000000")
    missing["download_end_year"] = missing["event_year_Y"].astype("Int64") - 1
    missing["download_start_year"] = pd.Series([start_year] * len(missing), dtype="Int64")
    missing["download_end_year"] = missing["download_end_year"].clip(lower=start_year, upper=end_year)
    missing["download_year_window_valid"] = missing["download_end_year"] >= missing["download_start_year"]
    missing = missing.rename(
        columns={
            "corp_name": "master_corp_name",
            "modify_date": "master_modify_date",
            "status": "master_status",
        }
    )
    return missing.reset_index(drop=True)


def save_filter_artifacts(
    artifact_dir: Path,
    kept_events: pd.DataFrame,
    excluded_events: pd.DataFrame,
    missing_manifest: pd.DataFrame,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    kept_events.to_csv(artifact_dir / "filtered_delisted_kept.csv", index=False, encoding="utf-8-sig")
    excluded_events.to_csv(artifact_dir / "filtered_delisted_excluded.csv", index=False, encoding="utf-8-sig")
    missing_manifest.to_csv(artifact_dir / "missing_financial_download_manifest.csv", index=False, encoding="utf-8-sig")


def to_company_record(row: pd.Series) -> CompanyRecord:
    return CompanyRecord(
        corp_code=str(row["corp_code"]),
        corp_name=str(row.get("기업명") or row.get("master_corp_name") or ""),
        corp_eng_name="",
        stock_code=str(row["종목코드"]),
        modify_date=str(row.get("상폐일") or ""),
        status="delisted",
        status_label=STATUS_LABELS["delisted"],
        source="filtered_delisted_csv_missing_financials",
    )


def download_missing_annuals(manifest: pd.DataFrame, args: argparse.Namespace) -> dict[str, Any]:
    targets = manifest[
        manifest["resolved_corp_code"] & manifest["download_year_window_valid"]
    ].copy()

    api_key = get_dart_api_key()
    session = make_session()
    summary: dict[str, Any] = {
        "target_companies": int(len(targets)),
        "saved_files": 0,
        "skipped_existing_files": 0,
        "no_data_files": 0,
        "failed_requests": 0,
        "quota_exceeded": False,
    }

    for index, (_, row) in enumerate(targets.iterrows(), start=1):
        company = to_company_record(row)
        start_year = int(row["download_start_year"])
        end_year = int(row["download_end_year"])
        print(f"[{index}/{len(targets)}] {company.corp_name} ({company.stock_code}) {start_year}-{end_year}")

        for year in range(start_year, end_year + 1):
            output_path = build_output_path(args.download_root.resolve(), company, year, ANNUAL_REPORT_CODE)
            if output_path.exists() and not args.overwrite:
                summary["skipped_existing_files"] += 1
                continue

            try:
                fetch_result = fetch_financial_statement(
                    session=session,
                    api_key=api_key,
                    company=company,
                    year=year,
                    report_code=ANNUAL_REPORT_CODE,
                    timeout=args.request_timeout,
                    retries=args.retries,
                )
            except DartQuotaExceeded:
                summary["quota_exceeded"] = True
                return summary
            except Exception:
                summary["failed_requests"] += 1
                continue

            payload = build_file_payload(company, year, ANNUAL_REPORT_CODE, fetch_result)
            write_json(output_path, payload)
            summary["saved_files"] += 1
            if payload["result"]["no_data"]:
                summary["no_data_files"] += 1

            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

    return summary


def refresh_ratios(args: argparse.Namespace) -> None:
    from financial_ratio_calculator import main as ratio_main
    import sys

    original_argv = sys.argv[:]
    try:
        sys.argv = [
            "financial_ratio_calculator.py",
            "--input-root",
            str(args.download_root.resolve()),
            "--output-path",
            str(args.ratios_input.resolve()),
            "--start-year",
            str(args.start_year),
            "--end-year",
            str(args.end_year),
        ]
        result = ratio_main()
        if result != 0:
            raise RuntimeError(f"financial_ratio_calculator.py exited with {result}")
    finally:
        sys.argv = original_argv


def main() -> int:
    args = parse_args()
    if args.start_year > args.end_year:
        raise ValueError("--start-year must be less than or equal to --end-year")

    events = read_events_csv(args.events_input.resolve())
    ratios = read_csv(args.ratios_input.resolve())
    company_master = read_company_master(args.company_master.resolve())

    kept_events, excluded_events = build_filtered_frames(events)
    missing_manifest = build_missing_manifest(kept_events, ratios, company_master, args.start_year, args.end_year)
    save_filter_artifacts(args.artifact_dir.resolve(), kept_events, excluded_events, missing_manifest)

    summary: dict[str, Any] = {
        "input_rows": int(len(events)),
        "excluded_rows": int(len(excluded_events)),
        "kept_rows": int(len(kept_events)),
        "kept_unique_codes": int(kept_events["종목코드"].nunique()),
        "missing_financial_codes": int(missing_manifest["종목코드"].nunique()),
        "resolved_missing_codes": int(missing_manifest["resolved_corp_code"].sum()),
        "downloadable_missing_codes": int(
            (missing_manifest["resolved_corp_code"] & missing_manifest["download_year_window_valid"]).sum()
        ),
    }

    if not args.skip_download:
        download_summary = download_missing_annuals(missing_manifest, args)
        summary.update({f"download_{key}": value for key, value in download_summary.items()})
        if not args.skip_ratio_refresh and not download_summary.get("quota_exceeded"):
            refresh_ratios(args)
            summary["ratio_refresh_done"] = True
        else:
            summary["ratio_refresh_done"] = False
    else:
        summary["ratio_refresh_done"] = False

    write_json(args.artifact_dir.resolve() / "filter_download_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
