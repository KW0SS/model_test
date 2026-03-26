from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import pandas as pd
import requests


WORKDIR = Path(r"C:\kwoss_C\model_test")
ENV_PATH = WORKDIR / ".env"
OUTPUT_ROOT = WORKDIR / "downloads" / "dart_financials"
DART_BASE_URL = "https://opendart.fss.or.kr/api"
KRX_LISTED_URL = "https://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13"
KRX_DELISTED_URL = "https://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=05"
DEFAULT_DELISTED_CSV = Path(
    r"C:\Users\minsu\OneDrive - 광운대학교\광운대 4학년\졸업작품\데이터\delisted_stock_code.csv"
)
REPORT_CODES: dict[str, str] = {
    "11013": "1Q",
    "11012": "HALF",
    "11014": "3Q",
    "11011": "ANNUAL",
}
DEFAULT_TIMEOUT = 60
NO_DATA_STATUSES = {"013", "014", "015", "018"}
DAILY_LIMIT_STATUS = "020"
STATUS_LABELS = {
    "listed": "\uc0c1\uc7a5\uae30\uc5c5",
    "delisted": "\uc0c1\ud3d0\uae30\uc5c5",
    "unlisted": "\ube44\uc0c1\uc7a5\uae30\uc5c5",
}
DOWNLOADABLE_STATUSES = {"listed", "delisted"}


class DartQuotaExceeded(RuntimeError):
    """Raised when the Open DART API daily request limit is exceeded."""


@dataclass(frozen=True)
class CompanyRecord:
    corp_code: str
    corp_name: str
    corp_eng_name: str
    stock_code: str
    modify_date: str
    status: str
    status_label: str
    source: str


def load_dotenv_file(env_path: Path) -> dict[str, str]:
    if not env_path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def default_delisted_csv_path() -> Path | None:
    return DEFAULT_DELISTED_CSV if DEFAULT_DELISTED_CSV.exists() else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download DART financial statements as JSON files."
    )
    parser.add_argument("--start-year", type=int, default=2015, help="First business year")
    parser.add_argument("--end-year", type=int, default=2025, help="Last business year")
    parser.add_argument(
        "--status",
        choices=["listed", "delisted", "unlisted", "all"],
        default="all",
        help="Filter companies by current listing status",
    )
    parser.add_argument(
        "--corp-code",
        action="append",
        default=[],
        help="Limit collection to one or more DART corp_code values",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N companies")
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.1,
        help="Sleep duration between API requests",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing JSON output files",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retry count for transient network/API errors",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="HTTP request timeout in seconds",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Root directory for JSON output",
    )
    parser.add_argument(
        "--delisted-stock-code-csv",
        type=Path,
        default=default_delisted_csv_path(),
        help="Optional CSV path with a stock_code column for delisted companies",
    )
    return parser.parse_args()


def ensure_runtime_directories(output_root: Path) -> tuple[Path, Path]:
    meta_dir = output_root / "_meta"
    log_dir = output_root / "_logs"
    output_root.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    return meta_dir, log_dir


def get_dart_api_key() -> str:
    dotenv_values = load_dotenv_file(ENV_PATH)
    api_key = dotenv_values.get("DART_API_KEY", "").strip()
    if not api_key:
        raise ValueError("DART_API_KEY is required in .env")
    return api_key


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "codex-dart-financial-downloader/1.2",
            "Accept": "application/json, text/html, application/xml;q=0.9, */*;q=0.8",
        }
    )
    return session


def normalize_stock_code(value: Any) -> str:
    code = str(value or "").strip().upper()
    if not code or code == "NAN":
        return ""
    if code.isdigit():
        return code.zfill(6)
    return code


def sanitize_filename_part(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value.strip())
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned[:120] or "unknown"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, ensure_ascii=False, indent=2)
    last_error: PermissionError | None = None
    for attempt in range(5):
        temp_path = path.parent / f"{path.name}.{os.getpid()}.{attempt}.tmp"
        try:
            temp_path.write_text(body, encoding="utf-8")
            os.replace(temp_path, path)
            return
        except PermissionError as exc:
            last_error = exc
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
            time.sleep(0.2 * (attempt + 1))
        except OSError:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    assert last_error is not None
    raise last_error


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def fetch_json(
    session: requests.Session,
    url: str,
    params: dict[str, Any],
    timeout: int,
    retries: int,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            status = str(payload.get("status", "")).strip()
            if status == DAILY_LIMIT_STATUS:
                raise DartQuotaExceeded(payload.get("message", "DART request limit exceeded"))
            return payload
        except DartQuotaExceeded:
            raise
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(min(2**attempt, 10))

    assert last_error is not None
    raise RuntimeError(f"Request failed for {url}: {last_error}") from last_error


def fetch_dart_corp_codes(
    session: requests.Session,
    api_key: str,
    timeout: int,
    retries: int,
) -> list[dict[str, str]]:
    url = f"{DART_BASE_URL}/corpCode.xml"
    params = {"crtfc_key": api_key}
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                name = archive.namelist()[0]
                records: list[dict[str, str]] = []
                with archive.open(name) as xml_file:
                    for _, item in ElementTree.iterparse(xml_file, events=("end",)):
                        if item.tag != "list":
                            continue
                        records.append(
                            {child.tag: (child.text or "").strip() for child in item}
                        )
                        item.clear()
            return records
        except (requests.RequestException, zipfile.BadZipFile, ElementTree.ParseError) as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(min(2**attempt, 10))

    assert last_error is not None
    raise RuntimeError(f"Failed to fetch DART corp codes: {last_error}") from last_error


def fetch_krx_stock_codes(
    session: requests.Session,
    url: str,
    timeout: int,
    retries: int,
) -> set[str]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            response.encoding = "cp949"
            tables = pd.read_html(io.StringIO(response.text), header=0, flavor="lxml")
            if not tables:
                raise RuntimeError("KRX table was not found")
            table = tables[0]
            code_column = table.columns[2]
            codes = {
                normalize_stock_code(value)
                for value in table[code_column].tolist()
                if normalize_stock_code(value)
            }
            if not codes:
                raise RuntimeError("KRX stock codes are empty")
            return codes
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(min(2**attempt, 10))

    assert last_error is not None
    raise RuntimeError(f"Failed to fetch KRX stock codes from {url}: {last_error}") from last_error


def load_stock_codes_from_csv(csv_path: Path | None) -> set[str]:
    if csv_path is None:
        return set()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file was not found: {csv_path}")

    encodings = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            with csv_path.open(encoding=encoding, newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames:
                    columns = {name.strip().lower(): name for name in reader.fieldnames if name}
                    stock_code_column = columns.get("stock_code")
                    if stock_code_column:
                        return {
                            normalize_stock_code(row.get(stock_code_column, ""))
                            for row in reader
                            if normalize_stock_code(row.get(stock_code_column, ""))
                        }

            with csv_path.open(encoding=encoding, newline="") as handle:
                reader = csv.reader(handle)
                next(reader, None)
                return {
                    normalize_stock_code(row[0])
                    for row in reader
                    if row and normalize_stock_code(row[0])
                }
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    assert last_error is not None
    raise RuntimeError(f"Failed to read delisted stock code CSV: {last_error}") from last_error


def classify_company(
    stock_code: str,
    listed_codes: set[str],
    delisted_codes: set[str],
) -> tuple[str, str]:
    if stock_code in delisted_codes:
        return "delisted", "Historical delisted stock code set (CSV + KRX)"
    if stock_code in listed_codes:
        return "listed", "DART corpCode + KRX current listed companies"
    return "unlisted", "DART corpCode only (not found in listed/delisted stock code sets)"


def build_company_master(
    dart_corp_codes: list[dict[str, str]],
    listed_codes: set[str],
    delisted_codes: set[str],
) -> list[CompanyRecord]:
    companies: list[CompanyRecord] = []
    for record in dart_corp_codes:
        stock_code = normalize_stock_code(record.get("stock_code", ""))
        if not stock_code:
            continue
        status, source = classify_company(stock_code, listed_codes, delisted_codes)
        companies.append(
            CompanyRecord(
                corp_code=record.get("corp_code", ""),
                corp_name=record.get("corp_name", ""),
                corp_eng_name=record.get("corp_eng_name", ""),
                stock_code=stock_code,
                modify_date=record.get("modify_date", ""),
                status=status,
                status_label=STATUS_LABELS[status],
                source=source,
            )
        )
    companies.sort(key=lambda item: (item.status, item.stock_code, item.corp_name))
    return companies


def filter_companies(companies: list[CompanyRecord], args: argparse.Namespace) -> list[CompanyRecord]:
    filtered = companies
    if args.status == "all":
        filtered = [company for company in filtered if company.status in DOWNLOADABLE_STATUSES]
    else:
        filtered = [company for company in filtered if company.status == args.status]

    corp_codes = {code.strip() for code in args.corp_code if code.strip()}
    if corp_codes:
        filtered = [company for company in filtered if company.corp_code in corp_codes]

    if args.limit is not None:
        filtered = filtered[: args.limit]
    return filtered


def build_output_path(output_root: Path, company: CompanyRecord, year: int, report_code: str) -> Path:
    folder_name = STATUS_LABELS[company.status]
    company_segment = sanitize_filename_part(company.corp_name)
    identifier = sanitize_filename_part(company.stock_code or company.corp_code)
    file_name = f"{identifier}__{company_segment}__{year}__{report_code}.json"
    return output_root / folder_name / company_segment / str(year) / file_name


def estimate_request_count(company_count: int, start_year: int, end_year: int) -> int:
    years = max(end_year - start_year + 1, 0)
    return company_count * years * len(REPORT_CODES) * 2


def fetch_financial_statement(
    session: requests.Session,
    api_key: str,
    company: CompanyRecord,
    year: int,
    report_code: str,
    timeout: int,
    retries: int,
) -> dict[str, Any]:
    responses: dict[str, Any] = {}
    selected_fs_div = ""
    selected_payload: dict[str, Any] | None = None

    for fs_div in ("CFS", "OFS"):
        payload = fetch_json(
            session=session,
            url=f"{DART_BASE_URL}/fnlttSinglAcntAll.json",
            params={
                "crtfc_key": api_key,
                "corp_code": company.corp_code,
                "bsns_year": str(year),
                "reprt_code": report_code,
                "fs_div": fs_div,
            },
            timeout=timeout,
            retries=retries,
        )
        responses[fs_div] = payload
        if payload.get("list"):
            selected_fs_div = fs_div
            selected_payload = payload
            break

    if selected_payload is None:
        selected_payload = responses.get("OFS") or responses.get("CFS") or {}

    has_data = bool(selected_payload.get("list"))
    status = str(selected_payload.get("status", "")).strip()
    no_data = not has_data and status in NO_DATA_STATUSES

    return {
        "selected_fs_div": selected_fs_div,
        "has_data": has_data,
        "no_data": no_data,
        "selected_response": selected_payload,
        "responses": responses,
    }


def build_file_payload(
    company: CompanyRecord,
    year: int,
    report_code: str,
    fetch_result: dict[str, Any],
) -> dict[str, Any]:
    selected_response = fetch_result["selected_response"]
    return {
        "downloaded_at": datetime.now().astimezone().isoformat(),
        "company": {
            "corp_code": company.corp_code,
            "corp_name": company.corp_name,
            "corp_eng_name": company.corp_eng_name,
            "stock_code": company.stock_code,
            "modify_date": company.modify_date,
        },
        "classification": {
            "status": company.status,
            "status_label": company.status_label,
            "classification_source": company.source,
        },
        "request": {
            "bsns_year": year,
            "reprt_code": report_code,
            "reprt_name": REPORT_CODES[report_code],
            "preferred_fs_div_order": ["CFS", "OFS"],
            "selected_fs_div": fetch_result["selected_fs_div"] or None,
        },
        "result": {
            "has_data": fetch_result["has_data"],
            "no_data": fetch_result["no_data"],
            "selected_status": selected_response.get("status"),
            "selected_message": selected_response.get("message"),
            "row_count": len(selected_response.get("list", [])),
        },
        "financial_statements": selected_response.get("list", []),
        "raw_responses": fetch_result["responses"],
    }


def write_master_metadata(
    meta_dir: Path,
    companies: list[CompanyRecord],
    filtered: list[CompanyRecord],
    csv_delisted_codes: set[str],
    krx_delisted_codes: set[str],
) -> None:
    summary = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "total_companies_with_stock_code": len(companies),
        "listed_count": sum(company.status == "listed" for company in companies),
        "delisted_count": sum(company.status == "delisted" for company in companies),
        "unlisted_count": sum(company.status == "unlisted" for company in companies),
        "selected_companies": len(filtered),
        "selected_statuses": sorted({company.status for company in filtered}),
        "csv_delisted_stock_code_count": len(csv_delisted_codes),
        "krx_delisted_stock_code_count": len(krx_delisted_codes),
    }
    write_json(meta_dir / "company_master_summary.json", summary)
    write_json(meta_dir / "company_master.json", [company.__dict__ for company in companies])


def write_run_summary(log_dir: Path, summary: dict[str, Any]) -> None:
    summary["finished_at"] = datetime.now().astimezone().isoformat()
    write_json(log_dir / "last_run_summary.json", summary)


def collect_financials(args: argparse.Namespace) -> int:
    if args.start_year > args.end_year:
        raise ValueError("--start-year must be less than or equal to --end-year")

    output_root = args.output_root
    meta_dir, log_dir = ensure_runtime_directories(output_root)
    api_key = get_dart_api_key()
    session = make_session()

    print("Fetching DART corporation codes...")
    dart_corp_codes = fetch_dart_corp_codes(session, api_key, args.request_timeout, args.retries)
    print("Fetching KRX listed companies...")
    listed_codes = fetch_krx_stock_codes(session, KRX_LISTED_URL, args.request_timeout, args.retries)
    print("Fetching KRX delisted companies...")
    krx_delisted_codes = fetch_krx_stock_codes(
        session, KRX_DELISTED_URL, args.request_timeout, args.retries
    )
    csv_delisted_codes = load_stock_codes_from_csv(args.delisted_stock_code_csv)
    if args.delisted_stock_code_csv is not None:
        print(
            f"Loaded {len(csv_delisted_codes)} delisted stock codes from "
            f"{args.delisted_stock_code_csv}"
        )
    delisted_codes = krx_delisted_codes | csv_delisted_codes

    companies = build_company_master(dart_corp_codes, listed_codes, delisted_codes)
    filtered_companies = filter_companies(companies, args)
    write_master_metadata(
        meta_dir,
        companies,
        filtered_companies,
        csv_delisted_codes,
        krx_delisted_codes,
    )

    estimated_requests = estimate_request_count(
        company_count=len(filtered_companies),
        start_year=args.start_year,
        end_year=args.end_year,
    )
    print(
        f"Selected companies: {len(filtered_companies)} "
        f"(listed={sum(item.status == 'listed' for item in filtered_companies)}, "
        f"delisted={sum(item.status == 'delisted' for item in filtered_companies)}, "
        f"unlisted={sum(item.status == 'unlisted' for item in filtered_companies)})"
    )
    print(f"Estimated API requests (worst case): {estimated_requests}")
    if args.status == "all":
        print("Note: unlisted companies are excluded from downloads when --status all is used.")

    summary: dict[str, Any] = {
        "started_at": datetime.now().astimezone().isoformat(),
        "args": {
            "start_year": args.start_year,
            "end_year": args.end_year,
            "status": args.status,
            "corp_code": args.corp_code,
            "limit": args.limit,
            "sleep_seconds": args.sleep_seconds,
            "overwrite": args.overwrite,
            "output_root": str(output_root),
            "delisted_stock_code_csv": (
                str(args.delisted_stock_code_csv) if args.delisted_stock_code_csv else None
            ),
        },
        "company_count": len(filtered_companies),
        "estimated_requests": estimated_requests,
        "saved_files": 0,
        "skipped_files": 0,
        "no_data_files": 0,
        "failed_requests": 0,
        "quota_exceeded": False,
        "stopped_early": False,
    }
    skipped_log = log_dir / "skipped.jsonl"
    failed_log = log_dir / "failures.jsonl"

    try:
        for company_index, company in enumerate(filtered_companies, start=1):
            print(
                f"[{company_index}/{len(filtered_companies)}] "
                f"{company.corp_name} ({company.stock_code}, {company.status})"
            )
            for year in range(args.start_year, args.end_year + 1):
                for report_code in REPORT_CODES:
                    output_path = build_output_path(output_root, company, year, report_code)
                    if output_path.exists() and not args.overwrite:
                        summary["skipped_files"] += 1
                        append_jsonl(
                            skipped_log,
                            {
                                "timestamp": datetime.now().astimezone().isoformat(),
                                "corp_code": company.corp_code,
                                "stock_code": company.stock_code,
                                "year": year,
                                "report_code": report_code,
                                "path": str(output_path),
                            },
                        )
                        continue

                    try:
                        fetch_result = fetch_financial_statement(
                            session=session,
                            api_key=api_key,
                            company=company,
                            year=year,
                            report_code=report_code,
                            timeout=args.request_timeout,
                            retries=args.retries,
                        )
                    except DartQuotaExceeded as exc:
                        summary["quota_exceeded"] = True
                        summary["stopped_early"] = True
                        append_jsonl(
                            failed_log,
                            {
                                "timestamp": datetime.now().astimezone().isoformat(),
                                "corp_code": company.corp_code,
                                "stock_code": company.stock_code,
                                "year": year,
                                "report_code": report_code,
                                "error": str(exc),
                                "error_type": "quota_exceeded",
                            },
                        )
                        raise
                    except Exception as exc:  # noqa: BLE001
                        summary["failed_requests"] += 1
                        append_jsonl(
                            failed_log,
                            {
                                "timestamp": datetime.now().astimezone().isoformat(),
                                "corp_code": company.corp_code,
                                "stock_code": company.stock_code,
                                "year": year,
                                "report_code": report_code,
                                "error": str(exc),
                                "error_type": type(exc).__name__,
                            },
                        )
                        continue

                    payload = build_file_payload(company, year, report_code, fetch_result)
                    write_json(output_path, payload)
                    summary["saved_files"] += 1
                    if payload["result"]["no_data"]:
                        summary["no_data_files"] += 1

                    if args.sleep_seconds > 0:
                        time.sleep(args.sleep_seconds)
    except DartQuotaExceeded as exc:
        print(f"Stopped because the DART API request limit was exceeded: {exc}")
    finally:
        write_run_summary(log_dir, summary)

    print("Run summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not summary["quota_exceeded"] else 2


def main() -> int:
    args = parse_args()
    return collect_financials(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nCollection cancelled by user.")
        raise SystemExit(130)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
