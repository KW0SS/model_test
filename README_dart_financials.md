# DART Financial Downloader

`dart_financial_downloader.py`는 Open DART API로 2015년부터 2025년까지 정기보고서 재무제표를 JSON으로 저장합니다.

## 실행 예시

```powershell
python .\dart_financial_downloader.py --start-year 2015 --end-year 2025 --status all
```

작은 범위로 먼저 검증하려면:

```powershell
python .\dart_financial_downloader.py --corp-code 00126380 --start-year 2024 --end-year 2025 --overwrite
```

## 출력 구조

- `downloads/dart_financials/상장기업/...`
- `downloads/dart_financials/상폐기업/...`
- `downloads/dart_financials/_meta/company_master.json`
- `downloads/dart_financials/_logs/last_run_summary.json`

## 동작 방식

- DART `corpCode.xml`에서 전체 법인 목록을 받습니다.
- KRX 현재 상장법인 목록과 KRX 상장폐지 목록을 함께 대조해 `상장기업`, `상폐기업`, `비상장기업`으로 분류합니다.
- 정기보고서 보고서코드 `11013`, `11012`, `11014`, `11011`을 순서대로 조회합니다.
- 재무제표는 연결(`CFS`) 우선, 없으면 별도(`OFS`)를 조회합니다.
- 기존 JSON 파일이 있으면 기본적으로 건너뜁니다. `--overwrite`를 주면 다시 저장합니다.
- `--status all`은 `상장기업`과 `상폐기업`만 수집하고 `비상장기업`은 제외합니다.

## 주의

- 요청 범위가 매우 크기 때문에 DART 일일 호출 제한에 걸릴 수 있습니다.
- 호출 제한으로 중단되더라도 이미 저장된 파일은 남아 있어 다음 실행 때 이어서 받을 수 있습니다.
