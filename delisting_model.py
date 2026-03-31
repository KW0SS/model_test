from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from delisting_data import (
    build_conflict_report,
    build_labeled_dataset,
    coerce_financial_frame,
    load_or_build_events,
    prepare_prediction_data,
    prepare_training_data,
    save_preparation_outputs,
)
from delisting_shared import (
    CODE_COLUMN,
    DEFAULT_COMPANY_MASTER,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PREDICT_INPUT,
    DEFAULT_TRAIN_INPUT,
    FEATURE_COLUMNS,
    PREDICTION_ELIGIBLE_COLUMN,
    PREDICTION_LABELS,
    PREDICTION_LABEL_COLUMN,
    PREDICTION_MODEL_COLUMN,
    PREDICTION_PROBABILITY_COLUMN,
    PREDICTION_SKIP_REASON_COLUMN,
    TARGET_COLUMN,
    YEAR_COLUMN,
    parse_thresholds,
    read_csv,
    write_json,
)
from delisting_train import (
    build_logistic_model,
    build_prediction_frame,
    build_threshold_comparison,
    compute_metrics,
    fit_and_save_logistic,
    merge_prediction_columns,
    select_threshold,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="상장폐지 미래예측 모델 전처리/학습/예측 CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_events_parser = subparsers.add_parser("build-events", help="company_master.json에서 상폐 사건 테이블을 생성합니다.")
    build_events_parser.add_argument("--company-master", type=Path, default=DEFAULT_COMPANY_MASTER, help="company_master.json 경로")
    build_events_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR / "delist_events.csv", help="출력 CSV 경로")

    prepare_parser = subparsers.add_parser("prepare-data", help="미래 상폐 라벨링과 품질 리포트를 생성합니다.")
    prepare_parser.add_argument("--input", type=Path, default=DEFAULT_TRAIN_INPUT, help="재무비율 CSV 경로")
    prepare_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="전처리 산출물 디렉터리")
    prepare_parser.add_argument("--events", type=Path, default=None, help="기존 상폐 사건 CSV 경로")
    prepare_parser.add_argument("--company-master", type=Path, default=DEFAULT_COMPANY_MASTER, help="자동 생성용 company_master.json 경로")

    train_parser = subparsers.add_parser("train", help="미래 상폐 라벨을 생성한 뒤 LogisticRegression 베이스라인을 학습합니다.")
    train_parser.add_argument("--input", type=Path, default=DEFAULT_TRAIN_INPUT, help="재무비율 CSV 경로")
    train_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="학습 산출물 디렉터리")
    train_parser.add_argument("--events", type=Path, default=None, help="기존 상폐 사건 CSV 경로")
    train_parser.add_argument("--company-master", type=Path, default=DEFAULT_COMPANY_MASTER, help="자동 생성용 company_master.json 경로")
    train_parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    train_parser.add_argument("--train-start-year", type=int, default=2015, help="학습 시작 연도")
    train_parser.add_argument("--train-end-year", type=int, default=2020, help="학습 종료 연도")
    train_parser.add_argument("--valid-year", type=int, default=2021, help="검증 연도")
    train_parser.add_argument("--test-year", type=int, default=2022, help="테스트 연도")
    train_parser.add_argument("--thresholds", default="0.1,0.2,0.3,0.4,0.5,0.6", help="threshold sweep 목록")

    predict_parser = subparsers.add_parser("predict", help="저장된 LogisticRegression 모델로 예측합니다.")
    predict_parser.add_argument("--model-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="모델 산출물 디렉터리")
    predict_parser.add_argument("--input", type=Path, default=DEFAULT_PREDICT_INPUT, help="예측용 CSV 경로")
    predict_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR / "predictions_2024.csv", help="예측 결과 CSV 경로")
    predict_parser.add_argument("--model", default="best", choices=["best", "logistic"], help="사용할 저장 모델")
    return parser.parse_args()


def run_build_events(args: argparse.Namespace) -> None:
    from delisting_data import build_delist_events_from_master

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    events_df = build_delist_events_from_master(args.company_master.resolve())
    events_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"상폐 사건 테이블 생성 완료: {output_path}")
    print(f"event rows: {len(events_df)}")


def run_prepare_data(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    raw_df = read_csv(args.input.resolve())
    events_df = load_or_build_events(output_dir, args.events.resolve() if args.events else None, args.company_master.resolve())
    labeled_df, quality_report, excluded_y_minus_1 = build_labeled_dataset(raw_df, events_df)
    conflicts = build_conflict_report(coerce_financial_frame(raw_df))
    save_preparation_outputs(output_dir, events_df, labeled_df, quality_report, conflicts, excluded_y_minus_1)

    summary = {
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "event_rows": int(len(events_df)),
        "labeled_rows": int(len(labeled_df)),
        "included_rows": int(labeled_df["include_for_training"].sum()),
        "positive_rows": int((labeled_df[TARGET_COLUMN] == 1).fillna(False).sum()),
        "excluded_y_minus_1_rows": int((labeled_df["exclude_reason"] == "excluded_y_minus_1").sum()),
        "conflict_rows": int(len(conflicts)),
    }
    write_json(output_dir / "preparation_summary.json", summary)
    print(f"라벨링 데이터 생성 완료: {output_dir}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def run_train(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = read_csv(args.input.resolve())
    events_df = load_or_build_events(output_dir, args.events.resolve() if args.events else None, args.company_master.resolve())
    labeled_df, quality_report, excluded_y_minus_1 = build_labeled_dataset(raw_df, events_df)
    conflicts = build_conflict_report(coerce_financial_frame(raw_df))
    save_preparation_outputs(output_dir, events_df, labeled_df, quality_report, conflicts, excluded_y_minus_1)

    training_data = prepare_training_data(
        labeled_df=labeled_df,
        train_start_year=args.train_start_year,
        train_end_year=args.train_end_year,
        valid_year=args.valid_year,
        test_year=args.test_year,
    )
    thresholds = parse_thresholds(args.thresholds)
    model = build_logistic_model(args.seed)

    X_train = training_data.train_frame[FEATURE_COLUMNS]
    y_train = training_data.train_frame[TARGET_COLUMN].astype(int)
    X_valid = training_data.valid_frame[FEATURE_COLUMNS]
    y_valid = training_data.valid_frame[TARGET_COLUMN].astype(int)
    X_test = training_data.test_frame[FEATURE_COLUMNS]
    y_test = training_data.test_frame[TARGET_COLUMN].astype(int)

    model.fit(X_train, y_train)
    valid_probabilities = model.predict_proba(X_valid)[:, 1]
    test_probabilities = model.predict_proba(X_test)[:, 1]

    threshold_frame = build_threshold_comparison(training_data.valid_frame, valid_probabilities, thresholds)
    selected_threshold = select_threshold(threshold_frame)

    split_rows = []
    for split_name, y_true, probabilities in [
        ("valid", y_valid.to_numpy(), valid_probabilities),
        ("test", y_test.to_numpy(), test_probabilities),
    ]:
        row_default = compute_metrics(y_true, probabilities, 0.5)
        row_default["split"] = split_name
        row_default["threshold_type"] = "default_0.5"
        split_rows.append(row_default)

        row_selected = compute_metrics(y_true, probabilities, selected_threshold)
        row_selected["split"] = split_name
        row_selected["threshold_type"] = "selected"
        split_rows.append(row_selected)

    split_metrics = pd.DataFrame(split_rows).sort_values(by=["split", "threshold_type"], kind="mergesort").reset_index(drop=True)

    valid_predictions = build_prediction_frame(training_data.valid_frame, valid_probabilities, selected_threshold, "valid")
    test_predictions = build_prediction_frame(training_data.test_frame, test_probabilities, selected_threshold, "test")
    valid_predictions.to_csv(output_dir / "valid_predictions.csv", index=False, encoding="utf-8-sig")
    test_predictions.to_csv(output_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    fit_and_save_logistic(
        output_dir=output_dir,
        training_data=training_data,
        split_metrics=split_metrics,
        threshold_frame=threshold_frame,
        selected_threshold=selected_threshold,
        seed=args.seed,
        split_config={
            "train_start_year": args.train_start_year,
            "train_end_year": args.train_end_year,
            "valid_year": args.valid_year,
            "test_year": args.test_year,
        },
    )

    print(f"학습 완료: {output_dir}")
    print(split_metrics.to_string(index=False))


def load_model_bundle(model_dir: Path, model_name: str) -> dict[str, object]:
    model_path = model_dir / ("best_model.joblib" if model_name == "best" else "model_logistic.joblib")
    if not model_path.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    return joblib.load(model_path)


def run_predict(args: argparse.Namespace) -> None:
    model_dir = args.model_dir.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = load_model_bundle(model_dir, args.model)
    input_df = read_csv(args.input.resolve())
    prepared = prepare_prediction_data(input_df)
    if prepared.working_frame.empty:
        prepared.output_frame.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"예측 가능한 행이 없어 빈 결과만 저장했습니다: {output_path}")
        return

    probabilities = bundle["estimator"].predict_proba(prepared.working_frame[bundle["feature_columns"]])[:, 1]
    predictions = (probabilities >= bundle["threshold"]).astype(int)
    prediction_frame = prepared.working_frame[[CODE_COLUMN, YEAR_COLUMN]].copy()
    prediction_frame[PREDICTION_PROBABILITY_COLUMN] = probabilities
    prediction_frame[PREDICTION_LABEL_COLUMN] = [PREDICTION_LABELS[int(value)] for value in predictions]
    prediction_frame[PREDICTION_MODEL_COLUMN] = bundle["model_name"]
    prediction_frame[PREDICTION_ELIGIBLE_COLUMN] = True
    prediction_frame[PREDICTION_SKIP_REASON_COLUMN] = ""
    merged = merge_prediction_columns(prepared.output_frame, prediction_frame)
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"예측 완료: {output_path}")
    print(f"입력 행 수: {len(input_df)}")
    print(f"예측 가능 행 수: {len(prepared.working_frame)}")
    print(f"선택 threshold: {bundle['threshold']:.2f}")


def main() -> None:
    args = parse_args()
    if args.command == "build-events":
        run_build_events(args)
    elif args.command == "prepare-data":
        run_prepare_data(args)
    elif args.command == "train":
        run_train(args)
    elif args.command == "predict":
        run_predict(args)
    else:
        raise ValueError(f"지원하지 않는 명령입니다: {args.command}")


if __name__ == "__main__":
    main()
