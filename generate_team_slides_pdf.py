from __future__ import annotations

import json
from pathlib import Path

import fitz
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph


ROOT = Path(r"C:\kwoss_C\model_test")
ARTIFACTS_DIR = ROOT / "artifacts" / "delisting"
OUTPUT_DIR = ROOT / "output" / "pdf"
TMP_DIR = ROOT / "tmp" / "pdfs"
ASSET_DIR = TMP_DIR / "assets"
PDF_PATH = OUTPUT_DIR / "delisting_project_team_slides.pdf"

FONT_REGULAR = r"C:\Windows\Fonts\malgun.ttf"
FONT_BOLD = r"C:\Windows\Fonts\malgunbd.ttf"

PAGE_W = 13.333 * 72
PAGE_H = 7.5 * 72
MARGIN = 34
NOTES_H = 108
CONTENT_BOTTOM = NOTES_H + 18

NAVY = colors.HexColor("#0C2747")
BLUE = colors.HexColor("#2F6BFF")
TEAL = colors.HexColor("#1FA5A3")
RED = colors.HexColor("#E14D57")
ORANGE = colors.HexColor("#F6A623")
GREEN = colors.HexColor("#2E9B4E")
LIGHT_BG = colors.HexColor("#F5F8FD")
CARD_BG = colors.HexColor("#FFFFFF")
TEXT = colors.HexColor("#1B2430")
MUTED = colors.HexColor("#5E6B7A")
NOTE_BG = colors.HexColor("#FFF9EE")
GRID = colors.HexColor("#D9E2F0")


def register_fonts() -> None:
    pdfmetrics.registerFont(TTFont("Malgun", FONT_REGULAR))
    pdfmetrics.registerFont(TTFont("Malgun-Bold", FONT_BOLD))


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)


def make_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Title"], fontName="Malgun-Bold", fontSize=26, leading=31, textColor=NAVY),
        "subtitle": ParagraphStyle("subtitle", parent=base["BodyText"], fontName="Malgun", fontSize=12, leading=16, textColor=MUTED),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontName="Malgun-Bold", fontSize=18, leading=22, textColor=NAVY),
        "body": ParagraphStyle("body", parent=base["BodyText"], fontName="Malgun", fontSize=11, leading=15, textColor=TEXT),
        "small": ParagraphStyle("small", parent=base["BodyText"], fontName="Malgun", fontSize=9.5, leading=12, textColor=MUTED),
        "note": ParagraphStyle("note", parent=base["BodyText"], fontName="Malgun", fontSize=9.6, leading=12.4, textColor=TEXT),
    }


STYLES = make_styles()


def draw_paragraph(c: canvas.Canvas, text: str, style: ParagraphStyle, x: float, y_top: float, width: float) -> float:
    para = Paragraph(text, style)
    _, h = para.wrap(width, 1000)
    para.drawOn(c, x, y_top - h)
    return h


def draw_bullets(c: canvas.Canvas, items: list[str], x: float, y_top: float, width: float, style: ParagraphStyle, bullet_color=BLUE) -> float:
    current = y_top
    total = 0
    for item in items:
        c.setFillColor(bullet_color)
        c.circle(x + 6, current - 8, 3, stroke=0, fill=1)
        h = draw_paragraph(c, item, style, x + 16, current, width - 16)
        current -= h + 6
        total += h + 6
    return total


def rounded_rect(c: canvas.Canvas, x: float, y: float, w: float, h: float, r: float, fill, stroke=None) -> None:
    c.setFillColor(fill)
    c.setStrokeColor(stroke or fill)
    c.roundRect(x, y, w, h, r, stroke=1 if stroke else 0, fill=1)


def header(c: canvas.Canvas, title: str, subtitle: str, page_no: int, total_pages: int) -> None:
    c.setFillColor(LIGHT_BG)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.rect(0, PAGE_H - 10, PAGE_W, 10, fill=1, stroke=0)
    c.setFillColor(BLUE)
    c.rect(0, PAGE_H - 10, PAGE_W * 0.22, 10, fill=1, stroke=0)
    c.setFont("Malgun-Bold", 23)
    c.setFillColor(NAVY)
    c.drawString(MARGIN, PAGE_H - 38, title)
    c.setFont("Malgun", 10.5)
    c.setFillColor(MUTED)
    c.drawString(MARGIN, PAGE_H - 54, subtitle)
    c.setFont("Malgun", 9)
    c.drawRightString(PAGE_W - MARGIN, PAGE_H - 40, f"{page_no:02d} / {total_pages:02d}")


def notes_box(c: canvas.Canvas, notes: list[str]) -> None:
    c.setFillColor(NOTE_BG)
    c.rect(0, 0, PAGE_W, NOTES_H, fill=1, stroke=0)
    c.setStrokeColor(colors.HexColor("#EEDFB8"))
    c.line(0, NOTES_H, PAGE_W, NOTES_H)
    c.setFont("Malgun-Bold", 11)
    c.setFillColor(NAVY)
    c.drawString(MARGIN, NOTES_H - 18, "발표 메모")
    draw_bullets(c, notes, MARGIN, NOTES_H - 28, PAGE_W - 2 * MARGIN, STYLES["note"], bullet_color=ORANGE)


def stat_card(c: canvas.Canvas, x: float, y: float, w: float, h: float, title: str, value: str, accent, sub: str = "") -> None:
    rounded_rect(c, x, y, w, h, 14, CARD_BG, GRID)
    c.setFillColor(accent)
    c.rect(x, y + h - 6, w, 6, fill=1, stroke=0)
    c.setFillColor(MUTED)
    c.setFont("Malgun", 10)
    c.drawString(x + 14, y + h - 24, title)
    c.setFillColor(NAVY)
    c.setFont("Malgun-Bold", 21)
    c.drawString(x + 14, y + h - 48, value)
    if sub:
        draw_paragraph(c, sub.replace("\n", "<br/>"), STYLES["small"], x + 14, y + h - 56, w - 28)


def draw_flow(c: canvas.Canvas, boxes: list[tuple[str, str]], x: float, y: float, w: float, h: float, colors_list: list) -> None:
    step_w = (w - 30 * (len(boxes) - 1)) / len(boxes)
    for idx, (title, body) in enumerate(boxes):
        bx = x + idx * (step_w + 30)
        rounded_rect(c, bx, y, step_w, h, 14, CARD_BG, GRID)
        c.setFillColor(colors_list[idx % len(colors_list)])
        c.roundRect(bx, y + h - 8, step_w, 8, 8, fill=1, stroke=0)
        c.setFillColor(NAVY)
        c.setFont("Malgun-Bold", 12)
        c.drawString(bx + 14, y + h - 28, title)
        draw_paragraph(c, body, STYLES["small"], bx + 14, y + h - 38, step_w - 28)
        if idx < len(boxes) - 1:
            ax = bx + step_w + 8
            ay = y + h / 2
            c.setStrokeColor(BLUE)
            c.setLineWidth(2)
            c.line(ax, ay, ax + 14, ay)
            c.line(ax + 10, ay + 4, ax + 14, ay)
            c.line(ax + 10, ay - 4, ax + 14, ay)


def draw_metric_bars(c: canvas.Canvas, metrics: pd.DataFrame, x: float, y: float, w: float, h: float) -> None:
    rounded_rect(c, x, y, w, h, 16, CARD_BG, GRID)
    c.setFont("Malgun-Bold", 12)
    c.setFillColor(NAVY)
    c.drawString(x + 14, y + h - 22, "모델별 핵심 지표")
    metrics_to_plot = [("recall", RED), ("precision", TEAL), ("f1", BLUE)]
    model_label = {"logistic": "Logistic", "xgboost": "XGBoost", "random_forest": "Random Forest"}
    left = x + 84
    top = y + h - 50
    row_gap = 54
    for idx, model in enumerate(metrics["model"].tolist()):
        row_y = top - idx * row_gap
        c.setFillColor(NAVY)
        c.setFont("Malgun-Bold", 10)
        c.drawString(x + 14, row_y + 12, model_label[model])
        for jdx, (metric, color) in enumerate(metrics_to_plot):
            val = float(metrics.loc[metrics["model"] == model, metric].iloc[0])
            bar_y = row_y - jdx * 12
            c.setFillColor(colors.HexColor("#E8EEF8"))
            c.roundRect(left, bar_y, w - 130, 8, 4, fill=1, stroke=0)
            c.setFillColor(color)
            c.roundRect(left, bar_y, (w - 130) * min(max(val, 0), 1), 8, 4, fill=1, stroke=0)
            c.setFillColor(TEXT)
            c.setFont("Malgun", 8.5)
            c.drawRightString(x + w - 14, bar_y + 1, f"{metric}: {val:.3f}")


def draw_imbalance_bar(c: canvas.Canvas, positive: int, negative: int, x: float, y: float, w: float, h: float) -> None:
    rounded_rect(c, x, y, w, h, 16, CARD_BG, GRID)
    total = positive + negative
    positive_ratio = positive / total
    c.setFont("Malgun-Bold", 12)
    c.setFillColor(NAVY)
    c.drawString(x + 14, y + h - 24, "클래스 불균형 시각화")
    bar_x = x + 20
    bar_y = y + h / 2 - 10
    bar_w = w - 40
    c.setFillColor(colors.HexColor("#DDE6F4"))
    c.roundRect(bar_x, bar_y, bar_w, 20, 10, fill=1, stroke=0)
    c.setFillColor(RED)
    c.roundRect(bar_x, bar_y, max(8, bar_w * positive_ratio), 20, 10, fill=1, stroke=0)
    c.setFillColor(NAVY)
    c.setFont("Malgun-Bold", 18)
    c.drawString(x + 20, y + 30, f"상폐기업 {positive:,}개")
    c.setFillColor(TEXT)
    c.setFont("Malgun", 10)
    c.drawString(x + 160, y + 34, f"전체의 {positive_ratio * 100:.1f}%")
    c.drawRightString(x + w - 20, y + 34, f"상장기업 {negative:,}개")
    c.setFillColor(MUTED)
    c.setFont("Malgun", 9)
    c.drawString(x + 20, y + 12, "상폐기업 비율이 너무 작아서 모델이 상장기업 쪽으로 치우치기 쉬운 구조입니다.")


def draw_confusion(c: canvas.Canvas, x: float, y: float, w: float, h: float) -> None:
    rounded_rect(c, x, y, w, h, 16, CARD_BG, GRID)
    c.setFillColor(NAVY)
    c.setFont("Malgun-Bold", 12)
    c.drawString(x + 14, y + h - 24, "Logistic 혼동행렬 해석")
    cell_w = (w - 80) / 2
    cell_h = (h - 74) / 2
    grid_x = x + 54
    grid_y = y + 20
    labels = [
        ("TN\n15,932", grid_x, grid_y + cell_h, colors.HexColor("#DDF2E3")),
        ("FP\n6,479", grid_x + cell_w, grid_y + cell_h, colors.HexColor("#FDE2E4")),
        ("FN\n178", grid_x, grid_y, colors.HexColor("#FFEFC6")),
        ("TP\n95", grid_x + cell_w, grid_y, colors.HexColor("#D9ECFF")),
    ]
    c.setFont("Malgun", 9)
    c.setFillColor(TEXT)
    c.drawCentredString(grid_x + cell_w / 2, y + h - 36, "실제 상장")
    c.drawCentredString(grid_x + cell_w * 1.5, y + h - 36, "실제 상폐")
    for label, bx, by, fill in labels:
        rounded_rect(c, bx, by, cell_w - 8, cell_h - 8, 12, fill, colors.white)
        txt = c.beginText(bx + 18, by + cell_h / 2 + 8)
        txt.setFont("Malgun-Bold", 16)
        txt.setFillColor(NAVY)
        for part in label.split("\n"):
            txt.textLine(part)
        c.drawText(txt)


def create_cover_image(path: Path) -> None:
    img = Image.new("RGB", (1400, 820), "#F4F8FF")
    draw = ImageDraw.Draw(img)
    bold = ImageFont.truetype(FONT_BOLD, 52)
    normal = ImageFont.truetype(FONT_REGULAR, 26)
    small = ImageFont.truetype(FONT_REGULAR, 22)
    draw.rounded_rectangle((780, 120, 1210, 620), radius=28, fill="#FFFFFF", outline="#D3DFF2", width=4)
    draw.rounded_rectangle((860, 180, 1280, 680), radius=28, fill="#FFFFFF", outline="#D3DFF2", width=4)
    draw.rounded_rectangle((940, 240, 1360, 740), radius=28, fill="#FFFFFF", outline="#D3DFF2", width=4)
    draw.rounded_rectangle((980, 280, 1320, 420), radius=22, fill="#2F6BFF")
    draw.text((1020, 315), "AI", font=bold, fill="#FFFFFF")
    draw.line((1040, 440, 1130, 540), fill="#1FA5A3", width=16)
    draw.line((1130, 540, 1220, 470), fill="#1FA5A3", width=16)
    draw.ellipse((1210, 455, 1270, 515), fill="#E14D57")
    draw.text((80, 120), "데이터 수집", font=normal, fill="#0C2747")
    draw.rounded_rectangle((70, 170, 320, 270), radius=24, fill="#FFFFFF", outline="#D3DFF2", width=3)
    draw.text((100, 205), "Open DART + KRX", font=small, fill="#2F6BFF")
    draw.text((80, 310), "재무비율 계산", font=normal, fill="#0C2747")
    draw.rounded_rectangle((70, 360, 320, 460), radius=24, fill="#FFFFFF", outline="#D3DFF2", width=3)
    draw.text((100, 395), "30개 지표 생성", font=small, fill="#1FA5A3")
    draw.text((80, 500), "모델 비교", font=normal, fill="#0C2747")
    draw.rounded_rectangle((70, 550, 320, 650), radius=24, fill="#FFFFFF", outline="#D3DFF2", width=3)
    draw.text((100, 585), "Logistic / RF / XGB", font=small, fill="#E14D57")
    img.save(path)


def create_data_image(path: Path) -> None:
    img = Image.new("RGB", (1280, 700), "#F5F8FD")
    draw = ImageDraw.Draw(img)
    bold = ImageFont.truetype(FONT_BOLD, 30)
    normal = ImageFont.truetype(FONT_REGULAR, 20)
    draw.rounded_rectangle((90, 210, 350, 470), radius=36, fill="#FFFFFF", outline="#D3DFF2", width=4)
    draw.rounded_rectangle((500, 210, 780, 470), radius=36, fill="#FFFFFF", outline="#D3DFF2", width=4)
    draw.rounded_rectangle((930, 210, 1190, 470), radius=36, fill="#FFFFFF", outline="#D3DFF2", width=4)
    draw.text((150, 250), "Open DART", font=bold, fill="#0C2747")
    draw.text((165, 300), "재무제표 JSON", font=normal, fill="#2F6BFF")
    draw.text((580, 250), "KRX", font=bold, fill="#0C2747")
    draw.text((555, 300), "상장/상폐 목록", font=normal, fill="#1FA5A3")
    draw.text((985, 250), "CSV", font=bold, fill="#0C2747")
    draw.text((985, 300), "학습 데이터셋", font=normal, fill="#E14D57")
    for sx, ex in [(350, 500), (780, 930)]:
        draw.line((sx, 340, ex, 340), fill="#2F6BFF", width=10)
        draw.polygon([(ex, 340), (ex - 25, 325), (ex - 25, 355)], fill="#2F6BFF")
    draw.text((370, 150), "데이터 수집에서 모델링까지의 흐름", font=bold, fill="#0C2747")
    img.save(path)


def create_roadmap_image(path: Path) -> None:
    img = Image.new("RGB", (1280, 700), "#F7FAFF")
    draw = ImageDraw.Draw(img)
    bold = ImageFont.truetype(FONT_BOLD, 26)
    normal = ImageFont.truetype(FONT_REGULAR, 18)
    steps = [("1", "라벨 재설계", "#2F6BFF"), ("2", "데이터 보강", "#1FA5A3"), ("3", "보정/파생변수", "#F6A623"), ("4", "검증/튜닝", "#E14D57")]
    x = 80
    for idx, (num, title, color) in enumerate(steps):
        draw.rounded_rectangle((x, 270, x + 240, 430), radius=28, fill="#FFFFFF", outline="#D3DFF2", width=4)
        draw.ellipse((x + 18, 292, x + 74, 348), fill=color)
        draw.text((x + 40, 303), num, font=bold, fill="#FFFFFF")
        draw.text((x + 95, 304), title, font=bold, fill="#0C2747")
        draw.text((x + 95, 346), "다음 단계 핵심 작업", font=normal, fill="#5E6B7A")
        if idx < len(steps) - 1:
            draw.line((x + 240, 350, x + 300, 350), fill=color, width=10)
            draw.polygon([(x + 300, 350), (x + 278, 335), (x + 278, 365)], fill=color)
        x += 300
    img.save(path)


def create_assets() -> dict[str, Path]:
    cover = ASSET_DIR / "cover.png"
    data = ASSET_DIR / "data_flow.png"
    roadmap = ASSET_DIR / "roadmap.png"
    create_cover_image(cover)
    create_data_image(data)
    create_roadmap_image(roadmap)
    return {"cover": cover, "data": data, "roadmap": roadmap}


def draw_cover(c: canvas.Canvas, assets: dict[str, Path], total_pages: int) -> None:
    header(c, "상장폐지 예측 프로젝트", "데이터 수집부터 모델 비교와 개선 방향까지", 1, total_pages)
    c.setFillColor(colors.white)
    c.roundRect(MARGIN, CONTENT_BOTTOM + 10, PAGE_W - 2 * MARGIN, PAGE_H - CONTENT_BOTTOM - 84, 20, fill=1, stroke=0)
    c.setFont("Malgun-Bold", 28)
    c.setFillColor(NAVY)
    c.drawString(MARGIN + 20, PAGE_H - 100, "상장기업과 상폐기업 재무비율 기반 AI 분류")
    draw_bullets(
        c,
        [
            "Open DART와 KRX 기반 데이터 수집 구조를 구축했습니다.",
            "재무제표에서 30개 재무비율을 계산해 학습용 CSV를 만들었습니다.",
            "Logistic, Random Forest, XGBoost를 학습해 성능을 비교했습니다.",
            "현재 한계와 다음 단계 개선 방향까지 문서화했습니다.",
        ],
        MARGIN + 24,
        PAGE_H - 132,
        360,
        STYLES["body"],
    )
    c.drawImage(ImageReader(str(assets["cover"])), PAGE_W - 430, CONTENT_BOTTOM + 40, width=360, height=250, mask="auto")
    stat_card(c, MARGIN + 20, CONTENT_BOTTOM + 26, 150, 82, "학습 행 수", "22,684", BLUE, "정제 후 최종 학습 데이터")
    stat_card(c, MARGIN + 182, CONTENT_BOTTOM + 26, 150, 82, "상폐기업", "273", RED, "전체의 1.2% 수준")
    stat_card(c, MARGIN + 344, CONTENT_BOTTOM + 26, 170, 82, "비교 모델", "3개", TEAL, "Logistic / RF / XGB")
    notes_box(
        c,
        [
            "이 자료는 데이터 수집부터 모델 결과와 개선 방향까지 전체 흐름을 팀원에게 공유하기 위한 슬라이드입니다.",
            "핵심 메시지는 1차 버전 파이프라인은 완성했지만, 데이터 구조와 라벨 정의 때문에 아직 실전형 성능은 아니라는 점입니다.",
            "이후 슬라이드에서 어떤 데이터를 모았고 왜 성능 한계가 생겼는지, 앞으로 무엇을 해야 하는지 순서대로 설명하겠습니다.",
        ],
    )


def draw_overview(c: canvas.Canvas, total_pages: int) -> None:
    header(c, "프로젝트 개요", "우리가 만든 전체 흐름", 2, total_pages)
    draw_flow(
        c,
        [
            ("1. 데이터 수집", "Open DART와 KRX를 이용해 상장기업과 상폐기업 재무제표를 연도별 JSON으로 수집"),
            ("2. 비율 계산", "재무제표 원본에서 30개 재무비율과 보조 금액 컬럼을 계산해 CSV 생성"),
            ("3. 모델 학습", "분류 모델 3개를 학습하고 그룹 기반 교차검증으로 성능 비교"),
            ("4. 결과 분석", "문제점과 개선 방향을 문서화하고 예측 결과 CSV 저장"),
        ],
        MARGIN,
        CONTENT_BOTTOM + 132,
        PAGE_W - 2 * MARGIN,
        160,
        [BLUE, TEAL, ORANGE, RED],
    )
    draw_bullets(
        c,
        [
            "현재 버전은 데이터 수집, 가공, 학습, 예측이 하나의 흐름으로 연결되어 있습니다.",
            "핵심 스크립트는 dart_financial_downloader.py, financial_ratio_calculator.py, delisting_model.py 입니다.",
            "이번 실험 목적은 완벽한 실무 모델 완성이 아니라, 상장폐지 예측의 초기 가능성과 한계를 확인하는 것입니다.",
        ],
        MARGIN + 10,
        CONTENT_BOTTOM + 108,
        PAGE_W - 2 * MARGIN - 20,
        STYLES["body"],
    )
    notes_box(
        c,
        [
            "이 슬라이드는 프로젝트가 단발성 실험이 아니라 end to end 파이프라인이라는 점을 보여주기 위한 페이지입니다.",
            "수집, 가공, 학습, 분석이 모두 연결되어 있어서 다음 단계 개선도 이 구조 위에서 바로 이어갈 수 있습니다.",
        ],
    )


def draw_data_collection(c: canvas.Canvas, assets: dict[str, Path], total_pages: int) -> None:
    header(c, "데이터 수집", "Open DART + KRX 기반 원천 데이터 확보", 3, total_pages)
    c.drawImage(ImageReader(str(assets["data"])), MARGIN, CONTENT_BOTTOM + 128, width=430, height=210, mask="auto")
    stat_card(c, PAGE_W - 460, CONTENT_BOTTOM + 232, 196, 96, "수집 연도 범위", "2015 - 2025", BLUE, "연도별 정기보고서 중심")
    stat_card(c, PAGE_W - 252, CONTENT_BOTTOM + 232, 196, 96, "수집 대상", "상장 + 상폐", TEAL, "학습용 양성/음성 동시 확보")
    draw_bullets(
        c,
        [
            "Open DART API에서 재무제표와 기업 메타 정보를 내려받고, KRX 목록으로 상장/상폐 상태를 구분했습니다.",
            "연결재무제표(CFS)를 우선 사용하고, 없을 때만 별도재무제표(OFS)를 사용했습니다.",
            "결과는 기업별, 연도별 JSON 파일로 저장되어 이후 재무비율 계산의 입력이 됩니다.",
        ],
        PAGE_W - 460,
        CONTENT_BOTTOM + 190,
        404,
        STYLES["body"],
    )
    draw_bullets(
        c,
        [
            "핵심 스크립트: dart_financial_downloader.py",
            "주요 출력: downloads/dart_financials/상장기업, downloads/dart_financials/상폐기업",
            "의미: 현재 상장기업뿐 아니라 상폐기업도 같이 모아야 분류 모델 학습이 가능합니다.",
        ],
        MARGIN + 10,
        CONTENT_BOTTOM + 92,
        PAGE_W - 2 * MARGIN - 20,
        STYLES["body"],
    )
    notes_box(
        c,
        [
            "설명의 포인트는 원천 데이터가 DART와 KRX라는 공신력 있는 출처에서 왔다는 점입니다.",
            "상폐기업 데이터까지 함께 모았다는 점이 중요합니다. 그래야 모델이 양성과 음성을 같이 배울 수 있습니다.",
        ],
    )


def draw_ratios(c: canvas.Canvas, total_pages: int) -> None:
    header(c, "재무비율 계산과 데이터셋", "재무제표를 모델이 읽기 좋은 형태로 변환", 4, total_pages)
    categories = [
        ("성장성", ["총자산증가율", "매출액증가율", "순이익증가율"], BLUE),
        ("수익성", ["매출액순이익률", "자기자본순이익률", "총자본영업이익률"], TEAL),
        ("활동성", ["매출채권회전율", "재고자산회전율", "총자본회전율"], ORANGE),
        ("안정성", ["부채비율", "유동비율", "자기자본비율"], RED),
    ]
    x = MARGIN
    for title, cols, color in categories:
        stat_card(c, x, CONTENT_BOTTOM + 198, 210, 112, title, "예시 3개", color, "\n".join(cols))
        x += 220
    draw_bullets(
        c,
        [
            "financial_ratio_calculator.py에서 원시 재무제표를 읽어 재무비율 30개와 진단용 금액 컬럼 7개를 계산했습니다.",
            "최종 학습용 CSV는 45개 컬럼이며, 모델에는 재무비율 30개만 feature로 사용했습니다.",
            "예측 테스트용으로 financial_ratios_2024_test.csv도 따로 준비했습니다.",
        ],
        MARGIN + 10,
        CONTENT_BOTTOM + 172,
        PAGE_W - 2 * MARGIN - 20,
        STYLES["small"],
    )
    stat_card(c, MARGIN + 40, CONTENT_BOTTOM + 16, 170, 76, "총 컬럼 수", "45", BLUE, "메타 8 + 비율 30 + 금액 7")
    stat_card(c, MARGIN + 222, CONTENT_BOTTOM + 16, 170, 76, "모델 입력", "30개", TEAL, "재무비율만 feature 사용")
    stat_card(c, MARGIN + 404, CONTENT_BOTTOM + 16, 170, 76, "테스트 파일", "2024", ORANGE, "예측용 CSV 별도 준비")
    notes_box(
        c,
        [
            "여기서 강조할 부분은 원시 재무제표를 바로 넣은 것이 아니라, 재무상태를 더 잘 보여주는 비율 형태로 바꿨다는 점입니다.",
            "비율화는 기업 규모 차이를 줄이고, 모델이 기업의 상대적인 위험 신호를 보게 하는 데 도움이 됩니다.",
        ],
    )


def draw_dataset_stats(c: canvas.Canvas, training_stats: dict[str, int], total_pages: int) -> None:
    header(c, "데이터 정제 결과", "원본에서 실제 학습 데이터로 가는 과정", 5, total_pages)
    counts = [("원본", training_stats["raw_rows"], BLUE), ("has_data=True", training_stats["after_has_data"], TEAL), ("전부 결측 제거", training_stats["after_all_null_feature_drop"], ORANGE), ("중복 정리 후", training_stats["after_dedup"], RED)]
    max_val = max(v for _, v, _ in counts)
    chart_x, chart_y, chart_w = MARGIN + 24, CONTENT_BOTTOM + 154, 420
    c.setFillColor(CARD_BG)
    c.roundRect(MARGIN, CONTENT_BOTTOM + 126, 470, 210, 18, fill=1, stroke=0)
    c.setFont("Malgun-Bold", 12)
    c.setFillColor(NAVY)
    c.drawString(MARGIN + 14, CONTENT_BOTTOM + 312, "정제 단계별 행 수")
    for idx, (label, value, color) in enumerate(counts):
        by = chart_y + (3 - idx) * 40
        c.setFillColor(colors.HexColor("#E8EEF8"))
        c.roundRect(chart_x, by, chart_w, 22, 8, fill=1, stroke=0)
        c.setFillColor(color)
        c.roundRect(chart_x, by, chart_w * value / max_val, 22, 8, fill=1, stroke=0)
        c.setFillColor(TEXT)
        c.setFont("Malgun-Bold", 10)
        c.drawString(MARGIN + 16, by + 5, label)
        c.drawRightString(chart_x + chart_w - 8, by + 5, f"{value:,}")
    stat_card(c, PAGE_W - 432, CONTENT_BOTTOM + 244, 180, 90, "최종 학습 행 수", f'{training_stats["after_dedup"]:,}', BLUE, "중복 정리와 결측 제거 반영")
    stat_card(c, PAGE_W - 240, CONTENT_BOTTOM + 244, 180, 90, "기업 수", f'{training_stats["company_count"]:,}', TEAL, "그룹 분할에 사용")
    stat_card(c, PAGE_W - 432, CONTENT_BOTTOM + 142, 180, 90, "상폐기업", f'{training_stats["positive_rows"]:,}', RED, "양성 샘플")
    stat_card(c, PAGE_W - 240, CONTENT_BOTTOM + 142, 180, 90, "상장기업", f'{training_stats["negative_rows"]:,}', GREEN, "음성 샘플")
    draw_bullets(
        c,
        [
            "has_data=False 행과 30개 비율이 모두 비어 있는 행을 제거했습니다.",
            "같은 종목코드와 연도가 중복일 때는 상폐 우선 규칙으로 하나만 남겼습니다.",
            "최종적으로 22,684행으로 정리되었고, 이 중 상폐기업은 273행입니다.",
        ],
        MARGIN + 8,
        CONTENT_BOTTOM + 92,
        PAGE_W - 2 * MARGIN - 16,
        STYLES["body"],
    )
    notes_box(
        c,
        [
            "이 페이지는 데이터 품질을 어떻게 정리했는지 보여주는 슬라이드입니다.",
            "중복 정리와 결측 제거를 거쳐도 상폐기업은 273개뿐이라서, 이후 모델 성능 한계와 직접 연결됩니다.",
        ],
    )


def draw_imbalance(c: canvas.Canvas, training_stats: dict[str, int], total_pages: int) -> None:
    header(c, "핵심 문제 1 - 클래스 불균형", "상폐기업이 너무 적어서 모델이 학습하기 어려운 구조", 6, total_pages)
    draw_imbalance_bar(c, training_stats["positive_rows"], training_stats["negative_rows"], MARGIN, CONTENT_BOTTOM + 180, PAGE_W - 2 * MARGIN, 140)
    draw_flow(
        c,
        [
            ("현상", "상폐기업 비율이 약 1.2%라서 모델이 대부분 상장기업만 보게 됩니다."),
            ("결과", "Random Forest는 상폐기업을 하나도 못 잡았고, XGBoost도 거의 못 잡았습니다."),
            ("해석", "데이터가 적은 양성 패턴을 못 배우거나, 잡으려 하면 오탐이 급증합니다."),
        ],
        MARGIN,
        CONTENT_BOTTOM + 54,
        PAGE_W - 2 * MARGIN,
        100,
        [RED, ORANGE, BLUE],
    )
    notes_box(
        c,
        [
            "팀 설명에서 가장 먼저 강조할 문제입니다. 양성 데이터가 너무 적으면 모델이 위험 신호를 배우기 어렵습니다.",
            "즉, 지금 성능 문제는 모델이 나빠서라기보다 데이터 구조 자체의 한계가 크다는 뜻입니다.",
        ],
    )


def draw_modeling(c: canvas.Canvas, total_pages: int) -> None:
    header(c, "모델링 파이프라인", "전처리, 모델 3종, 그룹 기반 교차검증", 7, total_pages)
    draw_flow(
        c,
        [
            ("전처리", "has_data=True, 전부 결측 제거, 중복 정리, 중앙값 대치"),
            ("모델", "Logistic Regression, Random Forest, XGBoost"),
            ("검증", "StratifiedGroupKFold 5-fold로 같은 기업이 train/test에 같이 들어가지 않도록 분리"),
            ("산출물", "metrics_summary.csv, comparison_report.md, predictions_2024.csv"),
        ],
        MARGIN,
        CONTENT_BOTTOM + 165,
        PAGE_W - 2 * MARGIN,
        155,
        [BLUE, TEAL, ORANGE, RED],
    )
    draw_bullets(
        c,
        [
            "학습 명령: python .\\delisting_model.py train --input .\\downloads\\dart_financials\\financial_ratios_2015_2025.csv --output-dir .\\artifacts\\delisting",
            "예측 명령: python .\\delisting_model.py predict --model-dir .\\artifacts\\delisting --input .\\downloads\\dart_financials\\financial_ratios_2024_test.csv --output .\\artifacts\\delisting\\predictions_2024.csv",
            "평가 지표는 Recall, Precision, F1, ROC-AUC, PR-AUC를 사용했고, 상폐기업을 놓치지 않는 Recall을 가장 중요하게 봤습니다.",
        ],
        MARGIN + 10,
        CONTENT_BOTTOM + 112,
        PAGE_W - 2 * MARGIN - 20,
        STYLES["body"],
    )
    notes_box(
        c,
        [
            "같은 기업이 학습과 검증에 동시에 들어가지 않도록 그룹 기반 검증을 사용했다는 점이 중요합니다.",
            "즉, 단순 정확도 장난이 아니라 최소한 데이터 누수를 줄이는 방향으로 평가했습니다.",
        ],
    )


def draw_metrics(c: canvas.Canvas, metrics: pd.DataFrame, total_pages: int) -> None:
    header(c, "모델 성능 비교", "3개 모델 중 Logistic이 상대적으로 가장 나은 결과", 8, total_pages)
    draw_metric_bars(c, metrics, MARGIN, CONTENT_BOTTOM + 160, PAGE_W - 2 * MARGIN, 170)
    stat_card(c, MARGIN + 10, CONTENT_BOTTOM + 48, 210, 86, "베스트 모델", "Logistic", BLUE, "Recall 기준으로 가장 높음")
    stat_card(c, MARGIN + 232, CONTENT_BOTTOM + 48, 210, 86, "Logistic Recall", "0.348", RED, "상폐기업 273개 중 95개 탐지")
    stat_card(c, MARGIN + 454, CONTENT_BOTTOM + 48, 210, 86, "Logistic Precision", "0.014", ORANGE, "오탐이 매우 많음")
    stat_card(c, MARGIN + 676, CONTENT_BOTTOM + 48, 210, 86, "핵심 해석", "최고지만 약함", TEAL, "3개 중 최고 != 실전형")
    notes_box(
        c,
        [
            "이 슬라이드에서 꼭 말해야 할 문장은 3개 중에서는 Logistic이 가장 낫지만 절대적인 성능은 아직 좋지 않다는 점입니다.",
            "상폐기업을 어느 정도 찾았다는 장점은 있지만, 정상 기업을 너무 많이 위험하다고 경고하는 문제가 큽니다.",
        ],
    )


def draw_interpretation(c: canvas.Canvas, total_pages: int) -> None:
    header(c, "결과 해석", "왜 현재 모델을 바로 실전에 쓰기 어려운가", 9, total_pages)
    draw_confusion(c, MARGIN, CONTENT_BOTTOM + 126, 430, 220)
    draw_bullets(
        c,
        [
            "Logistic은 상폐기업 273개 중 95개를 맞췄지만, 178개를 놓쳤습니다.",
            "동시에 상장기업 6,479개를 상폐기업이라고 잘못 경고했습니다.",
            "즉, 현재 상태로는 실무 최종 판정용보다 1차 위험 탐지용에 더 가깝습니다.",
        ],
        PAGE_W - 470,
        CONTENT_BOTTOM + 278,
        398,
        STYLES["small"],
    )
    draw_flow(
        c,
        [
            ("좋은 점", "데이터 수집부터 예측까지 전체 파이프라인을 만들었고, 재무비율만으로 어느 정도 신호가 있다는 점을 확인"),
            ("한계", "오탐이 많고 미래 예측 구조가 아니라서 실제 운영 지표와 차이가 큼"),
            ("현재 위치", "실전용 완성 모델보다 1차 프로토타입, 방향 확인용 시스템에 가까움"),
        ],
        PAGE_W - 470,
        CONTENT_BOTTOM + 32,
        420,
        136,
        [GREEN, RED, BLUE],
    )
    notes_box(
        c,
        [
            "여기서는 실패라는 표현보다 기반 시스템을 만들었고, 어디가 병목인지 확인한 단계라고 설명하는 것이 좋습니다.",
            "즉, 다음 단계의 우선순위를 정할 수 있게 해준 실험이라는 점이 중요합니다.",
        ],
    )


def draw_problems(c: canvas.Canvas, total_pages: int) -> None:
    header(c, "왜 한계가 생겼는가", "지금 구조에서 생기는 대표 문제 4가지", 10, total_pages)
    cards = [
        ("문제 1", "상폐기업이 너무 적음", "양성 데이터가 273개뿐이라 불균형이 심함", RED),
        ("문제 2", "라벨 정의 한계", "현재 상태 분류에 가까워 미래 상폐 예측과 완전히 같지 않음", BLUE),
        ("문제 3", "변수 부족", "감사의견, 관리종목, 거래정지 같은 핵심 신호가 빠짐", TEAL),
        ("문제 4", "오탐 과다", "위험 기업을 잡으려 하면 정상 기업 경고가 급증함", ORANGE),
    ]
    for idx, (title, head, body, color) in enumerate(cards):
        stat_card(c, MARGIN + idx * 230, CONTENT_BOTTOM + 188, 210, 128, title, head, color, body)
    draw_bullets(
        c,
        [
            "결국 지금 성능 문제는 모델 알고리즘 하나의 문제가 아니라 데이터 구조와 문제 정의의 문제입니다.",
            "따라서 다음 단계에서는 모델만 바꾸기보다 라벨 재설계, 데이터 보강, 보정 기법, 시간 기준 검증을 먼저 개선해야 합니다.",
        ],
        MARGIN + 8,
        CONTENT_BOTTOM + 110,
        PAGE_W - 2 * MARGIN - 16,
        STYLES["body"],
    )
    notes_box(
        c,
        [
            "이 페이지는 팀원들과 문제를 공유하는 핵심 슬라이드입니다.",
            "특히 상폐기업 데이터 부족과 라벨 정의 한계를 먼저 해결하지 않으면, 모델만 더 복잡하게 만들어도 큰 개선이 어렵다고 설명하면 됩니다.",
        ],
    )


def draw_corrections(c: canvas.Canvas, total_pages: int) -> None:
    header(c, "재무제표 데이터 보정", "모델 성능을 높이기 위해 적용할 수 있는 전처리", 11, total_pages)
    draw_flow(
        c,
        [
            ("결측치 보정", "중앙값 또는 업종/연도별 중앙값 대치 + 결측 플래그"),
            ("이상치 완화", "Winsorization, clipping, 금액 컬럼 log 변환"),
            ("파생변수", "전년 대비 변화, 2년 연속 적자, 부채 급등 여부"),
            ("불균형 대응", "class_weight, scale_pos_weight, threshold tuning"),
        ],
        MARGIN,
        CONTENT_BOTTOM + 175,
        PAGE_W - 2 * MARGIN,
        150,
        [BLUE, TEAL, ORANGE, RED],
    )
    draw_bullets(
        c,
        [
            "재무제표 데이터라고 해서 원본을 그대로 쓰는 것이 일반적인 것은 아닙니다. 모델용 데이터셋은 보정을 거치는 경우가 훨씬 많습니다.",
            "다만 전체 데이터로 기준을 잡으면 데이터 누수가 생기므로, 중앙값과 이상치 기준은 반드시 train 데이터에서만 계산해야 합니다.",
            "현재 프로젝트에서는 결측치 보정, 이상치 완화, 시계열 파생변수, 불균형 대응이 우선순위가 높습니다.",
        ],
        MARGIN + 10,
        CONTENT_BOTTOM + 100,
        PAGE_W - 2 * MARGIN - 20,
        STYLES["body"],
    )
    notes_box(
        c,
        [
            "보정이라는 것은 회계 데이터를 임의로 왜곡하는 것이 아니라, 모델이 학습하기 좋게 안정화하는 전처리 작업이라고 설명하면 됩니다.",
            "핵심은 회계 의미를 해치지 않으면서 train 기준으로만 보정해 데이터 누수를 막는 것입니다.",
        ],
    )


def draw_roadmap(c: canvas.Canvas, assets: dict[str, Path], total_pages: int) -> None:
    header(c, "개선 로드맵 - 라벨 재설계 우선", "다음 연도 상폐 여부 예측 구조를 먼저 확정", 12, total_pages)
    draw_flow(
        c,
        [
            ("목표 변경", "현재 상태 분류가 아니라 다음 연도 상폐 여부 예측으로 문제를 다시 정의"),
            ("라벨 규칙", "예: 2021년 재무비율을 입력으로 넣고 2022년 실제 상폐 여부를 정답으로 연결"),
            ("완료 기준", "상폐 시점 정의와 라벨 작성 규칙이 문서로 정리되고 한 줄로 설명 가능해야 함"),
        ],
        MARGIN,
        CONTENT_BOTTOM + 204,
        PAGE_W - 2 * MARGIN,
        118,
        [BLUE, TEAL, ORANGE],
    )
    stat_card(c, MARGIN + 10, CONTENT_BOTTOM + 112, 280, 80, "해야 할 일", "상폐 시점 정의", RED, "실제 상폐 연도를 표로 정리하고\n직전 연도를 양성으로 둘지 규칙 확정")
    stat_card(c, MARGIN + 302, CONTENT_BOTTOM + 112, 280, 80, "예시 구조", "2021 -> 2022", BLUE, "2021 재무비율로 2022 상폐 여부를 맞히는\n미래 예측형 라벨 구조로 변경")
    stat_card(c, MARGIN + 594, CONTENT_BOTTOM + 112, 280, 80, "추천", "1년 뒤 상폐", TEAL, "가장 먼저 다음 연도 상폐 여부 버전부터\n만들어 기준 모델을 다시 학습")
    draw_bullets(
        c,
        [
            "현재 구조는 기업이 지금 상장/상폐 그룹인지 분류하는 데 가까워서, 실제로 쓰고 싶은 미래 예측 문제와 차이가 있습니다.",
            "그래서 먼저 상폐기업의 실제 상폐 연도를 정리하고, 어떤 연도 데이터를 넣었을 때 몇 년 뒤 상폐를 맞히는지 라벨 규칙을 명확히 해야 합니다.",
            "이 작업이 끝나야 이후의 추가 변수 결합, 시간 기준 검증, threshold tuning도 의미 있게 진행됩니다.",
        ],
        MARGIN + 10,
        CONTENT_BOTTOM + 88,
        PAGE_W - 2 * MARGIN - 20,
        STYLES["small"],
    )
    notes_box(
        c,
        [
            "이 슬라이드에서는 다음 단계의 출발점이 왜 라벨 재설계인지 설명하면 됩니다.",
            "핵심은 다음 연도 상폐 여부를 맞히는 구조로 바꿔야 이후 성능 비교가 실제 목적과 맞아진다는 점입니다.",
        ],
    )


def draw_team_tasks(c: canvas.Canvas, total_pages: int) -> None:
    header(c, "팀 다음 액션", "바로 실행할 수 있는 체크리스트", 13, total_pages)
    draw_flow(
        c,
        [
            ("데이터 담당", "상폐 연도 정리, 관리종목/거래정지/감사의견 데이터 수집 가능 여부 확인"),
            ("모델 담당", "라벨 재설계, 시간 기준 분할, 보정 기법과 threshold tuning 실험"),
            ("문서 담당", "실험 로그, 성능 비교표, 오류 사례 분석 정리"),
        ],
        MARGIN,
        CONTENT_BOTTOM + 172,
        PAGE_W - 2 * MARGIN,
        150,
        [BLUE, TEAL, ORANGE],
    )
    stat_card(c, MARGIN + 18, CONTENT_BOTTOM + 54, 260, 88, "바로 해야 할 일 1", "상폐 연도 표 정리", RED, "미래 예측 라벨 생성의 출발점")
    stat_card(c, MARGIN + 300, CONTENT_BOTTOM + 54, 260, 88, "바로 해야 할 일 2", "추가 변수 확보", BLUE, "감사의견, 관리종목, 거래정지")
    stat_card(c, MARGIN + 582, CONTENT_BOTTOM + 54, 260, 88, "바로 해야 할 일 3", "재학습/재검증", TEAL, "time split + tuning + error analysis")
    notes_box(
        c,
        [
            "팀원들에게 넘길 때는 역할을 나눠서 움직이는 것이 가장 효율적입니다.",
            "데이터 확보와 라벨 재설계가 먼저 끝나야 모델 실험도 의미 있게 돌아가므로, 이 순서를 맞추는 것이 중요합니다.",
        ],
    )


def build_pdf() -> Path:
    register_fonts()
    ensure_dirs()
    assets = create_assets()
    metrics = pd.read_csv(ARTIFACTS_DIR / "metrics_summary.csv")
    with open(ARTIFACTS_DIR / "training_stats.json", "r", encoding="utf-8") as f:
        training_stats = json.load(f)
    total_pages = 13
    c = canvas.Canvas(str(PDF_PATH), pagesize=(PAGE_W, PAGE_H))
    c.setTitle("상장폐지 예측 프로젝트 팀 공유 슬라이드")
    c.setAuthor("OpenAI Codex")
    c.setSubject("상장폐지 예측 프로젝트 발표 자료")
    draw_cover(c, assets, total_pages); c.showPage()
    draw_overview(c, total_pages); c.showPage()
    draw_data_collection(c, assets, total_pages); c.showPage()
    draw_ratios(c, total_pages); c.showPage()
    draw_dataset_stats(c, training_stats, total_pages); c.showPage()
    draw_imbalance(c, training_stats, total_pages); c.showPage()
    draw_modeling(c, total_pages); c.showPage()
    draw_metrics(c, metrics, total_pages); c.showPage()
    draw_interpretation(c, total_pages); c.showPage()
    draw_problems(c, total_pages); c.showPage()
    draw_corrections(c, total_pages); c.showPage()
    draw_roadmap(c, assets, total_pages); c.showPage()
    draw_team_tasks(c, total_pages); c.save()
    return PDF_PATH


def render_pdf(pdf_path: Path) -> list[Path]:
    doc = fitz.open(pdf_path)
    images = []
    for idx, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
        out = TMP_DIR / f"delisting_team_slide_{idx:02d}.png"
        pix.save(out)
        images.append(out)
    doc.close()
    return images


def validate_pdf(pdf_path: Path) -> None:
    from pypdf import PdfReader
    reader = PdfReader(str(pdf_path))
    print(f"pdf_pages={len(reader.pages)}")
    for idx, page in enumerate(reader.pages[:3], start=1):
        snippet = (page.extract_text() or "").replace("\n", " ")
        print(f"page_{idx}_text={snippet[:180]}")


def main() -> None:
    pdf_path = build_pdf()
    rendered = render_pdf(pdf_path)
    validate_pdf(pdf_path)
    print(f"pdf_path={pdf_path}")
    print(f"rendered_pages={len(rendered)}")
    print(f"first_render={rendered[0] if rendered else 'N/A'}")


if __name__ == "__main__":
    main()
