import base64
import io
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF
import pandas as pd
import requests
import streamlit as st
from dateutil import parser as date_parser
from PIL import Image

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Spacer, Paragraph, Table, TableStyle


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Expense Parser",
    page_icon="🧾",
    layout="wide",
)


def load_css(file_name: str):
    if os.path.exists(file_name):
        with open(file_name, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css("assets/style.css")

# fallback styles if file is missing
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: "Inter", sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(120,180,255,0.22), transparent 28%),
        radial-gradient(circle at top right, rgba(255,255,255,0.18), transparent 24%),
        linear-gradient(135deg, #0b1220 0%, #121a2b 45%, #0a1320 100%);
    color: #eef4ff;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

.glass-card {
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 24px;
    padding: 20px;
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    margin-bottom: 1rem;
}

h1, h2, h3 {
    color: #f7fbff;
}

.stButton > button,
.stDownloadButton > button {
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.18);
    background: rgba(255,255,255,0.12);
    color: white;
    backdrop-filter: blur(12px);
    padding: 0.6rem 1.1rem;
}

div[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.16);
    border-radius: 18px;
    padding: 12px;
}

div[data-testid="stDataEditor"] {
    border-radius: 18px;
    overflow: hidden;
}

[data-testid="stMetric"] {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.12);
    padding: 14px;
    border-radius: 18px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# CONFIG
# =========================
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY", "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"))
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "openrouter/free"

DOCUMENT_TYPES = ["поезд", "самолет", "такси", "проживание"]

REQUIRED_FIELDS = [
    "document_type",
    "first_name",
    "last_name",
    "middle_name",
    "departure_date",
    "arrival_date",
    "amount",
    "currency",
    "provider",
    "route",
    "ticket_or_booking_number",
    "payment_date",
    "missing_fields",
    "fraud_score",
    "fraud_flags",
    "review_status",
    "source_file",
]

MODEL_JSON_SCHEMA = {
    "name": "expense_document",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "document_type": {"type": "string"},
            "first_name": {"type": "string"},
            "last_name": {"type": "string"},
            "middle_name": {"type": "string"},
            "departure_date": {"type": "string"},
            "arrival_date": {"type": "string"},
            "amount": {"type": ["number", "null"]},
            "currency": {"type": "string"},
            "provider": {"type": "string"},
            "route": {"type": "string"},
            "ticket_or_booking_number": {"type": "string"},
            "payment_date": {"type": "string"},
            "missing_fields": {
                "type": "array",
                "items": {"type": "string"}
            },
            "fraud_score": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100
            },
            "fraud_flags": {
                "type": "array",
                "items": {"type": "string"}
            },
            "review_status": {
                "type": "string",
                "enum": ["ok", "warning", "suspicious"]
            }
        },
        "required": [
            "document_type",
            "first_name",
            "last_name",
            "middle_name",
            "departure_date",
            "arrival_date",
            "amount",
            "currency",
            "provider",
            "route",
            "ticket_or_booking_number",
            "payment_date",
            "missing_fields",
            "fraud_score",
            "fraud_flags",
            "review_status"
        ],
        "additionalProperties": False
    }
}

# =========================
# APP TITLE
# =========================
st.title("🧾 Expense Parser")
st.caption("Разбор билетов, чеков, бронирований и документов на проживание через Streamlit + OpenRouter API")

if "documents" not in st.session_state:
    st.session_state.documents = []


# =========================
# HELPERS
# =========================
def safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_date(value: Any) -> str:
    if not value:
        return ""
    text = safe_str(value)
    try:
        dt = date_parser.parse(text, dayfirst=True, fuzzy=True)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return text


def parse_date_safe(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        return date_parser.parse(str(value), dayfirst=True, fuzzy=True)
    except Exception:
        return None


def normalize_amount(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    text = safe_str(value)
    text = text.replace(" ", "").replace(",", ".")
    match = re.search(r"[-+]?\d*\.?\d+", text)
    if not match:
        return None
    try:
        return float(match.group())
    except Exception:
        return None


def image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return encoded


def render_pdf_pages(pdf_bytes: bytes, max_pages: Optional[int] = None) -> List[Image.Image]:
    images = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(doc)
    pages_to_process = total_pages if max_pages is None else min(total_pages, max_pages)

    for page_num in range(pages_to_process):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)

    return images


def extract_text_from_pdf(pdf_bytes: bytes, max_pages: Optional[int] = None) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    texts = []
    total_pages = len(doc)
    pages_to_process = total_pages if max_pages is None else min(total_pages, max_pages)

    for page_num in range(pages_to_process):
        page = doc[page_num]
        txt = page.get_text("text")
        if txt:
            texts.append(f"\n--- PAGE {page_num + 1} ---\n{txt.strip()}")

    return "\n\n".join([t for t in texts if t])


def quality_of_text(text: str) -> bool:
    if not text:
        return False
    stripped = text.strip()
    if len(stripped) < 80:
        return False
    letters = re.findall(r"[A-Za-zА-Яа-я0-9]", stripped)
    return len(letters) > 50


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def clean_model_json(raw_content: str) -> Dict[str, Any]:
    raw_content = raw_content.strip()

    code_block_match = re.search(r"```json\s*(.*?)\s*```", raw_content, flags=re.DOTALL)
    if code_block_match:
        raw_content = code_block_match.group(1).strip()

    obj_match = re.search(r"\{.*\}", raw_content, flags=re.DOTALL)
    if obj_match:
        raw_content = obj_match.group(0)

    try:
        return json.loads(raw_content)
    except Exception:
        return {
            "document_type": "",
            "first_name": "",
            "last_name": "",
            "middle_name": "",
            "departure_date": "",
            "arrival_date": "",
            "amount": None,
            "currency": "",
            "provider": "",
            "route": "",
            "ticket_or_booking_number": "",
            "payment_date": "",
            "missing_fields": ["model_json_parse_error"],
            "fraud_score": 0,
            "fraud_flags": ["model_json_parse_error"],
            "review_status": "warning",
        }


def build_extraction_prompt(filename: str, source_type: str) -> str:
    return f"""
Ты — система извлечения данных из транспортных и гостиничных документов для бухгалтерии.

Нужно извлечь данные из документа.
НЕЛЬЗЯ придумывать значения.
Если поля нет — оставь пустую строку или null, а название поля добавь в missing_fields.

Также нужно оценить РИСК подозрительности документа.
Важно:
- НЕ утверждай, что документ точно поддельный.
- Оцени только подозрительные признаки.
- fraud_score: число от 0 до 100
- review_status:
  - ok = подозрительных признаков нет
  - warning = есть слабые или средние признаки
  - suspicious = есть сильные признаки, документ требует ручной проверки

Подозрительные признаки:
- следы редактирования или наложения текста
- разный стиль шрифтов в ключевых местах
- нелогичные даты
- отсутствующие обязательные реквизиты
- подозрительные суммы
- несоответствие маршрута, типа документа и дат
- обрезанные или неестественные фрагменты документа

Типы документов:
- поезд
- самолет
- такси
- проживание

Верни строго JSON по заданной схеме.

Имя файла: {filename}
Тип источника: {source_type}
""".strip()


def call_openrouter(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("Не найден OPENROUTER_API_KEY. Добавь его в st.secrets или переменные окружения.")

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "temperature": 0,
        "response_format": {
            "type": "json_schema",
            "json_schema": MODEL_JSON_SCHEMA
        },
        "plugins": [
            {"id": "response-healing"}
        ]
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-OpenRouter-Title": "Expense Parser MVP",
    }

    response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=240)
    response.raise_for_status()
    data = response.json()

    try:
        content = data["choices"][0]["message"]["content"]
    except Exception:
        raise RuntimeError(f"Неожиданный ответ OpenRouter: {data}")

    return clean_model_json(content)


def call_openrouter_with_text(text: str, filename: str) -> Dict[str, Any]:
    messages = [
        {
            "role": "system",
            "content": "Ты аккуратный парсер документов. Всегда возвращай только валидный JSON."
        },
        {
            "role": "user",
            "content": build_extraction_prompt(filename, "text") + f"\n\nТекст документа:\n{text}"
        }
    ]
    return call_openrouter(messages)


def call_openrouter_with_images(images: List[Image.Image], filename: str) -> Dict[str, Any]:
    content = [
        {
            "type": "text",
            "text": build_extraction_prompt(filename, "image")
        }
    ]

    for img in images:
        b64 = image_to_base64(img)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{b64}"
            }
        })

    messages = [
        {
            "role": "system",
            "content": "Ты аккуратный парсер документов. Всегда возвращай только валидный JSON."
        },
        {
            "role": "user",
            "content": content
        }
    ]
    return call_openrouter(messages)


def local_fraud_checks(record: Dict[str, Any]) -> Dict[str, Any]:
    score = int(record.get("fraud_score") or 0)

    flags = record.get("fraud_flags", [])
    if isinstance(flags, str):
        flags = [x.strip() for x in flags.split(",") if x.strip()]
    if not isinstance(flags, list):
        flags = []

    departure_dt = parse_date_safe(record.get("departure_date"))
    arrival_dt = parse_date_safe(record.get("arrival_date"))
    payment_dt = parse_date_safe(record.get("payment_date"))
    amount = normalize_amount(record.get("amount"))
    document_type = safe_str(record.get("document_type"))
    provider = safe_str(record.get("provider"))
    route = safe_str(record.get("route"))

    missing_fields = record.get("missing_fields", [])
    if isinstance(missing_fields, str):
        missing_fields = [x.strip() for x in missing_fields.split(",") if x.strip()]
    if not isinstance(missing_fields, list):
        missing_fields = []

    if departure_dt and arrival_dt and departure_dt > arrival_dt:
        score += 25
        flags.append("дата отправления позже даты прибытия")

    if payment_dt and departure_dt and payment_dt > departure_dt:
        score += 10
        flags.append("дата оплаты позже даты отправления")

    if amount is not None and amount <= 0:
        score += 15
        flags.append("некорректная сумма")

    if document_type in ("поезд", "самолет") and not route:
        score += 10
        flags.append("для транспортного документа отсутствует маршрут")

    if amount is not None and not provider:
        score += 10
        flags.append("есть сумма, но отсутствует поставщик или перевозчик")

    if len(missing_fields) >= 5:
        score += 10
        flags.append("слишком много отсутствующих полей")

    score = max(0, min(score, 100))

    if score >= 60:
        review_status = "suspicious"
    elif score >= 30:
        review_status = "warning"
    else:
        review_status = "ok"

    return {
        "fraud_score": score,
        "fraud_flags": sorted(set(flags)),
        "review_status": review_status,
    }


def finalize_record(record: Dict[str, Any], source_file: str) -> Dict[str, Any]:
    normalized = {
        "document_type": safe_str(record.get("document_type")),
        "first_name": safe_str(record.get("first_name")),
        "last_name": safe_str(record.get("last_name")),
        "middle_name": safe_str(record.get("middle_name")),
        "departure_date": normalize_date(record.get("departure_date")),
        "arrival_date": normalize_date(record.get("arrival_date")),
        "amount": normalize_amount(record.get("amount")),
        "currency": safe_str(record.get("currency")),
        "provider": safe_str(record.get("provider")),
        "route": safe_str(record.get("route")),
        "ticket_or_booking_number": safe_str(record.get("ticket_or_booking_number")),
        "payment_date": normalize_date(record.get("payment_date")),
        "missing_fields": record.get("missing_fields", []),
        "fraud_score": int(record.get("fraud_score", 0) or 0),
        "fraud_flags": record.get("fraud_flags", []),
        "review_status": safe_str(record.get("review_status")) or "ok",
        "source_file": source_file,
    }

    auto_missing = []
    for field in [
        "document_type",
        "first_name",
        "last_name",
        "middle_name",
        "departure_date",
        "arrival_date",
        "amount",
        "currency",
        "provider",
        "route",
        "ticket_or_booking_number",
        "payment_date",
    ]:
        value = normalized.get(field)
        if value in ("", None):
            auto_missing.append(field)

    existing_missing = normalized.get("missing_fields", [])
    if not isinstance(existing_missing, list):
        existing_missing = [safe_str(existing_missing)] if existing_missing else []

    normalized["missing_fields"] = sorted(set(existing_missing + auto_missing))

    fraud_checked = local_fraud_checks(normalized)
    normalized["fraud_score"] = fraud_checked["fraud_score"]
    normalized["fraud_flags"] = fraud_checked["fraud_flags"]
    normalized["review_status"] = fraud_checked["review_status"]

    normalized["missing_fields"] = ", ".join(normalized["missing_fields"]) if normalized["missing_fields"] else ""
    normalized["fraud_flags"] = ", ".join(normalized["fraud_flags"]) if normalized["fraud_flags"] else ""

    return normalized


def merge_records(records: List[Dict[str, Any]], source_file: str) -> Dict[str, Any]:
    if not records:
        return finalize_record({}, source_file)

    merged = {
        "document_type": "",
        "first_name": "",
        "last_name": "",
        "middle_name": "",
        "departure_date": "",
        "arrival_date": "",
        "amount": None,
        "currency": "",
        "provider": "",
        "route": "",
        "ticket_or_booking_number": "",
        "payment_date": "",
        "missing_fields": [],
        "fraud_score": 0,
        "fraud_flags": [],
        "review_status": "ok",
    }

    for record in records:
        for field in [
            "document_type",
            "first_name",
            "last_name",
            "middle_name",
            "departure_date",
            "arrival_date",
            "currency",
            "provider",
            "route",
            "ticket_or_booking_number",
            "payment_date",
        ]:
            if not merged[field] and record.get(field):
                merged[field] = record.get(field)

        if merged["amount"] in (None, "", 0) and record.get("amount") not in (None, ""):
            merged["amount"] = record.get("amount")

        missing = record.get("missing_fields", [])
        if isinstance(missing, str):
            missing = [x.strip() for x in missing.split(",") if x.strip()]
        if isinstance(missing, list):
            merged["missing_fields"].extend(missing)

        flags = record.get("fraud_flags", [])
        if isinstance(flags, str):
            flags = [x.strip() for x in flags.split(",") if x.strip()]
        if isinstance(flags, list):
            merged["fraud_flags"].extend(flags)

        try:
            merged["fraud_score"] = max(merged["fraud_score"], int(record.get("fraud_score", 0) or 0))
        except Exception:
            pass

    return finalize_record(merged, source_file)


def process_uploaded_file(uploaded_file) -> Dict[str, Any]:
    file_name = uploaded_file.name
    file_type = uploaded_file.type
    file_bytes = uploaded_file.getvalue()

    if file_type == "application/pdf":
        extracted_text = extract_text_from_pdf(file_bytes, max_pages=None)

        if quality_of_text(extracted_text):
            parsed = call_openrouter_with_text(extracted_text, file_name)
            return finalize_record(parsed, file_name)

        images = render_pdf_pages(file_bytes, max_pages=None)
        image_batches = chunk_list(images, chunk_size=4)

        partial_records = []
        for batch_index, image_batch in enumerate(image_batches, start=1):
            parsed = call_openrouter_with_images(image_batch, f"{file_name} [batch {batch_index}]")
            partial_records.append(parsed)

        return merge_records(partial_records, file_name)

    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    parsed = call_openrouter_with_images([image], file_name)
    return finalize_record(parsed, file_name)


# =========================
# PDF REPORT
# =========================
def register_pdf_fonts():
    font_dir = "assets/fonts"
    regular_path = os.path.join(font_dir, "DejaVuSans.ttf")
    bold_path = os.path.join(font_dir, "DejaVuSans-Bold.ttf")

    if not os.path.exists(regular_path):
        raise FileNotFoundError(
            f"Не найден файл шрифта: {regular_path}"
        )

    pdfmetrics.registerFont(TTFont("DejaVuSans", regular_path))

    if os.path.exists(bold_path):
        pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", bold_path))
    else:
        pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", regular_path))

def dataframe_for_report(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    columns_order = [
        "source_file",
        "document_type",
        "last_name",
        "first_name",
        "middle_name",
        "departure_date",
        "arrival_date",
        "payment_date",
        "provider",
        "route",
        "ticket_or_booking_number",
        "amount",
        "currency",
        "review_status",
        "fraud_score",
        "fraud_flags",
        "missing_fields",
    ]
    for col in columns_order:
        if col not in result.columns:
            result[col] = ""
    return result[columns_order]


def build_pdf_report(df: pd.DataFrame) -> bytes:
    register_pdf_fonts()

    normal_font = "DejaVuSans"
    bold_font = "DejaVuSans-Bold"

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        fontName=bold_font,
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#10233d"),
        alignment=TA_LEFT,
        spaceAfter=8,
    )
    heading_style = ParagraphStyle(
        "HeadingCustom",
        parent=styles["Heading2"],
        fontName=bold_font,
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#17365d"),
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyCustom",
        parent=styles["BodyText"],
        fontName=normal_font,
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#222222"),
    )
    small_style = ParagraphStyle(
        "SmallCustom",
        parent=styles["BodyText"],
        fontName=normal_font,
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#4a5568"),
    )

    story = []

    total_amount = pd.to_numeric(df["amount"], errors="coerce").fillna(0).sum()
    suspicious_count = int((df["review_status"] == "suspicious").sum())
    warning_count = int((df["review_status"] == "warning").sum())

    story.append(Paragraph("Отчет по расходам", title_style))
    story.append(Paragraph(
        f"Дата формирования: {datetime.now().strftime('%Y-%m-%d %H:%M')}<br/>"
        f"Документов: {len(df)}<br/>"
        f"Общая сумма: {total_amount:,.2f}".replace(",", " "),
        body_style
    ))
    story.append(Spacer(1, 8))

    summary_data = [
        ["Показатель", "Значение"],
        ["Всего документов", str(len(df))],
        ["Общая сумма", f"{total_amount:,.2f}".replace(",", " ")],
        ["Подозрительных", str(suspicious_count)],
        ["С предупреждением", str(warning_count)],
    ]
    summary_table = Table(summary_data, colWidths=[70 * mm, 90 * mm])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365d")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), bold_font),
        ("FONTNAME", (0, 1), (-1, -1), normal_font),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#b7c6d9")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f7fbff")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Детализация документов", heading_style))

    table_data = [[
        "Файл", "Тип", "ФИО", "Даты", "Сумма", "Поставщик", "Статус"
    ]]

    for _, row in df.iterrows():
        fio = " ".join(filter(None, [
            safe_str(row.get("last_name")),
            safe_str(row.get("first_name")),
            safe_str(row.get("middle_name")),
        ]))
        dates = f"{safe_str(row.get('departure_date'))} - {safe_str(row.get('arrival_date'))}"
        amount_str = ""
        if pd.notna(row.get("amount")):
            amount_str = f"{float(row.get('amount')):,.2f} {safe_str(row.get('currency'))}".replace(",", " ")

        status = safe_str(row.get("review_status"))
        score = safe_str(row.get("fraud_score"))
        status_text = f"{status} ({score})"

        table_data.append([
            Paragraph(safe_str(row.get("source_file")), small_style),
            Paragraph(safe_str(row.get("document_type")), small_style),
            Paragraph(fio, small_style),
            Paragraph(dates, small_style),
            Paragraph(amount_str, small_style),
            Paragraph(safe_str(row.get("provider")), small_style),
            Paragraph(status_text, small_style),
        ])

    detail_table = Table(
        table_data,
        colWidths=[35 * mm, 18 * mm, 34 * mm, 28 * mm, 22 * mm, 28 * mm, 20 * mm]
    )

    table_style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365d")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), bold_font),
        ("FONTNAME", (0, 1), (-1, -1), normal_font),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c8d4e3")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]

    for row_idx in range(1, len(table_data)):
        status_value = safe_str(df.iloc[row_idx - 1]["review_status"])
        if status_value == "suspicious":
            bg = colors.HexColor("#ffe5e5")
        elif status_value == "warning":
            bg = colors.HexColor("#fff6dd")
        else:
            bg = colors.HexColor("#f8fbff")
        table_style.append(("BACKGROUND", (0, row_idx), (-1, row_idx), bg))

    detail_table.setStyle(TableStyle(table_style))
    story.append(detail_table)
    story.append(Spacer(1, 12))

    suspicious_df = df[df["review_status"].isin(["warning", "suspicious"])]
    if not suspicious_df.empty:
        story.append(Paragraph("Документы, требующие проверки", heading_style))
        for _, row in suspicious_df.iterrows():
            fraud_flags = safe_str(row.get("fraud_flags"))
            missing = safe_str(row.get("missing_fields"))
            text = (
                f"<b>Файл:</b> {safe_str(row.get('source_file'))}<br/>"
                f"<b>Статус:</b> {safe_str(row.get('review_status'))}<br/>"
                f"<b>Fraud score:</b> {safe_str(row.get('fraud_score'))}<br/>"
                f"<b>Подозрительные признаки:</b> {fraud_flags or '—'}<br/>"
                f"<b>Отсутствующие поля:</b> {missing or '—'}"
            )
            story.append(Paragraph(text, body_style))
            story.append(Spacer(1, 6))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("⚙️ Настройки")
    st.write("**API:** `OpenRouter`")
    st.write(f"**Model:** `{OPENROUTER_MODEL}`")
    st.write(f"**API key loaded:** `{'yes' if OPENROUTER_API_KEY else 'no'}`")
    st.write("Обработка PDF: сначала текст, затем vision fallback.")
    st.write("Проверка подделки: оценка риска, а не окончательный forensic-вердикт.")
    st.markdown('</div>', unsafe_allow_html=True)


# =========================
# UPLOAD SECTION
# =========================
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
uploaded_files = st.file_uploader(
    "Загрузи документы",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
)
process_btn = st.button("Распознать документы")
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_files:
    st.markdown("### Предпросмотр")
    preview_cols = st.columns(min(3, len(uploaded_files)))
    for idx, uploaded_file in enumerate(uploaded_files):
        with preview_cols[idx % len(preview_cols)]:
            st.write(f"**{uploaded_file.name}**")
            file_bytes = uploaded_file.getvalue()
            if uploaded_file.type == "application/pdf":
                try:
                    preview_images = render_pdf_pages(file_bytes, max_pages=1)
                    if preview_images:
                        st.image(preview_images[0], use_container_width=True)
                except Exception:
                    st.caption("Не удалось показать PDF preview")
            else:
                st.image(file_bytes, use_container_width=True)


if process_btn and uploaded_files:
    st.session_state.documents = []
    progress = st.progress(0)
    status = st.empty()

    for idx, uploaded_file in enumerate(uploaded_files, start=1):
        try:
            status.info(f"Обработка: {uploaded_file.name}")
            record = process_uploaded_file(uploaded_file)
            st.session_state.documents.append(record)
        except Exception as e:
            st.session_state.documents.append({
                "document_type": "",
                "first_name": "",
                "last_name": "",
                "middle_name": "",
                "departure_date": "",
                "arrival_date": "",
                "amount": None,
                "currency": "",
                "provider": "",
                "route": "",
                "ticket_or_booking_number": "",
                "payment_date": "",
                "missing_fields": f"processing_error: {str(e)}",
                "fraud_score": 0,
                "fraud_flags": f"processing_error: {str(e)}",
                "review_status": "warning",
                "source_file": uploaded_file.name,
            })

        progress.progress(idx / len(uploaded_files))

    status.success("Обработка завершена")


# =========================
# DATA REVIEW
# =========================
if st.session_state.documents:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📋 Результаты распознавания и ручная корректировка")

    df = pd.DataFrame(st.session_state.documents)

    for col in REQUIRED_FIELDS:
        if col not in df.columns:
            df[col] = ""

    edited_df = st.data_editor(
        df,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "document_type": st.column_config.SelectboxColumn(
                "Тип документа",
                options=["", *DOCUMENT_TYPES],
                required=False,
            ),
            "first_name": st.column_config.TextColumn("Имя"),
            "last_name": st.column_config.TextColumn("Фамилия"),
            "middle_name": st.column_config.TextColumn("Отчество"),
            "departure_date": st.column_config.TextColumn("Дата отправления"),
            "arrival_date": st.column_config.TextColumn("Дата прибытия"),
            "amount": st.column_config.NumberColumn("Сумма", format="%.2f"),
            "currency": st.column_config.TextColumn("Валюта"),
            "provider": st.column_config.TextColumn("Поставщик / перевозчик / отель"),
            "route": st.column_config.TextColumn("Маршрут"),
            "ticket_or_booking_number": st.column_config.TextColumn("Номер билета / брони"),
            "payment_date": st.column_config.TextColumn("Дата оплаты"),
            "missing_fields": st.column_config.TextColumn("Отсутствующие поля"),
            "fraud_score": st.column_config.NumberColumn("Fraud score", format="%d"),
            "fraud_flags": st.column_config.TextColumn("Подозрительные признаки"),
            "review_status": st.column_config.SelectboxColumn(
                "Статус проверки",
                options=["ok", "warning", "suspicious"],
                required=False,
            ),
            "source_file": st.column_config.TextColumn("Файл"),
        },
        key="editor",
    )

    st.session_state.documents = edited_df.to_dict(orient="records")
    st.markdown('</div>', unsafe_allow_html=True)

    suspicious_docs = edited_df[edited_df["review_status"].isin(["warning", "suspicious"])]

    if not suspicious_docs.empty:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🚨 Документы, требующие проверки")
        for _, row in suspicious_docs.iterrows():
            level = row["review_status"]
            msg = (
                f"**{row['source_file']}** — статус: **{level}**, "
                f"fraud score: **{row['fraud_score']}**  \n"
                f"Причины: {row['fraud_flags'] or 'не указаны'}"
            )
            if level == "suspicious":
                st.error(msg, icon="🚨")
            else:
                st.warning(msg, icon="⚠️")
        st.markdown('</div>', unsafe_allow_html=True)

    clean_amounts = pd.to_numeric(edited_df["amount"], errors="coerce")
    total_amount = clean_amounts.fillna(0).sum()
    docs_count = len(edited_df)
    missing_count = (edited_df["missing_fields"].fillna("").astype(str).str.len() > 0).sum()
    suspicious_count = int((edited_df["review_status"] == "suspicious").sum())
    warning_count = int((edited_df["review_status"] == "warning").sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Документов", docs_count)
    c2.metric("Общая сумма", f"{total_amount:,.2f}".replace(",", " "))
    c3.metric("С пропусками", int(missing_count))
    c4.metric("Подозрительных", suspicious_count)
    c5.metric("С предупреждением", warning_count)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📦 Экспорт")

    report_df = dataframe_for_report(edited_df)
    pdf_bytes = build_pdf_report(report_df)

    file_name = f"expense_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    st.download_button(
        label="Скачать PDF-отчет",
        data=pdf_bytes,
        file_name=file_name,
        mime="application/pdf",
    )

    st.caption("Для корректного русского текста в PDF положи DejaVuSans.ttf и DejaVuSans-Bold.ttf в assets/fonts/")
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("Загрузи документы и нажми «Распознать документы».")