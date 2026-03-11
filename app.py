import base64
import io
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF
import pandas as pd
import requests
import streamlit as st
from dateutil import parser as date_parser
from PIL import Image


def load_css(file_name: str):
    with open(file_name, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("assets/style.css")

# =========================
# CONFIG
# =========================
LM_STUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"
MODEL_NAME = "qwen/qwen3-vl-8b"

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
    "source_file",
]

# =========================
# UI / PAGE
# =========================
st.set_page_config(
    page_title="Expense Parser MVP",
    page_icon="🧾",
    layout="wide",
)

# Liquid glass style
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

st.title("🧾 Expense Parser MVP")
st.caption("Разбор билетов, чеков, бронирований и документов на проживание через Streamlit + LM Studio + VLM")

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


def render_pdf_pages(pdf_bytes: bytes, max_pages: int = 3) -> List[Image.Image]:
    images = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for page_num in range(min(len(doc), max_pages)):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    return images


def extract_text_from_pdf(pdf_bytes: bytes, max_pages: int = 5) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    texts = []
    for page_num in range(min(len(doc), max_pages)):
        page = doc[page_num]
        txt = page.get_text("text")
        if txt:
            texts.append(txt.strip())
    return "\n\n".join([t for t in texts if t])


def quality_of_text(text: str) -> bool:
    """
    Простая эвристика:
    если текста мало или он слишком 'грязный', считаем недостаточным.
    """
    if not text:
        return False
    stripped = text.strip()
    if len(stripped) < 80:
        return False
    letters = re.findall(r"[A-Za-zА-Яа-я0-9]", stripped)
    return len(letters) > 50


def clean_model_json(raw_content: str) -> Dict[str, Any]:
    """
    Пытаемся аккуратно вытащить JSON из ответа модели.
    """
    raw_content = raw_content.strip()

    # Если модель вернула JSON в markdown-блоке
    code_block_match = re.search(r"```json\s*(.*?)\s*```", raw_content, flags=re.DOTALL)
    if code_block_match:
        raw_content = code_block_match.group(1).strip()

    # Попытка найти JSON-объект
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
            "_raw_model_response": raw_content,
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

    # Объединяем missing_fields от модели и вычисленные
    existing_missing = normalized.get("missing_fields", [])
    if not isinstance(existing_missing, list):
        existing_missing = [safe_str(existing_missing)] if existing_missing else []

    merged_missing = sorted(set(existing_missing + auto_missing))
    normalized["missing_fields"] = ", ".join(merged_missing) if merged_missing else ""

    return normalized


def call_lmstudio_with_text(text: str, filename: str) -> Dict[str, Any]:
    prompt = f"""
Ты — система извлечения данных из транспортных и гостиничных документов для бухгалтерии.

Нужно извлечь данные только из предоставленного текста документа.
НЕЛЬЗЯ придумывать значения.
Если поля нет — оставь пустую строку или null, а название поля добавь в массив missing_fields.

Типы документов:
- поезд
- самолет
- такси
- проживание

Верни строго JSON без пояснений в следующем формате:
{{
  "document_type": "",
  "first_name": "",
  "last_name": "",
  "middle_name": "",
  "departure_date": "",
  "arrival_date": "",
  "amount": null,
  "currency": "",
  "provider": "",
  "route": "",
  "ticket_or_booking_number": "",
  "payment_date": "",
  "missing_fields": []
}}

Правила:
- document_type: только одно из значений: поезд, самолет, такси, проживание
- даты желательно вернуть в том виде, как есть в документе, позже они будут нормализованы
- amount: только число, без валюты
- route: маршрут, если он указан
- provider: перевозчик, агрегатор, отель или сервис
- ticket_or_booking_number: номер билета, заказа или брони
- missing_fields: список реально отсутствующих или нераспознанных полей

Имя файла: {filename}

Текст документа:
{text}
""".strip()

    payload = {
        "model": MODEL_NAME,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": "Ты аккуратный парсер документов. Всегда возвращай только валидный JSON."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    response = requests.post(LM_STUDIO_URL, json=payload, timeout=180)
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    return clean_model_json(content)


def call_lmstudio_with_images(images: List[Image.Image], filename: str) -> Dict[str, Any]:
    content = [
        {
            "type": "text",
            "text": f"""
Ты — система извлечения данных из транспортных и гостиничных документов для бухгалтерии.

Проанализируй изображения документа и извлеки данные.
НЕЛЬЗЯ придумывать значения.
Если поля нет — оставь пустую строку или null, а название поля добавь в missing_fields.

Верни строго JSON без пояснений в формате:
{{
  "document_type": "",
  "first_name": "",
  "last_name": "",
  "middle_name": "",
  "departure_date": "",
  "arrival_date": "",
  "amount": null,
  "currency": "",
  "provider": "",
  "route": "",
  "ticket_or_booking_number": "",
  "payment_date": "",
  "missing_fields": []
}}

Правила:
- document_type: только одно из значений: поезд, самолет, такси, проживание
- amount: только число, без валюты
- missing_fields: список реально отсутствующих или нераспознанных полей

Имя файла: {filename}
""".strip()
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

    payload = {
        "model": MODEL_NAME,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": "Ты аккуратный парсер документов. Всегда возвращай только валидный JSON."
            },
            {
                "role": "user",
                "content": content
            }
        ]
    }

    response = requests.post(LM_STUDIO_URL, json=payload, timeout=240)
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    return clean_model_json(content)


def process_uploaded_file(uploaded_file) -> Dict[str, Any]:
    file_name = uploaded_file.name
    file_type = uploaded_file.type
    file_bytes = uploaded_file.read()

    if file_type == "application/pdf":
        extracted_text = extract_text_from_pdf(file_bytes)
        if quality_of_text(extracted_text):
            parsed = call_lmstudio_with_text(extracted_text, file_name)
            return finalize_record(parsed, file_name)
        else:
            images = render_pdf_pages(file_bytes, max_pages=3)
            parsed = call_lmstudio_with_images(images, file_name)
            return finalize_record(parsed, file_name)

    else:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        parsed = call_lmstudio_with_images([image], file_name)
        return finalize_record(parsed, file_name)


def dataframe_for_export(df: pd.DataFrame) -> pd.DataFrame:
    export_df = df.copy()
    # удобнее для бухгалтерии
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
        "missing_fields",
    ]
    for col in columns_order:
        if col not in export_df.columns:
            export_df[col] = ""
    export_df = export_df[columns_order]
    return export_df


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        export_df = dataframe_for_export(df)
        export_df.to_excel(writer, index=False, sheet_name="expenses")

        ws = writer.sheets["expenses"]
        for column_cells in ws.columns:
            max_len = 0
            col_letter = column_cells[0].column_letter
            for cell in column_cells:
                try:
                    value_len = len(str(cell.value)) if cell.value is not None else 0
                    max_len = max(max_len, value_len)
                except Exception:
                    pass
            ws.column_dimensions[col_letter].width = min(max_len + 2, 40)

    output.seek(0)
    return output.getvalue()

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("⚙️ Настройки")
    st.write(f"**LM Studio URL:** `{LM_STUDIO_URL}`")
    st.write(f"**Model:** `{MODEL_NAME}`")
    st.write("Обработка PDF: сначала текст, затем VLM fallback.")
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

if process_btn and uploaded_files:
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
            "source_file": st.column_config.TextColumn("Файл"),
        },
        key="editor",
    )

    st.session_state.documents = edited_df.to_dict(orient="records")
    st.markdown('</div>', unsafe_allow_html=True)

    # summary
    clean_amounts = pd.to_numeric(edited_df["amount"], errors="coerce")
    total_amount = clean_amounts.fillna(0).sum()
    docs_count = len(edited_df)
    missing_count = (edited_df["missing_fields"].fillna("").astype(str).str.len() > 0).sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Документов", docs_count)
    c2.metric("Общая сумма", f"{total_amount:,.2f}".replace(",", " "))
    c3.metric("Документов с пропусками", int(missing_count))

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📦 Экспорт")
    excel_bytes = to_excel_bytes(edited_df)

    file_name = f"expense_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    st.download_button(
        label="Скачать Excel-отчет",
        data=excel_bytes,
        file_name=file_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("Загрузи документы и нажми «Распознать документы».")