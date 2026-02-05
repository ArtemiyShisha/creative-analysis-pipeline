"""
Streamlit Web UI for Creative Analysis Pipeline

Launch with: streamlit run app.py
Then open the URL shown in terminal (usually http://localhost:8501)
"""

import os
import io
import streamlit as st
import pandas as pd
from fpdf import FPDF
from analyze_creative_final import analyze_creative_final


def generate_pdf_report(results, heatmap_path):
    """Generate simple PDF report with analysis results"""
    
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Use built-in font only (no Unicode issues)
    pdf.set_font('helvetica', 'B', 18)
    pdf.cell(0, 12, 'Creative Analysis Report', ln=True, align='C')
    pdf.ln(8)
    
    # Overall Score
    score = results.get('overall_score', 0)
    pdf.set_font('helvetica', 'B', 14)
    pdf.cell(0, 8, f'Score: {score}/5.0', ln=True)
    pdf.ln(4)
    
    # Reasoning (transliterated)
    pdf.set_font('helvetica', '', 10)
    reasoning = transliterate_text(results.get('reasoning', '')[:400])
    pdf.multi_cell(0, 5, reasoning)
    pdf.ln(6)
    
    # Attention Distribution (simple list, no table)
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 8, 'Attention Distribution:', ln=True)
    pdf.ln(2)
    
    pdf.set_font('helvetica', '', 10)
    for zone in results.get('zones', [])[:10]:
        zone_type = zone.get('type', 'unknown')
        attention = zone.get('attention_pct', 0)
        label = transliterate_text(zone.get('label', '')[:40])
        line = f"  - {zone_type}: {attention:.1f}% ({label})"
        pdf.cell(0, 6, line, ln=True)
    
    pdf.ln(2)
    total = results.get('total_zones_attention', 0)
    pdf.set_font('helvetica', 'B', 10)
    pdf.cell(0, 6, f'Total coverage: {total:.1f}%', ln=True)
    pdf.ln(6)
    
    # Heatmap image
    if os.path.exists(heatmap_path):
        pdf.set_font('helvetica', 'B', 12)
        pdf.cell(0, 8, 'Attention Heatmap:', ln=True)
        pdf.ln(2)
        try:
            pdf.image(heatmap_path, x=15, w=180)
        except Exception:
            pdf.cell(0, 6, '(Image not available)', ln=True)
        pdf.ln(4)
    
    # Recommendations
    pdf.add_page()
    pdf.set_font('helvetica', 'B', 14)
    pdf.cell(0, 10, 'Recommendations', ln=True)
    pdf.ln(4)
    
    for i, rec in enumerate(results.get('recommendations', [])[:5], 1):
        priority = rec.get('priority', 'Medium')
        title = transliterate_text(rec.get('title', '')[:80])
        desc = transliterate_text(rec.get('description', '')[:300])
        impact = transliterate_text(rec.get('expected_impact', '')[:150])
        
        pdf.set_font('helvetica', 'B', 10)
        pdf.multi_cell(0, 5, f"{i}. [{priority}] {title}")
        
        pdf.set_font('helvetica', '', 9)
        pdf.multi_cell(0, 4, desc)
        
        if impact:
            pdf.set_font('helvetica', 'I', 8)
            pdf.multi_cell(0, 4, f"Impact: {impact}")
        
        pdf.ln(3)
    
    return bytes(pdf.output())


def transliterate_text(text):
    """Simple transliteration for non-Unicode fonts"""
    translit_map = {
        'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'e',
        'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm',
        'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u',
        'ф': 'f', 'х': 'h', 'ц': 'ts', 'ч': 'ch', 'ш': 'sh', 'щ': 'sch',
        'ъ': '', 'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya',
        'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D', 'Е': 'E', 'Ё': 'E',
        'Ж': 'Zh', 'З': 'Z', 'И': 'I', 'Й': 'Y', 'К': 'K', 'Л': 'L', 'М': 'M',
        'Н': 'N', 'О': 'O', 'П': 'P', 'Р': 'R', 'С': 'S', 'Т': 'T', 'У': 'U',
        'Ф': 'F', 'Х': 'H', 'Ц': 'Ts', 'Ч': 'Ch', 'Ш': 'Sh', 'Щ': 'Sch',
        'Ъ': '', 'Ы': 'Y', 'Ь': '', 'Э': 'E', 'Ю': 'Yu', 'Я': 'Ya'
    }
    return ''.join(translit_map.get(c, c) for c in text)

# Page config
st.set_page_config(
    page_title="Анализ креатива",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Minimal white theme CSS
st.markdown("""
<style>
    /* Clean white design */
    .stApp {
        background-color: #ffffff;
    }

    /* Headers - simple and clean */
    h1 {
        color: #000000 !important;
        font-size: 2.5rem !important;
        font-weight: 600 !important;
        margin-bottom: 0.5rem !important;
        letter-spacing: -0.02em;
    }

    h2, h3 {
        color: #000000 !important;
        font-weight: 500 !important;
    }

    /* Text */
    .stMarkdown p {
        color: #666666;
        font-size: 1rem;
        line-height: 1.5;
    }

    /* File uploader - minimal border */
    [data-testid="stFileUploader"] {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 2rem;
    }

    /* Primary button - black */
    .stButton > button[kind="primary"] {
        background-color: #000000 !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.6rem 1.5rem !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
    }

    .stButton > button[kind="primary"]:hover {
        background-color: #333333 !important;
    }

    /* Download buttons - outlined */
    .stDownloadButton > button {
        background-color: white !important;
        border: 1px solid #e0e0e0 !important;
        color: #000000 !important;
        border-radius: 6px !important;
        padding: 0.6rem 1.5rem !important;
    }

    .stDownloadButton > button:hover {
        border-color: #000000 !important;
    }

    /* Tables - minimal */
    [data-testid="stDataFrame"] {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
    }

    /* Images - subtle border */
    [data-testid="stImage"] {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
    }

    /* Progress bar - black */
    .stProgress > div > div {
        background-color: #000000 !important;
    }

    /* Expander - clean */
    .streamlit-expanderHeader {
        border: 1px solid #e0e0e0;
        border-radius: 6px;
    }

    /* Container spacing */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 1100px !important;
    }

    /* Horizontal rule */
    hr {
        border-color: #e0e0e0 !important;
        margin: 2rem 0 !important;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Import and validation
def validate_api_key():
    """Check if OpenAI API key is configured"""
    # Try to get from environment (Streamlit Cloud Secrets)
    api_key = os.environ.get("OPENAI_API_KEY", "")

    # If not in env, try importing from config.py (local development)
    if not api_key:
        try:
            from config import OPENAI_API_KEY
            api_key = OPENAI_API_KEY
        except ImportError:
            pass

    if not api_key or api_key == "":
        return False, "⚠️ Отсутствует API ключ OpenAI.\n\n**Для Streamlit Cloud:** Добавьте `OPENAI_API_KEY` в Settings → Secrets\n\n**Для локальной разработки:** Скопируйте config.example.py в config.py и добавьте API ключ"

    return True, None


def validate_image(image_file):
    """Validate uploaded image"""
    if not image_file:
        return False, "Изображение не загружено"

    if not image_file.name.lower().endswith(('.png', '.jpg', '.jpeg')):
        return False, "Неверный формат. Загрузите PNG или JPG"

    # Check file size (max 10MB)
    if image_file.size > 10 * 1024 * 1024:
        size_mb = image_file.size / (1024 * 1024)
        return False, f"Файл слишком большой ({size_mb:.1f}МБ). Максимум 10МБ"

    return True, None


def format_results(results):
    """Convert analysis results to UI-friendly format"""

    # Score with emoji stars
    score = results['overall_score']
    stars = "⭐" * int(score)
    score_md = f"## Общая оценка: {stars} {score}/5.0\n\n{results['reasoning']}"

    # Zone table (pandas DataFrame)
    zones_df = pd.DataFrame([
        {
            'Тип': z['type'],
            'Текст': z['label'][:40] + ('...' if len(z['label']) > 40 else ''),
            'Внимание %': f"{z['attention_pct']:.1f}%"
        }
        for z in results['zones']
    ])

    # Visualization and heatmap paths
    base_name = os.path.splitext(os.path.basename(results['image']))[0]
    viz_path = f"{base_name}_final.jpg"
    heatmap_path = f"{base_name}_heatmap.jpg"

    # Recommendations in markdown
    recs_md = ""
    priority_emoji = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}
    priority_ru = {'High': 'Высокий', 'Medium': 'Средний', 'Low': 'Низкий'}
    for i, rec in enumerate(results['recommendations'], 1):
        emoji = priority_emoji.get(rec['priority'], '⚪')
        priority_text = priority_ru.get(rec['priority'], rec['priority'])
        recs_md += f"### {i}. {emoji} {priority_text}: {rec['title']}\n\n"
        recs_md += f"{rec['description']}\n\n"
        recs_md += f"**Ожидаемый эффект:** {rec['expected_impact']}\n\n"
        recs_md += "---\n\n"

    # JSON file path
    json_path = f"{base_name}_final.json"

    return score_md, zones_df, viz_path, heatmap_path, recs_md, json_path


# Header
st.markdown("# Анализ креатива")
st.markdown("Автоматический анализ рекламных креативов с помощью AI и симуляции взгляда")

st.markdown("---")

# Validate API key on startup
api_valid, api_msg = validate_api_key()
if not api_valid:
    st.error(api_msg)
    st.stop()

# Input section
st.markdown("---")
uploaded_file = st.file_uploader(
    "Загрузите креатив (PNG/JPG)",
    type=['png', 'jpg', 'jpeg'],
    help="Выберите изображение рекламного креатива"
)

# Display uploaded image preview
if uploaded_file:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(uploaded_file, caption="Загруженное изображение", use_container_width=True)

if st.button("Анализировать", type="primary", use_container_width=True):

    # Validate image
    img_valid, img_msg = validate_image(uploaded_file)
    if not img_valid:
        st.error(f"## ❌ Error\n\n{img_msg}")
        st.stop()

    # Save uploaded file temporarily
    temp_path = f"/tmp/{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        # Progress tracking
        progress_bar = st.progress(0, text="Начинаем анализ...")

        with st.spinner("Генерация карты внимания и анализ зон..."):
            progress_bar.progress(0.05, text="Генерация карты внимания...")

            # Run analysis
            results = analyze_creative_final(temp_path, filter_legal=True)

            progress_bar.progress(1.0, text="✅ Анализ завершен!")

        # Format results
        score_md, zones_df, viz_path, heatmap_path, recs_md, json_path = format_results(results)

        # Clear progress bar
        progress_bar.empty()

        # Display results
        st.markdown("---")

        # Overall Score
        st.markdown(score_md)

        # Zone table
        st.markdown("### Распределение внимания")
        st.dataframe(zones_df, use_container_width=True, hide_index=True)

        # Heatmap visualization
        st.markdown("### Тепловая карта внимания")
        if os.path.exists(heatmap_path):
            st.image(heatmap_path, use_container_width=True)
        else:
            st.warning("Тепловая карта не найдена")

        # Recommendations
        with st.expander("Рекомендации", expanded=True):
            st.markdown(recs_md)

        # Download buttons
        st.markdown("### Скачать результаты")
        col1, col2, col3 = st.columns(3)

        with col1:
            # Generate PDF report
            try:
                pdf_bytes = generate_pdf_report(results, heatmap_path)
                pdf_filename = f"{os.path.splitext(os.path.basename(results['image']))[0]}_report.pdf"
                st.download_button(
                    label="📄 PDF отчёт",
                    data=pdf_bytes,
                    file_name=pdf_filename,
                    mime="application/pdf"
                )
            except Exception as e:
                st.warning(f"PDF недоступен: {str(e)[:50]}")

        with col2:
            if os.path.exists(json_path):
                with open(json_path, "rb") as f:
                    st.download_button(
                        label="JSON",
                        data=f,
                        file_name=json_path,
                        mime="application/json"
                    )

        with col3:
            if os.path.exists(heatmap_path):
                with open(heatmap_path, "rb") as f:
                    st.download_button(
                        label="Тепловая карта",
                        data=f,
                        file_name=heatmap_path,
                        mime="image/jpeg"
                    )

    except Exception as e:
        error_msg = str(e)

        # Check for specific error types
        if "401" in error_msg or "Unauthorized" in error_msg:
            st.error("⚠️ Неверный API ключ OpenAI. Проверьте config.py")
        elif "insufficient_quota" in error_msg.lower() or "billing" in error_msg.lower():
            st.error("⚠️ Недостаточно средств на балансе OpenAI. Пополните баланс на platform.openai.com")
        elif "429" in error_msg or "rate limit" in error_msg.lower():
            st.error("⚠️ Превышен лимит запросов. Попробуйте позже")
        elif "api" in error_msg.lower() or "openai" in error_msg.lower():
            st.error(f"⚠️ Ошибка API: {error_msg}")
        else:
            st.error(f"⚠️ Непредвиденная ошибка: {error_msg}")

    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #999; font-size: 0.875rem;'>Creative Analysis Pipeline</p>", unsafe_allow_html=True)
