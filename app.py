"""
Streamlit Web UI for Creative Analysis Pipeline

Launch with: streamlit run app.py
Then open the URL shown in terminal (usually http://localhost:8501)
"""

import os
import io
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from analyze_creative_final import (
    analyze_creative_final, build_edit_prompt, regenerate_creative,
    generate_saliency_map, detect_text_blocks, group_and_classify_text_zones,
    detect_visual_elements_gpt41, refine_cta_bbox, merge_all_zones,
    calculate_attention, generate_recommendations
)

# Max dimension for uploaded images (prevents OOM on Streamlit Cloud 1GB)
MAX_UPLOAD_DIMENSION = 1024


def generate_pdf_report(results, heatmap_path):
    """Generate PDF report using matplotlib (supports Unicode)"""

    pdf_buffer = io.BytesIO()

    with PdfPages(pdf_buffer) as pdf:
        # Page 1: Score, Reasoning, Attention Table
        fig, axes = plt.subplots(2, 1, figsize=(8.5, 11), gridspec_kw={'height_ratios': [1, 2]})
        fig.suptitle('Анализ креатива', fontsize=18, fontweight='bold', y=0.98)

        # Top section: Score and reasoning
        ax1 = axes[0]
        ax1.axis('off')

        score = results.get('overall_score', 0)
        stars = '*' * int(score) + '-' * (5 - int(score))

        text_content = f"Общая оценка: {stars} {score}/5.0\n\n"
        text_content += results.get('reasoning', '')[:500]

        ax1.text(0.05, 0.95, text_content, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', wrap=True,
                bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

        # Bottom section: Attention table
        ax2 = axes[1]
        ax2.axis('off')
        ax2.set_title('Распределение внимания', fontsize=12, fontweight='bold', loc='left', pad=10)

        zones = results.get('zones', [])[:8]
        if zones:
            table_data = [[z.get('type', ''), z.get('label', '')[:35], f"{z.get('attention_pct', 0):.1f}%"] for z in zones]
            table = ax2.table(cellText=table_data,
                            colLabels=['Тип', 'Текст', 'Внимание'],
                            loc='upper center',
                            cellLoc='left',
                            colWidths=[0.2, 0.55, 0.15])
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.5)

            # Style header
            for i in range(3):
                table[(0, i)].set_facecolor('#e0e0e0')
                table[(0, i)].set_text_props(fontweight='bold')

        total = results.get('total_zones_attention', 0)
        ax2.text(0.05, 0.05, f"Общее покрытие: {total:.1f}%", transform=ax2.transAxes, fontsize=10, fontweight='bold')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Page 2: Heatmap
        if os.path.exists(heatmap_path):
            fig2, ax = plt.subplots(figsize=(8.5, 11))
            ax.set_title('Тепловая карта внимания', fontsize=14, fontweight='bold', pad=20)
            img = mpimg.imread(heatmap_path)
            ax.imshow(img)
            ax.axis('off')
            pdf.savefig(fig2, bbox_inches='tight')
            plt.close(fig2)

        # Page 3: Recommendations
        fig3, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        ax.set_title('Рекомендации', fontsize=14, fontweight='bold', loc='left', y=1.02)

        recs = results.get('recommendations', [])[:5]
        rec_text = ""
        priority_label = {'High': '[ВЫСОКИЙ]', 'Medium': '[СРЕДНИЙ]', 'Low': '[НИЗКИЙ]'}

        for i, rec in enumerate(recs, 1):
            priority = priority_label.get(rec.get('priority', 'Medium'), '[СРЕДНИЙ]')
            rec_text += f"{i}. {priority} {rec.get('title', '')}\n"
            rec_text += f"   {rec.get('description', '')[:200]}\n"
            rec_text += f"   → {rec.get('expected_impact', '')[:100]}\n\n"

        ax.text(0.02, 0.98, rec_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', wrap=True, family='sans-serif')

        pdf.savefig(fig3, bbox_inches='tight')
        plt.close(fig3)

    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()


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

    return score_md, zones_df, heatmap_path, recs_md


# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'temp_path' not in st.session_state:
    st.session_state.temp_path = None
if 'heatmap_path' not in st.session_state:
    st.session_state.heatmap_path = None
if 'improved_bytes' not in st.session_state:
    st.session_state.improved_bytes = None
if 'improved_score' not in st.session_state:
    st.session_state.improved_score = None
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None


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

# Reset state when a different file is uploaded
if uploaded_file and uploaded_file.name != st.session_state.uploaded_file_name:
    st.session_state.results = None
    st.session_state.temp_path = None
    st.session_state.heatmap_path = None
    st.session_state.improved_bytes = None
    st.session_state.improved_score = None
    st.session_state.uploaded_file_name = uploaded_file.name

# Display uploaded image preview
if uploaded_file:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(uploaded_file, caption="Загруженное изображение", width="stretch")

# --- Analyze button ---
if st.button("Анализировать", type="primary", use_container_width=True):

    # Validate image
    img_valid, img_msg = validate_image(uploaded_file)
    if not img_valid:
        st.error(f"## ❌ Error\n\n{img_msg}")
        st.stop()

    # Save uploaded file temporarily, resizing if too large
    temp_path = f"/tmp/{uploaded_file.name}"
    img = Image.open(uploaded_file)
    w, h = img.size
    if max(w, h) > MAX_UPLOAD_DIMENSION:
        scale = MAX_UPLOAD_DIMENSION / max(w, h)
        new_size = (int(w * scale), int(h * scale))
        img = img.resize(new_size, Image.LANCZOS)
        st.info(f"Изображение уменьшено: {w}×{h} → {new_size[0]}×{new_size[1]}")
    img.save(temp_path)
    img.close()

    try:
        # Progress tracking
        progress_bar = st.progress(0, text="Начинаем анализ...")

        with st.spinner("Генерация карты внимания и анализ зон..."):
            progress_bar.progress(0.05, text="Генерация карты внимания...")

            # Run analysis
            results = analyze_creative_final(temp_path, filter_legal=True)

            progress_bar.progress(1.0, text="✅ Анализ завершен!")

        # Save to session state
        score_md, zones_df, heatmap_path, recs_md = format_results(results)
        st.session_state.results = results
        st.session_state.temp_path = temp_path
        st.session_state.heatmap_path = heatmap_path
        st.session_state.improved_bytes = None  # reset regeneration on new analysis
        st.session_state.improved_score = None

        progress_bar.empty()

    except Exception as e:
        error_msg = str(e)

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

# --- Display results from session state (persists across reruns) ---
if st.session_state.results is not None:
    results = st.session_state.results
    heatmap_path = st.session_state.heatmap_path
    temp_path = st.session_state.temp_path

    score_md, zones_df, _, recs_md = format_results(results)

    st.markdown("---")

    # Overall Score
    st.markdown(score_md)

    # Zone table
    st.markdown("### Распределение внимания")
    st.dataframe(zones_df, use_container_width=True, hide_index=True)

    # Heatmap visualization
    st.markdown("### Тепловая карта внимания")
    if heatmap_path and os.path.exists(heatmap_path):
        st.image(heatmap_path, width="stretch")
    else:
        st.warning("Тепловая карта не найдена")

    # Recommendations
    with st.expander("Рекомендации", expanded=True):
        st.markdown(recs_md)

    # Download PDF
    st.markdown("### Скачать отчёт")
    try:
        pdf_bytes = generate_pdf_report(results, heatmap_path)
        pdf_filename = f"{os.path.splitext(os.path.basename(results['image']))[0]}_report.pdf"
        st.download_button(
            label="📄 Скачать PDF отчёт",
            data=pdf_bytes,
            file_name=pdf_filename,
            mime="application/pdf",
            use_container_width=True
        )
    except Exception as e:
        st.error(f"Ошибка генерации PDF: {str(e)}")

    # --- Regeneration section ---
    st.markdown("---")
    st.markdown("### Сгенерировать улучшенный вариант")
    st.caption("AI-концепт для вдохновения. Может содержать неточности в тексте и деталях бренда.")

    if st.button("🎨 Сгенерировать", type="secondary", use_container_width=True):
        try:
            regen_progress = st.progress(0, text="Формирование задания...")

            # Step 10: Build edit prompt
            regen_progress.progress(0.3, text="Формирование задания для генерации...")
            img_for_size = Image.open(temp_path)
            regen_w, regen_h = img_for_size.size
            img_for_size.close()
            edit_data = build_edit_prompt(
                results['zones'],
                results['recommendations'],
                regen_w, regen_h,
                image_path=temp_path
            )

            if edit_data is None:
                regen_progress.empty()
                priorities = [r.get('priority', 'N/A') for r in results['recommendations']]
                st.warning(f"Не удалось сформировать задание для генерации. Приоритеты рекомендаций: {priorities}")
            else:
                # Step 11: Regenerate
                regen_progress.progress(0.5, text="Генерация улучшенного варианта...")
                base_name = os.path.splitext(os.path.basename(temp_path))[0]
                improved_path = f"/tmp/{base_name}_improved.jpg"
                result_path = regenerate_creative(temp_path, edit_data, improved_path)

                if result_path and os.path.exists(result_path):
                    with open(result_path, 'rb') as f:
                        st.session_state.improved_bytes = f.read()

                    # Quick scoring: reuse zones from original, recalculate attention on new saliency
                    regen_progress.progress(0.7, text="Оцениваем улучшенный вариант...")
                    try:
                        orig_zones = [
                            {k: v for k, v in z.items() if k != 'attention_pct'}
                            for z in results['zones']
                        ]
                        _, imp_saliency = generate_saliency_map(result_path)
                        imp_zones, imp_total = calculate_attention(imp_saliency, orig_zones)
                        imp_bg = 100 - imp_total
                        imp_recs = generate_recommendations(
                            imp_zones, imp_total, imp_bg,
                            image_path=result_path
                        )
                        st.session_state.improved_score = imp_recs['overall_score']
                    except Exception as e:
                        print(f"  ⚠️ Could not score improved banner: {e}")
                        st.session_state.improved_score = None

                    regen_progress.progress(1.0, text="✅ Готово!")
                    regen_progress.empty()
                else:
                    st.error("Не удалось сгенерировать — модель отклонила запрос. Используйте рекомендации для ручной доработки.")

        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "rate limit" in error_msg.lower():
                st.error("⚠️ Превышен лимит запросов. Попробуйте позже")
            elif "billing" in error_msg.lower() or "insufficient_quota" in error_msg.lower():
                st.error("⚠️ Недостаточно средств на балансе OpenAI")
            else:
                st.error(f"⚠️ Ошибка генерации: {error_msg}")

    # --- Display regeneration result from session state ---
    if st.session_state.improved_bytes is not None:
        st.markdown("### Сравнение")

        orig_score = results['overall_score']
        improved_score = st.session_state.improved_score

        col1, col2 = st.columns(2)
        with col1:
            score_text = f"**Оригинал** — {orig_score}/5.0"
            st.markdown(score_text)
            st.image(temp_path, width="stretch")
        with col2:
            if improved_score is not None:
                delta = improved_score - orig_score
                delta_icon = "🟢" if delta > 0 else ("🔴" if delta < 0 else "🟡")
                delta_str = f"+{delta:.1f}" if delta > 0 else f"{delta:.1f}"
                score_text = f"**Улучшенный** — {improved_score}/5.0 ({delta_icon} {delta_str})"
            else:
                score_text = "**Улучшенный вариант**"
            st.markdown(score_text)
            st.image(st.session_state.improved_bytes, width="stretch")

        base_name = os.path.splitext(os.path.basename(temp_path))[0]
        st.download_button(
            label="📥 Скачать улучшенный баннер (JPG)",
            data=st.session_state.improved_bytes,
            file_name=f"{base_name}_improved.jpg",
            mime="image/jpeg",
            use_container_width=True
        )

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #999; font-size: 0.875rem;'>Creative Analysis Pipeline</p>", unsafe_allow_html=True)
