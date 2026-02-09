# Quick Start Commands

Шпаргалка для быстрого старта работы с проектом.

---

## 📦 First Time Setup

```bash
# 1. Перейти в проект
cd /Users/artemshishkin/personal-ai-workspace/projects/saliency-test

# 2. Установить зависимости
pip install -r requirements.txt

# 3. Настроить API ключ
cp config.example.py config.py
# Отредактировать config.py - добавить OpenAI API key
```

---

## 🌐 Web UI (рекомендуемый способ)

```bash
streamlit run app.py
# Откроется http://localhost:8501
# Загрузите изображение → Анализировать → Скачать PDF
```

---

## 🚀 CLI Analysis

### Анализ одного креатива
```bash
python analyze_creative_final.py data/yandex_pay.png
```

### Batch анализ
```bash
for img in data/*.png; do
  python analyze_creative_final.py "$img"
done
```

---

## 📊 Check Results

```bash
# Посмотреть JSON результат
cat yandex_pay_final.json | jq .

# Открыть визуализацию
open yandex_pay_final.jpg

# Посмотреть примеры
open examples/yandex_pay_final.jpg
```

---

## 📖 Read Documentation

```bash
# Quick reference (1 мин)
cat STATUS.md

# User documentation (5 мин)
cat README.md

# AI context / full documentation (10 мин)
cat CLAUDE.md
```

---

## 🧪 Test on Sample Data

```bash
# Test 1: Yandex Pay (expected score: 3.4)
python analyze_creative_final.py data/yandex_pay.png

# Test 2: Yandex Market (expected score: 3.4)
python analyze_creative_final.py data/yandex_market.png
```

---

## 🔧 Development

### Check project status
```bash
ls -lh *.md
```

### Update dependencies
```bash
pip install -r requirements.txt --upgrade
```

### Clean results
```bash
rm -f *_final.json *_final.jpg
```

---

## 📝 Common Issues

### "config.py not found"
```bash
cp config.example.py config.py
# Отредактируй config.py
```

### "ModuleNotFoundError: No module named 'easyocr'"
```bash
pip install -r requirements.txt
```

### Медленная первая загрузка
- Норма: DeepGaze загружает ~500MB
- EasyOCR загружает языковые модели
- Последующие запуски быстрее (cache)

---

## 🎯 Key Files

| File | Purpose | Size |
|------|---------|------|
| `analyze_creative_final.py` | Main pipeline | 43K |
| `app.py` | Web UI (Streamlit) | 15K |
| `STATUS.md` | Quick status | 4K |
| `CLAUDE.md` | AI context | 20K |
| `README.md` | User docs | 9K |
| `config.py` | API keys | - |
| `requirements.txt` | Dependencies | 440B |

---

## 💡 Quick Tips

1. **First run slow?** Норма. Модели кэшируются.
2. **API costs?** ~$0.03-0.07 per image.
3. **Best results?** Креативы 600-1000px width.
4. **Legal text?** Автоматически фильтруется.
5. **CTA not found?** Может быть norm для баннеров.

---

**Last Updated:** 2026-02-09
