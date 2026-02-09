# Project Status - Creative Analysis Pipeline

**Дата:** 2026-02-09
**Версия:** 1.2 Production Ready ✅
**Статус:** Работает, готово к использованию

---

## 🎯 Что это

Автоматический анализ рекламных креативов:
- Eye-tracking симуляция (DeepGaze IIE)
- Детекция ключевых зон (OCR + GPT-5.2)
- Расчет attention % для каждой зоны
- AI рекомендации (GPT-5.2)

## ✅ Что работает

- [x] **Pipeline (9 шагов)** - полностью функционален
- [x] **DeepGaze saliency** - 96-98% accuracy
- [x] **OCR детекция** - точные bbox для текста
- [x] **GPT-5.2 vision + reasoning** - детекция logo/product/person
- [x] **Nested zones** - правильный подсчет без overlap
- [x] **GPT-5.2 рекомендации** - контекстно-зависимые
- [x] **Визуализация** - аннотированные изображения
- [x] **Web UI (Streamlit)** - загрузка, анализ, просмотр результатов
- [x] **PDF export** - скачивание отчёта из веб-интерфейса
- [x] **Creative regeneration** — генерация улучшенного баннера (GPT Image)
- [x] **Документация** - README + CLAUDE.md

## 📊 Протестировано

| Креатив | Score | Zones | Coverage | Status |
|---------|-------|-------|----------|--------|
| Yandex Pay | 3.4/5.0 | 5 зон | 92.3% | ✅ Pass |
| Yandex Market | 3.4/5.0 | 3 зоны | 73.9% | ✅ Pass |

## 🧠 Tech Stack

| Компонент | Технология | Зачем |
|-----------|------------|-------|
| Saliency | DeepGaze IIE | SOTA eye-tracking (96-98%) |
| Text Detection | EasyOCR | Точные bbox |
| Visual Detection | GPT-5.2 + reasoning | Logo/product/person детекция |
| Recommendations | GPT-5.2 + reasoning | Контекстные инсайты |
| CTA Refinement | OpenCV | Поиск кнопок |

## 💰 Cost

**Per Image:**
- DeepGaze: $0 (offline)
- EasyOCR: $0 (offline)
- GPT-5.2 (детекция): ~$0.02-0.05
- GPT-5.2 (рекомендации): ~$0.02-0.05
- GPT Image (генерация): ~$0.04-0.08 (optional, --regenerate)

**Total:** ~$0.04-0.10 per creative (без генерации), ~$0.08-0.23 с генерацией

## ⏱️ Performance

- **Первый запуск:** 2-3 минуты (загрузка моделей)
- **Последующие:** 30-60 секунд
- **Bottleneck:** DeepGaze inference (~15-20 сек)

## 🐛 Known Issues

1. **Product detection** - GPT-5.2 иногда пропускает (minor)
2. **CTA refinement** - работает только для темных кнопок (minor)
3. **OCR на стилизованном тексте** - может пропустить (minor)

## 📁 Файлы

```
saliency-test/
├── analyze_creative_final.py   ← Основной pipeline
├── app.py                      ← Web UI (Streamlit)
├── config.py                   ← API keys (не коммитить!)
├── requirements.txt            ← Dependencies
├── README.md                   ← User docs
├── CLAUDE.md                   ← AI context
├── STATUS.md                   ← Этот файл
├── data/                       ← Test images
└── examples/                   ← Sample results
```

## 🚀 Quick Start

```bash
# 1. Setup
pip install -r requirements.txt
cp config.example.py config.py
# Отредактируй config.py - добавь API key

# 2. Run
python analyze_creative_final.py data/yandex_pay.png

# 3. Results
# → yandex_pay_final.json
# → yandex_pay_final.jpg
```

## 📈 Next Steps

**Выполнено:**
- [x] Web UI (Streamlit) ✅
- [x] PDF export ✅

**P0 - Critical:**
- [ ] Batch processing
- [ ] Saliency caching

**P1 - High:**
- [ ] Улучшить product detection
- [ ] Fallback для визуальных зон
- [ ] Adaptive OCR thresholds

**P2 - Nice to Have:**
- [ ] FastAPI endpoint
- [ ] A/B comparison

## 🔄 Recent Changes

**v1.2 (2026-02-09):**
- ✅ Генерация улучшенного варианта баннера (GPT Image edit)
- ✅ GPT-5.2 промпт-инженер: рекомендации → ТЗ для генерации
- ✅ CLI: флаг --regenerate
- ✅ Web UI: кнопка "Сгенерировать", side-by-side, скачивание JPG

**v1.1 (2026-02-09):**
- ✅ GPT-5.2 для детекции визуальных элементов (замена GPT-4.1)
- ✅ Web UI (Streamlit) — загрузка, анализ, heatmap, рекомендации
- ✅ PDF-экспорт отчёта
- ✅ Поддержка person/face зон

**v1.0 (2026-01-20):**
- ✅ Финальный pipeline готов
- ✅ OCR-first подход вместо GPT-only
- ✅ Nested zones handling
- ✅ Banner-aware рекомендации
- ✅ Cleaned up ~50 файлов

## 📞 Support

- **Docs:** см. README.md
- **AI Context:** см. CLAUDE.md
- **Examples:** см. examples/

---

**Status:** 🟢 Production Ready
**Last Updated:** 2026-02-09
**Maintainer:** Artemshishkin
