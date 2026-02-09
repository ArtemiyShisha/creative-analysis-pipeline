# Установка и запуск Web UI

## Быстрый старт (локально)

### 1. Установите зависимости

```bash
cd /Users/artemshishkin/personal-ai-workspace/projects/saliency-test

# Создайте виртуальное окружение (опционально, но рекомендуется)
python3 -m venv venv
source venv/bin/activate  # На Windows: venv\Scripts\activate

# Установите зависимости
pip install -r requirements.txt
```

### 2. Настройте API ключ

Убедитесь, что в `config.py` есть ваш OpenAI API key:

```python
OPENAI_API_KEY = "sk-proj-..."
```

### 3. Запустите приложение

```bash
streamlit run app.py
```

Откройте URL, который покажет Streamlit в консоли (обычно http://localhost:8501).

## Первый запуск

При первом запуске:
- DeepGaze загрузит модели (~500MB) - займет 2-3 минуты
- EasyOCR загрузит языковые модели
- Последующие запуски будут быстрее (30-60 секунд на анализ)

## Использование

1. Нажмите "Browse files" и выберите PNG/JPG изображение
2. Нажмите "Анализировать"
3. Дождитесь завершения (30-60 сек)
4. Просмотрите результаты:
   - Overall Score с обоснованием
   - Таблица зон с attention %
   - Тепловая карта внимания (heatmap)
   - Рекомендации по улучшению
5. Скачайте PDF-отчёт

## Тестовые изображения

Попробуйте на примерах:
- `data/yandex_pay.png`
- `data/yandex_market.png`

## Troubleshooting

### Ошибка: "Missing OpenAI API key"
Проверьте `config.py` - добавьте ваш API key.

### Ошибка: "ModuleNotFoundError: No module named 'streamlit'"
```bash
pip install -r requirements.txt
# или напрямую:
pip install streamlit pandas
```

### Порт 8501 занят
Streamlit автоматически выберет следующий свободный порт (8502, 8503, и т.д.).
Или укажите порт вручную:
```bash
streamlit run app.py --server.port 8502
```

### Слишком долго при первом запуске
Это нормально - загружаются ML модели. При следующих запусках будет быстрее.

### Streamlit показывает пустую страницу
Попробуйте:
1. Очистить кеш браузера
2. Открыть в режиме инкогнито
3. Перезапустить Streamlit:
```bash
# Остановить (Ctrl+C в терминале)
# Запустить снова
streamlit run app.py
```

### Публичный доступ
Streamlit может создать публичный URL для удаленного доступа:
```bash
streamlit run app.py --server.enableCORS false
```
При первом запуске Streamlit предложит создать бесплатный аккаунт для публичного URL.
