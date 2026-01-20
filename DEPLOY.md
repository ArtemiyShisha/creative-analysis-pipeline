# Деплой на Streamlit Cloud

## Пошаговая инструкция

### 1. Подготовка репозитория (если еще не сделано)

```bash
# Перейдите в папку проекта
cd /Users/artemshishkin/personal-ai-workspace/projects/saliency-test

# Инициализируйте Git (если еще не сделали)
git init

# Настройте Git (если первый раз)
git config user.name "Ваше Имя"
git config user.email "your.email@example.com"

# Добавьте все файлы
git add .

# Создайте первый коммит
git commit -m "Initial commit: Creative Analysis Pipeline with Streamlit UI"
```

### 2. Создайте GitHub репозиторий

1. Зайдите на [github.com](https://github.com)
2. Нажмите "New repository"
3. Название: `creative-analysis-pipeline` (или любое другое)
4. Выберите **Public** (для бесплатного Streamlit Cloud)
5. НЕ добавляйте README, .gitignore (они уже есть)
6. Нажмите "Create repository"

### 3. Отправьте код на GitHub

```bash
# Добавьте remote (замените YOUR_USERNAME на ваш GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/creative-analysis-pipeline.git

# Отправьте код
git branch -M main
git push -u origin main
```

### 4. Деплой на Streamlit Cloud

1. Зайдите на [share.streamlit.io](https://share.streamlit.io)
2. Войдите через GitHub
3. Нажмите "New app"
4. Выберите:
   - **Repository:** `YOUR_USERNAME/creative-analysis-pipeline`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Нажмите "Advanced settings"
6. В разделе **Secrets** добавьте:
   ```toml
   OPENAI_API_KEY = "sk-proj-ваш-ключ-тут"
   ```
7. Нажмите "Deploy!"

### 5. Ожидание деплоя

- Первый деплой займет 5-10 минут (установка зависимостей и загрузка ML моделей)
- Streamlit Cloud выделит вам URL вида: `https://your-app-name.streamlit.app`
- После деплоя приложение будет доступно по этому URL

### 6. Обновление кода

После изменений в коде:

```bash
git add .
git commit -m "Описание изменений"
git push
```

Streamlit Cloud автоматически подхватит изменения и передеплоит приложение.

## Важные моменты

### API ключ в Secrets
- **НЕ коммитьте** `config.py` с реальным API ключом
- Всегда используйте Secrets в Streamlit Cloud
- Приложение будет читать `OPENAI_API_KEY` из environment variables

### Обновление config.py для production

Streamlit Cloud передает secrets как environment variables. Обновите `config.py`:

```python
import os

# Try environment variable first (Streamlit Cloud), then fallback to local
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# For local development, uncomment:
# OPENAI_API_KEY = "sk-proj-your-local-key"
```

### Лимиты Streamlit Cloud (бесплатный tier)

- **Память:** 1GB RAM (может быть недостаточно для больших моделей)
- **CPU:** Shared resources
- **Cold start:** Приложение "засыпает" после 7 дней неактивности
- **Private apps:** До 1 приватного app (нужен GitHub private repo)

### Если возникли проблемы

1. **"Out of memory"**
   - DeepGaze + EasyOCR могут превысить 1GB лимит
   - Решение: используйте платный tier Streamlit Cloud или VPS

2. **"Module not found"**
   - Проверьте `requirements.txt` - все ли зависимости указаны
   - Проверьте `packages.txt` - системные пакеты (libgl1 для OpenCV)

3. **Медленная загрузка**
   - Первая загрузка медленная (модели ~500MB)
   - После первого запуска модели кэшируются

4. **API key не работает**
   - Проверьте Secrets в настройках приложения
   - Формат должен быть: `OPENAI_API_KEY = "sk-proj-..."`

## Альтернативы (если Streamlit Cloud не подходит)

### Render (если нужно больше памяти)
- Free tier: 512MB (недостаточно)
- Starter tier: $7/month, 1GB+ RAM
- [render.com](https://render.com)

### Railway (гибкий вариант)
- $5 бесплатных кредитов каждый месяц
- Pay-as-you-go после
- [railway.app](https://railway.app)

### VPS (полный контроль)
- DigitalOcean: $6-12/month
- Hetzner: €4-8/month
- Требует настройки nginx + SSL

## Мониторинг

После деплоя:
- Проверьте логи в Streamlit Cloud dashboard
- Тестируйте с тестовыми изображениями (`data/yandex_pay.png`)
- Убедитесь, что API calls работают (проверьте OpenAI dashboard)

## Безопасность

- ✅ `config.py` в `.gitignore`
- ✅ API ключ в Secrets (не в коде)
- ✅ HTTPS из коробки (Streamlit Cloud)
- ✅ Публичный репозиторий безопасен (код без секретов)

---

**Готово!** Теперь ваше приложение доступно всем по публичному URL 🎉
