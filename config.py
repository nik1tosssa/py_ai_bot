import os
from dotenv import load_dotenv

load_dotenv()

# Выносим настройки в отдельный объект или переменные
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

if not BOT_TOKEN:
    exit("Ошибка: TELEGRAM_BOT_TOKEN не установлен!")