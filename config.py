import os
from dotenv import load_dotenv

load_dotenv()

# Выносим настройки в отдельный объект или переменные
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
AI_API_KEY = os.getenv("AI_API_KEY")
BASE_URL = os.getenv("BASE_URL")

if not BOT_TOKEN:
    exit("Ошибка: TELEGRAM_BOT_TOKEN не установлен!")