import asyncio
import logging
import sys

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

# Импортируем настройки и роутеры
from config import BOT_TOKEN
from handlers import router

async def main():
    # Настраиваем бота
    bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp = Dispatcher()

    # САМОЕ ВАЖНОЕ: подключаем наш роутер к главному диспетчеру
    dp.include_router(router)

    # Запускаем логирование и бота
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())