from aiogram import Router, html
from aiogram.types import Message
from aiogram.filters import CommandStart

# ПРАВИЛЬНЫЙ ИМПОРТ: из папки.файла импортируем класс
from ai.predictor import XPAnalyst

# Создаем объект анализатора один раз при запуске
analyst = XPAnalyst()

router = Router()


@router.message(CommandStart())
async def command_start_handler(message: Message):
    await message.answer(f"Привет, {html.bold(message.from_user.full_name)}! Я готов оценивать твои действия.")


@router.message()
async def echo_handler(message: Message):
    if not message.text:
        return

    try:
        # УБИРАЕМ await, так как analyze — обычная функция
        result = analyst.analyze(message.text)

        if result:
            response = (
                f"📈 {html.bold('Анализ действия:')}\n"
                f"📝 {result['text']}\n"
                f"───────────────────\n"
                f"Сложность: {result['complexity']}/10\n"
                f"Статус: {result['social']}/5 {result['status']}\n"
                f"💰 {html.bold('Опыт:')} +{result['xp']} XP"
            )
            await message.answer(response)

    except Exception as e:
        await message.answer(f"Ошибка при анализе: {e}")