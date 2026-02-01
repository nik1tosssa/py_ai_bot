import os
import csv
import string
from datetime import datetime
from dotenv import load_dotenv

from aiogram import Router, html, F, types
from aiogram.types import Message, CallbackQuery
from aiogram.filters import CommandStart, StateFilter
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.fsm.state import StatesGroup, State
from aiogram.fsm.context import FSMContext

from logger import Logger
from ai.predictor import XPAnalyst

# --- ИНИЦИАЛИЗАЦИЯ ---
load_dotenv()
analyst = XPAnalyst()
router = Router()
logger = Logger()

# Получаем ID админа из .env (обязательно числом)
try:
    ADMIN_ID = int(os.getenv("ADMINS_TELEGRAM_ID"))
except (TypeError, ValueError):
    ADMIN_ID = None
    print("Ошибка: ADMINS_TELEGRAM_ID не найден в .env или имеет неверный формат")


# --- 1. СОСТОЯНИЯ (FSM) ---
class FeedbackStates(StatesGroup):
    waiting_for_complexity = State()


# --- 2. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---

def clean_text(text: str) -> str:
    """Очистка для нейросети (нижний регистр, без знаков препинания)."""
    if not text:
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return " ".join(text.split())


def extract_action_text(message_text: str):
    """Извлекает оригинальный текст действия из сообщения бота."""
    for line in message_text.split('\n'):
        if line.startswith("Действие: "):
            return line.replace("Действие: ", "").strip()
    return "неизвестное действие"


def log_user_feedback(text, complexity, status):
    """Запись в датасеты ai/dataset/."""
    text = clean_text(text)
    filename = "good_user_dataset.csv" if status == "good" else "bad_user_dataset.csv"
    folder_path = os.path.join("ai", "dataset")
    os.makedirs(folder_path, exist_ok=True)
    filepath = os.path.join(folder_path, filename)

    file_exists = os.path.isfile(filepath)
    with open(filepath, mode='a', encoding='utf-16', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        if not file_exists or os.path.getsize(filepath) == 0:
            writer.writerow(['text', 'complexity'])
        writer.writerow([text, complexity])


def get_confirm_keyboard(complexity):
    """Клавиатура с кнопками."""
    builder = InlineKeyboardBuilder()
    builder.row(
        types.InlineKeyboardButton(text="✅ Согласен", callback_data=f"confirm_ok:{complexity}"),
        types.InlineKeyboardButton(text="❌ Нет", callback_data="confirm_no")
    )
    return builder.as_markup()


# --- 3. ХЕНДЛЕРЫ ---

@router.message(CommandStart())
async def command_start_handler(message: Message):
    await message.answer(f"Привет, {html.bold(message.from_user.full_name)}! Отправь мне описание действия.")


# А) Хэндлер для кнопки "Согласен"
@router.callback_query(F.data.startswith("confirm_ok:"))
async def process_ok_rating(callback_query: CallbackQuery, state: FSMContext):
    current_complexity = callback_query.data.split(":")[1]
    original_action = extract_action_text(callback_query.message.text)
    user_tag = f"@{callback_query.from_user.username}" if callback_query.from_user.username else "User"

    # Редактируем отчет у админа
    data = await state.get_data()
    report_id = data.get("admin_report_id")
    if report_id and ADMIN_ID:
        try:
            await callback_query.bot.edit_message_text(
                chat_id=ADMIN_ID,
                message_id=report_id,
                text=f"✅ {user_tag} подтвердил оценку.\nТекст: \"{original_action}\"\nСложность: {current_complexity}"
            )
        except Exception:
            pass

    log_user_feedback(original_action, current_complexity, "good")
    await state.clear()
    await callback_query.answer("Записано! ✅")
    await callback_query.message.edit_text(f"{callback_query.message.text}\n\n🤖 Спасибо! Оценка подтверждена.")


# Б) Хэндлер для кнопки "Нет"
@router.callback_query(F.data == "confirm_no")
async def process_bad_rating(callback_query: CallbackQuery, state: FSMContext):
    original_action = extract_action_text(callback_query.message.text)
    await state.update_data(wrong_text=original_action)
    await state.set_state(FeedbackStates.waiting_for_complexity)

    await callback_query.message.answer(
        f"🤖 Ошибка в оценке: «{original_action}»\n"
        "Введи свою сложность (0-10) или /cancel"
    )
    await callback_query.answer()


# В) Хэндлер ПРИЕМА РУЧНОГО ЧИСЛА (Работает только когда стейт активен)
@router.message(FeedbackStates.waiting_for_complexity)
async def manual_complexity_input(message: Message, state: FSMContext):
    if message.text == "/cancel":
        await state.clear()
        await message.answer("❌ Ввод отменен.")
        return

    user_input = message.text.replace(',', '.')
    user_tag = f"@{message.from_user.username}" if message.from_user.username else "User"

    try:
        new_val = float(user_input)
        if 0 <= new_val <= 10:
            data = await state.get_data()
            original_text = data.get("wrong_text")
            report_id = data.get("admin_report_id")

            # Обновляем все логи
            log_user_feedback(original_text, new_val, "bad")
            logger.update_complexity(message.from_user.id, new_val)

            # Редактируем отчет админа
            if report_id and ADMIN_ID:
                try:
                    await message.bot.edit_message_text(
                        chat_id=ADMIN_ID,
                        message_id=report_id,
                        text=f"⚠️ {user_tag} ИСПРАВИЛ оценку\nТекст: \"{original_text}\"\nНовая сложность: {new_val}"
                    )
                except Exception:
                    pass

            await message.answer(f"✅ Готово! Оценка {new_val} сохранена.\n Давай оценим новую активность!")
            await state.clear()
        else:
            await message.answer("⚠ Число должно быть от 0 до 10.")
    except ValueError:
        await message.answer("⚠ Введи число (например 5.5) или /cancel.")


# Г) ОСНОВНОЙ ХЭНДЛЕР (Работает только если стейт пустой)
@router.message(StateFilter(None))
async def send_answer(message: Message, state: FSMContext):
    if not message.text or message.text.startswith('/'):
        return

    user_action = clean_text(message.text)
    user_tag = f"@{message.from_user.username}" if message.from_user.username else message.from_user.full_name

    try:
        result = analyst.analyze(user_action)
        if result:
            comp = result['complexity']

            # Пишем в общий лог
            logger.log(
                message.from_user.id,
                user_tag,
                message.text,
                comp,
                datetime.now().strftime("%d.%m.%Y %H:%M:%S")
            )

            # Отправляем отчет админу
            if ADMIN_ID:
                try:
                    admin_msg = await message.bot.send_message(
                        chat_id=ADMIN_ID,
                        text=f"🔔 {user_tag} оценивает:\n\"{message.text}\"\nОценка: {comp}"
                    )
                    await state.update_data(admin_report_id=admin_msg.message_id)
                except Exception as e:
                    print(f"Ошибка отправки админу: {e}")

            await message.answer(
                f"📊 Сложность действия: **{comp}**\n\n"
                f"Действие: {message.text}\n"
                f"💰 Опыт: +{result.get('xp', int(comp * 100))} XP\n\n"
                f"Вы согласны с оценкой?",
                reply_markup=get_confirm_keyboard(comp),
                parse_mode="Markdown"
            )
        else:
            await message.answer("Не удалось оценить действие.")
    except Exception as e:
        await message.answer(f"Ошибка: {e}")