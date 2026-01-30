import os
import csv
import string
from aiogram import Router, html, F, types
from aiogram.types import Message, CallbackQuery
from aiogram.filters import CommandStart
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.fsm.state import StatesGroup, State
from aiogram.fsm.context import FSMContext

# Импорт твоего анализатора
from ai.predictor import XPAnalyst

analyst = XPAnalyst()
router = Router()


# --- 1. СОСТОЯНИЯ (FSM) ---
class FeedbackStates(StatesGroup):
    waiting_for_complexity = State()


# --- 2. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ОЧИСТКИ ---

def clean_text(text: str) -> str:
    """Приводит к нижнему регистру и удаляет все знаки препинания"""
    if not text:
        return ""
    # Переводим в нижний регистр
    text = text.lower()
    # Удаляем пунктуацию через таблицу подстановки
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Убираем лишние пробелы по краям и внутри
    return " ".join(text.split())


def extract_action_text(message_text):
    """Ищет строку 'Действие: ' и возвращает её очищенной"""
    for line in message_text.split('\n'):
        if line.startswith("Действие: "):
            raw_action = line.replace("Действие: ", "").strip()
            return clean_text(raw_action)
    return "неизвестное действие"


# --- 3. ЛОГИРОВАНИЕ ---
def log_user_feedback(text, complexity, status):
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
    print(f"📊 Чистая запись ({status}): {text};{complexity}")


# --- 4. КЛАВИАТУРА ---
def get_confirm_keyboard(complexity):
    builder = InlineKeyboardBuilder()
    builder.row(
        types.InlineKeyboardButton(text="✅ Согласен", callback_data=f"confirm_ok:{complexity}"),
        types.InlineKeyboardButton(text="❌ Нет", callback_data="confirm_no")
    )
    return builder.as_markup()


# --- 5. ХЕНДЛЕРЫ ---

@router.message(CommandStart())
async def command_start_handler(message: Message):
    await message.answer(f"Привет, {html.bold(message.from_user.full_name)}! Я готов анализировать твои действия.")


@router.callback_query(F.data.startswith("confirm_ok:"))
async def process_ok_rating(callback_query: CallbackQuery):
    current_complexity = callback_query.data.split(":")[1]
    original_action = extract_action_text(callback_query.message.text)
    log_user_feedback(original_action, current_complexity, "good")
    await callback_query.answer("Записано! ✅")
    await callback_query.message.edit_text(f"{callback_query.message.text}\n\n🤖 Спасибо! Оценка подтверждена.")


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


@router.message(FeedbackStates.waiting_for_complexity)
async def manual_complexity_input(message: Message, state: FSMContext):
    if message.text == "/cancel":
        await state.clear()
        await message.answer("Ввод отменен.")
        return
    user_input = message.text.replace(',', '.')
    try:
        new_val = float(user_input)
        if 0 <= new_val <= 10:
            data = await state.get_data()
            original_text = data.get("wrong_text")
            log_user_feedback(original_text, new_val, "bad")
            await message.answer(f"✅ Записал: '{original_text}' как {new_val} XP.")
            await state.clear()
        else:
            await message.answer("⚠ Введи число от 0 до 10.")
    except ValueError:
        await message.answer("⚠ Введи число или /cancel.")


@router.message()
async def echo_handler(message: Message):
    if not message.text or message.text.startswith('/'):
        return

    # Очищаем текст перед анализом
    user_action = clean_text(message.text)

    try:
        result = analyst.analyze(user_action)
        if result:
            comp = result['complexity']
            await message.answer(
                f"📊 Сложность действия: **{comp}**\n\n"
                f"Действие: {user_action}\n"
                f"💰 Опыт: +{result['xp']} XP\n\n"
                f"Вы согласны с оценкой?",
                reply_markup=get_confirm_keyboard(comp),
                parse_mode="Markdown"
            )
    except Exception as e:
        await message.answer(f"Ошибка при анализе: {e}")