import random
import csv
import os
import re
import time
import pynvml
from datetime import datetime
from typing import Optional, Set
from openai import OpenAI

# --- КОНФИГУРАЦИЯ ---
BASE_URL="http://10.14.0.2:1234/v1"
AI_API_KEY="lm-studio"
client = OpenAI(base_url=BASE_URL, api_key=AI_API_KEY)

# Темы для обеспечения разнообразия датасета
CATEGORIES = [
    "домашняя инженерия и электроника", "агротехника и садоводство",
    "кулинария и нутрициология", "финансы и планирование",
    "образование и сложные навыки", "ремонт и реставрация",
    "информационные технологии и софт", "здоровье и биохакинг",
    "химия и физика в быту", "организация пространства и логистика"
]


def final_clean(text: str) -> str:
    """Глубокая очистка текста от артефактов модели."""
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'(сложность|параметры|значение|баллы|одобряемость)\s*\d*', '', text, flags=re.IGNORECASE)
    # Удаляем всё кроме букв и пробелов
    text = "".join(c for c in text if c.isalpha() or c.isspace())
    return " ".join(text.split()).lower().strip()


def get_action(complexity: int) -> Optional[str]:
    """Шаг 1: Генератор действия с принудительной сменой темы."""
    category = random.choice(CATEGORIES)
    prompt = f"""
    Придумай одно уникальное действие в сфере быта или обучения.
    КАТЕГОРИЯ: {category}
    ЦЕЛЕВАЯ СЛОЖНОСТЬ: {complexity} из 10.

    ПРАВИЛА:
    1. Формат: 5-10 слов, прошедшее время, нижний регистр, БЕЗ МЕСТОИМЕНИЙ (мой, свой).
    2. Если сложность > 5, придумай серьезную интеллектуальную или физическую задачу.
    3. Ориентируйся на эти оценки
        0 - секундное действие не требующее каких-то усилий
        1 - рутинное действие, о котором даже не задумываемся
        2 - ежедневное базовое бытовое действие, не более пары минут для реализации
        3 - ежедневное базовое бытовое действие, не более получаса для реализации
        4 - средней сложности действие, от получаса до двух часов непрерывной работы
        5 - средней сложности действие, от двух до четырех часов непрерывной работы
        6 - средней сложности действие, требующее физических усилий, более двух часов работы
        7 - средней сложности действие, требующее физических и интеллектуальных усилий, более двух часов работы
        8 - действие повышенной сложности, завязанное на комбинации интеллектуального и физического труда
        9 - действие повышенной сложности, завязанное на комбинации интеллектуального и физического труда, создание или реставрация чего-то уникального
        10 - многонедельный интеллектуально и физически сложный процесс 
    4. ЗАПРЕЩЕНО: посуда, стирка, фотографии, котята, часы (уже много в базе).
    5. Будь конкретным в терминах. Только текст действия!
    """
    try:
        res = client.chat.completions.create(
            model="local-model",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9
        )
        return final_clean(res.choices[0].message.content)
    except:
        return None


def evaluate_complexity(action: str) -> float:
    """Шаг 2: ИИ-Судья оценивает реальную сложность фразы."""
    prompt = f"""
    Оцени сложность действия по шкале 0-10 (физический и умственный труд).
    Действие: "{action}"

    Критерии:
        0 - секундное действие не требующее каких-то усилий
        1 - рутинное действие, о котором даже не задумываемся
        2 - ежедневное базовое бытовое действие, не более пары минут для реализации
        3 - ежедневное базовое бытовое действие, не более получаса для реализации
        4 - средней сложности действие, от получаса до двух часов непрерывной работы
        5 - средней сложности действие, от двух до четырех часов непрерывной работы
        6 - средней сложности действие, требующее физических усилий, более двух часов работы
        7 - средней сложности действие, требующее физических и интеллектуальных усилий, более двух часов работы
        8 - действие повышенной сложности, завязанное на комбинации интеллектуального и физического труда
        9 - действие повышенной сложности, завязанное на комбинации интеллектуального и физического труда, создание или реставрация чего-то уникального
        10 - многонедельный интеллектуально и физически сложный процесс 

    Верни ТОЛЬКО число (например 7.5).
    """
    try:
        res = client.chat.completions.create(
            model="local-model",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        nums = re.findall(r'\d+\.\d+|\d+', res.choices[0].message.content)
        return float(nums[0]) if nums else 5.0
    except:
        return 5.0


def load_existing_actions(filename: str) -> Set[str]:
    """Загрузка существующих фраз для предотвращения дублей."""
    actions = set()
    if os.path.exists(filename):
        try:
            with open(filename, mode='r', encoding='utf-16') as f:
                reader = csv.reader(f, delimiter=';')
                next(reader, None)  # Пропуск заголовка
                for row in reader:
                    if row: actions.add(row[0].strip().lower())
        except:
            print("Предупреждение: Не удалось прочитать старый файл.")
    return actions


def write_to_csv(action: str, c: float, filename: str = 'dataset/new_dataset.csv'):
    """Запись в CSV в папку 'dataset'."""
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', encoding='utf-16', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        if not file_exists:
            writer.writerow(['text', 'complexity'])
        writer.writerow([action, round(c, 2)])

def get_gpu_temp() -> int:
    """Получает текущую температуру первой видеокарты в системе."""
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0) # 0 — это первая видеокарта
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        pynvml.nvmlShutdown()
        return temp
    except Exception as e:
        print(f"Ошибка чтения температуры: {e}")
        return 0

if __name__ == "__main__":
    # Создаем папку если её нет
    if not os.path.exists('dataset'):
        os.makedirs('dataset')

    DATASET_PATH = os.path.join('dataset', input("Введите название файла датасета:"))
    seen_actions = load_existing_actions(DATASET_PATH)

    # --- НАСТРОЙКИ ЗАПУСКА ---
    MIN_RANDOM = int(input("Минимальный порог шкалы:"))
    MAX_RANDOM = int(input("Максимальный порог шкалы:"))
    TOTAL_RECORDS = int(input("Количество генерируемых записей:"))  # Сколько НОВЫХ уникальных строк нужно добавить
    RANDOM_COMPLEXITY = int(input(
        f"Случайная сложность \n"
        f"1 - случайное действие в диапозоне {MIN_RANDOM} - {MAX_RANDOM}, \n"
        f"0 - сложность действий будет возрастать на 1\n"
        f"Ваш ответ:"
    ))
    PAUSE_INTERVAL = 0  # Через сколько строк делать паузу
    PAUSE_TIME = 0  # Длительность паузы в секундах


    new_count = 0
    target_c = 0
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Скрипт запущен.")
    print(f"Уже известно фраз: {len(seen_actions)}")

    while new_count < TOTAL_RECORDS:
        if RANDOM_COMPLEXITY == 1: target_c =  random.randint(MIN_RANDOM, MAX_RANDOM)
        elif target_c == 10:
            target_c = 0
        else:
            target_c += 1
        action = get_action(target_c)

        current_temp = get_gpu_temp()
        if current_temp > 78:  # Если карта горячее 78 градусов
            print(f"--- GPU ПЕРЕГРЕТА ({current_temp}°C). Ждем охлаждения... ---")
            while get_gpu_temp() > 60:  # Ждем, пока остынет до 60
                time.sleep(5)
            print("--- Температура в норме, продолжаем работу. ---")

        # Проверка на длину и уникальность
        if action and len(action.split()) >= 3 and action not in seen_actions:
            real_c = evaluate_complexity(action)

            # Среднее арифметическое для сглаживания данных
            final_score = (real_c + target_c) / 2

            write_to_csv(action, final_score, DATASET_PATH)
            seen_actions.add(action)
            new_count += 1

            print(
                f"[{new_count}/{TOTAL_RECORDS}] | {get_gpu_temp()}°C | {datetime.now().strftime('%H:%M:%S')} | {action} | C:{final_score:.1f}")
        else:
            continue

    print(f"--- ГЕНЕРАЦИЯ ЗАВЕРШЕНА ---")
    print(f"Итого в базе {len(seen_actions)} уникальных записей.")