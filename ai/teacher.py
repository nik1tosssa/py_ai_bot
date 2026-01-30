import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

# --- 1. НАСТРОЙКА ПУТЕЙ ---
# Все модели и токенизаторы сохраняем в 'models', как ты просил
MODELS_DIR = 'models'
TOKENIZER_DIR = 'tokenizers'
DATASET_DIR = 'dataset'
DATASET_PATH = os.path.join(DATASET_DIR, 'dataset.csv')

# Константы для нейросети
MAX_WORDS = 10000
MAX_LEN = 30


def load_data(path):
    """Загрузка данных с защитой от ошибок типа (Dtype error)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Ошибка: Файл {path} не найден.")

    # Пытаемся прочитать с учетом твоей кодировки
    try:
        df = pd.read_csv(path, sep=';', encoding='utf-16')
    except Exception:
        df = pd.read_csv(path, sep=';', encoding='utf-8')

    # Очистка имен колонок
    df.columns = df.columns.str.strip()

    # ИСПРАВЛЕНИЕ ОШИБКИ DTYPE:
    # Превращаем колонку сложности в числа. Если там текст — станет NaN
    df['complexity'] = pd.to_numeric(df['complexity'], errors='coerce')

    # Удаляем пустые строки или строки, где сложность не определилась
    initial_count = len(df)
    df = df.dropna(subset=['complexity', 'text'])

    # Убеждаемся, что текст — это действительно строки
    df['text'] = df['text'].astype(str)

    if len(df) < initial_count:
        print(f"⚠️ Удалено {initial_count - len(df)} некорректных строк (мусор/ошибки форматирования).")

    # Перемешиваем (важно, так как у тебя мусор в начале файла!)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    sentences = df['text'].tolist()
    # Принудительно в float32 для TensorFlow
    labels = df['complexity'].astype('float32').values

    print(f"✅ Данные загружены! Чистых строк для обучения: {len(df)}")
    return sentences, labels


def create_model():
    """Создание архитектуры нейросети (LSTM)."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(MAX_LEN,)),
        tf.keras.layers.Embedding(MAX_WORDS, 128),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.4),  # Защита от переобучения
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)  # Выход — одно число (сложность)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def main():
    # Создаем папку для моделей, если её нет
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Загрузка
    try:
        sentences, labels = load_data(DATASET_PATH)
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        return

    # 2. Подготовка текста (Токенизация)
    tokenizer = Tokenizer(num_words=MAX_WORDS, lower=True)
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)
    padded_data = pad_sequences(sequences, maxlen=MAX_LEN)

    # 3. Обучение
    model = create_model()

    # Остановка, если модель перестала учиться (обычно на 20-35 эпохе для 25к строк)
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    print(f"\n🚀 Обучение начато...")
    model.fit(
        padded_data,
        labels,
        epochs=100,
        batch_size=64,
        validation_split=0.15,
        callbacks=[early_stop],
        verbose=1
    )

    # 4. Сохранение результатов в папку models
    model_path = os.path.join(MODELS_DIR, 'complexity_model.keras')
    tokenizer_path = os.path.join(TOKENIZER_DIR, 'tokenizer.pickle')

    model.save(model_path)
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)

    print(f"\n✨ Обучение завершено успешно!")
    print(f"📦 Модель сохранена: {model_path}")
    print(f"📦 Токенизатор сохранен: {tokenizer_path}")


if __name__ == "__main__":
    main()