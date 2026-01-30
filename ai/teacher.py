import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

# --- 1. НАСТРОЙКА ПУТЕЙ ---
MODELS_DIR = 'models'
TOKENIZERS_DIR = 'tokenizers'
DATASET_DIR = 'dataset'

# Создаем папки, если их нет
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TOKENIZERS_DIR, exist_ok=True)

DATASET_PATH = os.path.join(DATASET_DIR, 'dataset.csv')

if not os.path.exists(DATASET_PATH):
    print(f"❌ Ошибка: Файл {DATASET_PATH} не найден.")
    exit()

# --- 2. ЗАГРУЗКА ДАННЫХ ---
try:
    try:
        df = pd.read_csv(DATASET_PATH, sep=';', encoding='utf-16')
    except UnicodeError:
        df = pd.read_csv(DATASET_PATH, sep=';', encoding='utf-8')

    df.columns = df.columns.str.strip()

    # КРИТИЧЕСКИ ВАЖНО: Перемешиваем, так как мусор в начале!
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    sentences = df['text'].astype(str).tolist()
    labels = df['complexity'].values
    print(f"✅ Данные загружены! Всего: {len(df)} строк.")
except Exception as e:
    print(f"❌ Ошибка чтения: {e}")
    exit()

# --- 3. ПОДГОТОВКА ТЕКСТА ---
MAX_WORDS = 10000  # Увеличили для 17к строк
MAX_LEN = 30  # Увеличили длину контекста для сложных описаний

tokenizer = Tokenizer(num_words=MAX_WORDS, lower=True)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
padded_data = pad_sequences(sequences, maxlen=MAX_LEN)

# --- 4. МОДЕЛЬ (Адаптирована под 17к строк) ---
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(MAX_LEN,)),
    tf.keras.layers.Embedding(MAX_WORDS, 128),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    # Помогает выделить самые важные слова в предложении
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# --- 5. ОБУЧЕНИЕ ---
# Для большого датасета ставим patience побольше
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print("\n🚀 Обучение на 17 000 строк начато...")
model.fit(
    padded_data,
    labels,
    epochs=100,
    batch_size=64,  # Увеличили размер батча для скорости
    validation_split=0.15,  # 15% на проверку достаточно для такого объема
    callbacks=[early_stop],
    verbose=1
)

# --- 6. СОХРАНЕНИЕ (РАЗДЕЛЬНОЕ) ---
model_path = os.path.join(MODELS_DIR, 'complexity_model.keras')
tokenizer_path = os.path.join(TOKENIZERS_DIR, 'tokenizer.pickle')

model.save(model_path)
with open(tokenizer_path, 'wb') as f:
    pickle.dump(tokenizer, f)

print(f"\n✨ Готово!")
print(f"📦 Модель: {model_path}")
print(f"📦 Токен: {tokenizer_path}")