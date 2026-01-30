import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

# --- 1. НАСТРОЙКА ПУТЕЙ ---
# Создаем папку models в текущей директории, если её нет
MODELS_DIR = 'models'
TOKENIZERS_DIR = 'tokenizers'
DATASET_DIR = 'dataset'

#os.makedirs(MODELS_DIR, exist_ok=True)

# Путь к датасету (теперь ищем его в корне или там, где ты его положил)
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
    sentences = df['text'].astype(str).tolist()
    labels = df['complexity'].values
    print(f"✅ Данные загружены! Записей: {len(df)}")
except Exception as e:
    print(f"❌ Ошибка чтения: {e}")
    exit()

# --- 3. ПОДГОТОВКА ТЕКСТА ---
MAX_WORDS = 5000
MAX_LEN = 20

tokenizer = Tokenizer(num_words=MAX_WORDS, lower=True)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
padded_data = pad_sequences(sequences, maxlen=MAX_LEN)

# --- 4. МОДЕЛЬ ---
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(MAX_LEN,)),
    tf.keras.layers.Embedding(MAX_WORDS, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# --- 5. ОБУЧЕНИЕ ---
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

print("\n🚀 Обучение начато...")
model.fit(
    padded_data,
    labels,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# --- 6. СОХРАНЕНИЕ В .KERAS ---
# Все файлы сохраняем в папку models
model_path = os.path.join(MODELS_DIR, 'complexity_model.keras')
tokenizer_path = os.path.join(TOKENIZERS_DIR, 'tokenizer.pickle')

model.save(model_path)  # Сохраняем в новом формате

with open(tokenizer_path, 'wb') as f:
    pickle.dump(tokenizer, f)

print(f"\n✨ Готово! Модель и токенизатор сохранены в папку '{MODELS_DIR}'")