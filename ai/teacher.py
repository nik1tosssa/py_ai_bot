import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- 1. НАСТРОЙКА ПУТЕЙ И ПАПОК ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data_set')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
TOKENIZER_DIR = os.path.join(BASE_DIR, 'tokenizers')
DATASET_PATH = os.path.join(DATA_DIR, 'data_set.csv')

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

if not os.path.exists(DATASET_PATH):
    print(f"❌ Файл не найден: {DATASET_PATH}")
    exit()

# --- 2. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ ---
# Используем sep=None для автоопределения (; или ,) и cp1251 для кириллицы
df = pd.read_csv(DATASET_PATH, sep=None, engine='python', encoding='utf-16')

df['text'] = df['text'].fillna(' ')
sentences = df['text'].astype(str).tolist()
complexity_labels = df['complexity'].fillna(0).values
social_labels = df['social'].fillna(0).values

# ПАРАМЕТРЫ (Важно!)
MAX_WORDS = 5000  # Сколько уникальных слов помнит нейросеть
MAX_LEN = 20     # Длина фразы (в словах)

# ТОКЕНИЗАЦИЯ (Превращаем слова в числа)
tokenizer = Tokenizer(num_words=MAX_WORDS, lower=True)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
padded_data = pad_sequences(sequences, maxlen=MAX_LEN)

# --- 3. АРХИТЕКТУРА НЕЙРОСЕТИ ---
input_layer = tf.keras.layers.Input(shape=(MAX_LEN,), name='input_layer')
x = tf.keras.layers.Embedding(MAX_WORDS, 64)(input_layer)
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)

out_comp = tf.keras.layers.Dense(1, name='complexity_head')(x)
out_soc = tf.keras.layers.Dense(1, name='social_head')(x)

model = tf.keras.Model(inputs=input_layer, outputs=[out_comp, out_soc])

# --- 4. КОМПИЛЯЦИЯ И ОБУЧЕНИЕ ---
model.compile(
    optimizer='adam',
    loss='mse',
    metrics={'complexity_head': 'mae', 'social_head': 'mae'}
)

print("\n🚀 Начинаю обучение...")
model.fit(
    padded_data,
    {'complexity_head': complexity_labels, 'social_head': social_labels},
    epochs=100,
    batch_size=32,
    verbose=1
)

# --- 5. ФУНКЦИЯ ПРЕДСКАЗАНИЯ ---
def predict_action(text):
    # Превращаем текст в такую же последовательность цифр, как при обучении
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=MAX_LEN)
    comp, soc = model.predict(pad, verbose=0)

    c = comp[0][0]
    s = soc[0][0]
    # Рассчитываем XP: сложность * социальный вес * 100
    total_xp = int(max(0, c * s * 100))

    print(f"\n--- Анализ ---")
    print(f"Текст: {text} | Сложность: {c:.2f} | Соц. вес: {s:.2f} | XP: {total_xp}")

# Проверка
predict_action("Катнул катку в дотку")
predict_action("Прочитал сложную статью по нейросетям")

# --- 6. СОХРАНЕНИЕ ---
model_path = os.path.join(MODELS_DIR, 'xp_model.keras')
tokenizer_path = os.path.join(TOKENIZER_DIR, 'tokenizer.pickle')

model.save(model_path)

with open(tokenizer_path, 'wb') as f:
    pickle.dump(tokenizer, f)

print(f"\n✅ Модель и токенизатор успешно сохранены в папку: {MODELS_DIR}")