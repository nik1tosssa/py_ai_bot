import os
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- НАСТРОЙКИ ПУТЕЙ ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Используем папку models для обоих файлов, как ты просил
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'xp_model.keras')
TOKENIZER_PATH = os.path.join(BASE_DIR, 'tokenizers', 'tokenizer.pickle')


class XPAnalyst:
    def __init__(self, model_path=MODEL_PATH, tokenizer_path=TOKENIZER_PATH):
        """При создании объекта сразу загружаем активы один раз"""
        try:
            self.model = tf.keras.models.load_model(model_path, compile=False)
            with open(tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
            self.is_ready = True
        except Exception as e:
            print(f"❌ Ошибка загрузки активов нейросети: {e}")
            self.is_ready = False

    def analyze(self, text: str):
        """Основная функция анализа текста"""
        if not self.is_ready:
            return None

        MAX_LEN = 20

        # Предобработка
        sequence = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=MAX_LEN)

        # Предсказание
        predictions = self.model.predict(padded, verbose=0)

        # Извлекаем значения (Keras для Multi-Output возвращает список)
        # predictions[0] - первый выход (complexity), predictions[1] - второй (social)
        comp = float(predictions[0][0][0])
        soc = float(predictions[1][0][0]) - 5

        # Твоя формула расчета XP
        # Ограничиваем сложность и вес, чтобы не уходить в дикие минусы
        total_xp = int(max(0, comp * soc * 100))

        # Возвращаем чистый словарь с данными
        return {
            "text": text,
            "complexity": round(comp, 2),
            "social": round(soc, 2),
            "xp": total_xp,
            "status": self._get_status(soc)
        }

    def _get_status(self, soc_weight):
        """Внутренняя функция для определения текстового статуса"""
        if soc_weight > 1.5: return "🌟 Полезно"
        if soc_weight < -0.5: return "💀 Деструктивно"
        return "😐 Нейтрально"


# --- ПРИМЕР ИСПОЛЬЗОВАНИЯ В ДРУГОМ СКРИПТЕ ---
if __name__ == "__main__":
    analyst = XPAnalyst()

    result = analyst.analyze("Прочитал главу учебника по химии")
    if result:
        print(f"Результат: {result['xp']} XP | Статус: {result['status']}")