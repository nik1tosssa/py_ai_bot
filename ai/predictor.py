import os
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- НАСТРОЙКИ ПУТЕЙ ---
# Указываем путь к папке 'models' в директории текущего файла
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
TOKENIZERS_DIR = os.path.join(BASE_DIR, 'tokenizers')

MODEL_PATH = os.path.join(MODELS_DIR, 'complexity_model.keras')
TOKENIZER_PATH = os.path.join(TOKENIZERS_DIR, 'tokenizer.pickle')


class XPAnalyst:
    def __init__(self, model_path=MODEL_PATH, tokenizer_path=TOKENIZER_PATH):
        """Загрузка модели и токенизатора из папки models"""
        try:
            # Загружаем модель сложности
            self.model = tf.keras.models.load_model(model_path, compile=False)
            # Загружаем токенизатор
            with open(tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
            self.is_ready = True
            print("✅ Нейросеть анализа сложности готова!")
        except Exception as e:
            print(f"❌ Ошибка загрузки активов: {e}")
            self.is_ready = False

    def analyze(self, text: str):
        """Возвращает только сложность действия и рассчитанный XP"""
        if not self.is_ready:
            return None

        MAX_LEN = 20  # Должно совпадать с параметром при обучении

        # 1. Предобработка текста
        sequence = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=MAX_LEN)

        # 2. Предсказание (теперь только один выход — сложность)
        prediction = self.model.predict(padded, verbose=0)

        # Извлекаем значение сложности
        # Если модель выдает одно значение, берем первый элемент
        comp = float(prediction[0][0])

        # 3. Расчет XP на основе сложности
        # Например: сложность (1-10) * базовую ставку 100
        total_xp = int(max(0, comp * 100))

        return {
            "text": text,
            "complexity": round(comp, 2),
            "xp": total_xp,
            "status": self._get_simple_status(comp)
        }

    def _get_simple_status(self, comp):
        """Статус на основе уровня сложности"""
        if comp > 7: return "🏆 Эпично"
        if comp > 4: return "⚡️ Непросто"
        return "🌱 Легко"


# --- ТЕСТ ---
if __name__ == "__main__":
    analyst = XPAnalyst()
    res = analyst.analyze("Спроектировал спорткар")
    if res:
        print(f"Текст: {res['text']} | Сложность: {res['complexity']} | XP: {res['xp']}")