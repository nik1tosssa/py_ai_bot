import os
import csv


class Logger:
    # Путь к файлу логов
    LOG_FILE_PATH = os.path.join('logs', 'logs.csv')

    def __init__(self):
        # Создаем папку logs при инициализации, если её еще нет
        os.makedirs(os.path.dirname(self.LOG_FILE_PATH), exist_ok=True)

    def log(self, user_id: int, username: str, message: str, complexity: float, dt_string: str):
        """Записывает данные в CSV файл."""

        # Проверяем, нужно ли писать заголовок (файл не существует или пустой)
        file_not_exist = not os.path.exists(self.LOG_FILE_PATH) or os.path.getsize(self.LOG_FILE_PATH) == 0

        # Используем utf-16, как и в твоих датасетах
        with open(self.LOG_FILE_PATH, mode='a', encoding='utf-16', newline='') as f:
            writer = csv.writer(f, delimiter=';')

            if file_not_exist:
                writer.writerow(['id', 'username', 'message', 'complexity', 'datetime'])

            writer.writerow([user_id, username, message, complexity, dt_string])

    def update_complexity(self, user_id: int, new_complexity: float):
        """Ищет последнюю запись пользователя в логах и меняет сложность."""
        if not os.path.exists(self.LOG_FILE_PATH):
            return

        rows = []
        updated = False

        # Читаем все данные
        with open(self.LOG_FILE_PATH, mode='r', encoding='utf-16') as f:
            reader = list(csv.reader(f, delimiter=';'))
            if not reader: return
            header = reader[0]
            data = reader[1:]

        # Ищем последнюю запись этого пользователя (идем с конца)
        for i in range(len(data) - 1, -1, -1):
            # data[i][0] — это id пользователя в твоем логгере
            if str(data[i][0]) == str(user_id) and not updated:
                data[i][3] = str(new_complexity)  # 3 — индекс колонки complexity
                updated = True

        # Перезаписываем файл
        with open(self.LOG_FILE_PATH, mode='w', encoding='utf-16', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(header)
            writer.writerows(data)