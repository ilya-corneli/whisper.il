import random
from typing import List, Tuple

class NameGenerator:
    def __init__(self, first_names_file: str, last_names_file: str):
        """
Инициализация генератора имен
:param first_names_file: путь к файлу с именами
:param last_names_file: путь к файлу с фамилиями
        """
        self.first_names = self._load_file(first_names_file)
        self.last_names = self._load_file(last_names_file)

    @staticmethod
    def _load_file(file_path: str) -> List[str]:
        """Загрузка имен/фамилий из файла"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"Файл {file_path} не найден. Использую тестовые данные.")
            return ["Тест"]

    def generate_name_variants(self, count: int = 1) -> List[Tuple[str, ...]]:
        """
Генерация вариантов имен в разных форматах
:param count: количество имен для генерации
:return: список кортежей с разными форматами имен
        """
        result = []
        for _ in range(count):
            first_name = random.choice(self.first_names)
            last_name = random.choice(self.last_names)

            # Генерируем все форматы
            variants = (
                f"{first_name} {last_name}",  # Имя Фамилия
                f"{first_name.lower()} {last_name.lower()}",  # имя фамилия
                f"{first_name}-{last_name}",  # Имя-Фамилия
                f"{first_name.lower()}-{last_name.lower()}",  # имя-фамилия
                f"{first_name}_{last_name}",  # Имя_Фамилия
                f"{first_name.lower()}_{last_name.lower()}"  # имя_фамилия
            )
            result.append(variants)
        return result

def save_names_to_file(names: List[str], filename: str):
    """
Сохранение списка имен в файл
:param names: список имен
:param filename: имя файла
    """
    with open(filename, 'w', encoding='utf-8') as f:
        for name in names:
            f.write(name + '\n')

def main():
    # Пример использования
    generator = NameGenerator('first_names.txt', 'last_names.txt')

    # Генерируем 5 вариантов имен
    generated_names = generator.generate_name_variants(5)

    # Выводим результаты
    for i, variants in enumerate(generated_names, 1):
        print(f"\nНабор {i}:")
        for variant in variants:
            print(variant)

if __name__ == "__main__":
    main()