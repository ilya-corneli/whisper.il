import os
import whisper
from whisper.utils import get_writer

# Запрос пути к папке у пользователя
input_folder = input("Введите путь к папке с аудио файлами: ")

# Проверка, существует ли указанная папка
if not os.path.isdir(input_folder):
    print(f"Папка {input_folder} не существует.")
    exit(1)

# Проверка доступности CUDA
try:
    import torch
    if torch.cuda.is_available():
        device = "cuda"
        print("Используется CUDA")
    else:
        device = "cpu"
        print("CUDA недоступна, используется CPU. Процесс может быть медленным.")
except ImportError:
    device = "cpu"
    print("PyTorch не установлен, используется CPU. Процесс может быть медленным. Рекомендуется установить PyTorch с поддержкой CUDA.")

# Инициализация модели Whisper
model = whisper.load_model("turbo", device=device)

def transcribe_folder(input_folder, model):
    """Транскрибирует все аудиофайлы в папке и ее подпапках."""
    for filename in os.listdir(input_folder):
        filepath = os.path.join(input_folder, filename)
        if os.path.isdir(filepath):
            # Рекурсивный вызов для подпапки
            transcribe_folder(filepath, model)
        elif filename.endswith((".wav", ".m4a", ".mp3")):
            try:
                result = model.transcribe(filepath)

                # Сохранение транскрипции в текстовый файл
                text_filename = os.path.splitext(filename)[0] + ".txt"
                text_path = os.path.join(input_folder, text_filename)
                with open(text_path, "w", encoding="utf-8") as f:
                    f.write(result["text"])

                # Сохранение транскрипции в VTT файл
                vtt_filename = os.path.splitext(filename)[0] + ".vtt"
                vtt_path = os.path.join(input_folder, vtt_filename)
                writer = get_writer("vtt", input_folder)  # Изменено на 'vtt'
                writer(result, filepath)

                print(f"Транскрибирован {filename} и сохранен в {text_filename} и {vtt_filename}")

            except Exception as e:
                print(f"Ошибка при обработке файла {filename}: {e}")

transcribe_folder(input_folder, model)