import os
import whisper
from whisper.utils import get_writer

# Запрос пути к папке у пользователя
root_folder = input("Введите путь к каталогу папок с аудиофайлами: ")

# Проверка, существует ли указанный каталог
if not os.path.isdir(root_folder):
    print(f"Каталог {root_folder} не существует.")
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
    """Транскрибирует все аудиофайлы в папке и её подпапках."""
    for root, _, files in os.walk(input_folder):
        for filename in files:
            if filename.endswith((".wav", ".m4a", ".mp3", ".ogg")):
                filepath = os.path.join(root, filename)
                # Создание папок для различных форматов
                text_folder = os.path.join(root, "Транскрипции_txt")
                vtt_folder = os.path.join(root, "Транскрипции_vtt")

                if not os.path.exists(text_folder):
                    os.makedirs(text_folder)
                if not os.path.exists(vtt_folder):
                    os.makedirs(vtt_folder)

                text_filename = os.path.splitext(filename)[0] + ".txt"
                text_path = os.path.join(text_folder, text_filename)
                vtt_filename = os.path.splitext(filename)[0] + ".vtt"
                vtt_path = os.path.join(vtt_folder, vtt_filename)

                # Проверка наличия уже созданных файлов
                if os.path.exists(text_path) and os.path.exists(vtt_path):
                    print(f"Файлы {text_filename} и {vtt_filename} уже существуют. Пропуск файла {filename}.")
                    continue

                try:
                    result = model.transcribe(filepath)
                    # Сохранение транскрипции в текстовый файл
                    with open(text_path, "w", encoding="utf-8") as f:
                        f.write(result["text"])
                    # Сохранение транскрипции в VTT файл
                    writer = get_writer("vtt", vtt_folder)
                    writer(result, filepath)
                    print(f"Транскрибирован {filename} и сохранён в {text_filename} и {vtt_filename}")
                except Exception as e:
                    print(f"Ошибка при обработке файла {filename}: {e}")

transcribe_folder(root_folder, model)
