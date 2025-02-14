import os
import whisper
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Инициализация модели Whisper
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

model = whisper.load_model("turbo", device=device)

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def sync_directories(src_dir, dest_dir):
    # Создание аналогичной структуры папок в папке назначения
    for root, dirs, _ in os.walk(src_dir):
        for dir in dirs:
            src_path = os.path.join(root, dir)
            relative_path = os.path.relpath(src_path, src_dir)
            dest_path = os.path.join(dest_dir, relative_path)
            os.makedirs(dest_path, exist_ok=True)

def transcribe_file(file_path, output_folder):
    text_filename = os.path.splitext(os.path.basename(file_path))[0] + ".txt"
    text_path = os.path.join(output_folder, text_filename)

    if os.path.exists(text_path):
        logging.info(f"Транскрипция для {file_path} уже существует. Пропуск.")
        return

    try:
        result = model.transcribe(file_path)
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(result["text"])
        logging.info(f"Транскрибирован {file_path} и сохранён в {text_path}")
    except Exception as e:
        logging.error(f"Ошибка при обработке файла {file_path}: {e}")

def transcribe_audio_files(input_folder, output_base_folder, base_input_folder):
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for root, _, files in os.walk(input_folder):
            for file in files:
                if file.endswith(('.wav', '.m4a', '.mp3', '.ogg')):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(root, base_input_folder)
                    output_folder = os.path.join(output_base_folder, relative_path)
                    os.makedirs(output_folder, exist_ok=True)
                    futures.append(executor.submit(transcribe_file, file_path, output_folder))

        for future in as_completed(futures):
            future.result()  # проверяем завершение задачи и обрабатываем исключения

if __name__ == "__main__":
    base_input_folder = "G:\\mediafiles\\audio\\audiobooks"
    output_base_folder = "G:\\mediafiles\\bigdata\\транскрипции"

    specific_folder = input("Введите название конкретной папки внутри Audiobooks: ")
    input_folder = os.path.join(base_input_folder, specific_folder)

    # Синхронизируем структуру папок перед транскрипцией
    sync_directories(base_input_folder, output_base_folder)

    # Транскрибируем аудиофайлы
    transcribe_audio_files(input_folder, output_base_folder, base_input_folder)
    