import os
import whisper
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import argparse
import time
from tqdm import tqdm

# Настройка аргументов командной строки
parser = argparse.ArgumentParser(description='Транскрибирование аудиофайлов с помощью Whisper')
parser.add_argument('input_folder', nargs='?', default=None, help='Путь к папке с аудиофайлами')
parser.add_argument('output_folder', nargs='?', default=None, help='Путь к папке для сохранения транскрипций')
args = parser.parse_args()

# Инициализация модели Whisper
try:
    import torch
    print(f"PyTorch версия: {torch.__version__}")
    print(f"CUDA доступна: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        device = "cuda"
        print(f"Количество устройств CUDA: {torch.cuda.device_count()}")
        print(f"Текущее устройство CUDA: {torch.cuda.current_device()}")
        print(f"Название устройства CUDA: {torch.cuda.get_device_name(0)}")
        print("Используется CUDA")
    else:
        device = "cpu"
        print("CUDA недоступна, используется CPU. Процесс может быть медленным.")
except ImportError:
    device = "cpu"
    print("PyTorch не установлен, используется CPU. Процесс может быть медленным. Рекомендуется установить PyTorch с поддержкой CUDA.")

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

def load_model():
    try:
        model = whisper.load_model("turbo", device=device)
        logging.info(f"Модель Whisper 'turbo' успешно загружена на устройство {device}")
        return model
    except Exception as e:
        logging.error(f"Ошибка при загрузке модели: {e}")
        try:
            logging.info("Попытка загрузить модель 'medium' вместо 'turbo'...")
            model = whisper.load_model("medium", device=device)
            logging.info(f"Модель Whisper 'medium' успешно загружена на устройство {device}")
            return model
        except Exception as e:
            logging.error(f"Ошибка при загрузке модели 'medium': {e}")
            try:
                logging.info("Попытка загрузить модель 'base'...")
                model = whisper.load_model("base", device=device)
                logging.info(f"Модель Whisper 'base' успешно загружена на устройство {device}")
                return model
            except Exception as e:
                logging.error(f"Не удалось загрузить ни одну модель: {e}")
                exit(1)

def format_time(seconds):
    """Форматирует время в часы:минуты:секунды"""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

def transcribe_file(file_path, output_folder, progress_bar=None):
    text_filename = os.path.splitext(os.path.basename(file_path))[0] + ".txt"
    text_path = os.path.join(output_folder, text_filename)

    if os.path.exists(text_path):
        logging.info(f"Транскрипция для {file_path} уже существует. Пропуск.")
        # Обновляем прогресс-бар, если он предоставлен
        if progress_bar is not None:
            progress_bar.update(1)
        return False

    try:
        # Обновляем описание прогресс-бара с текущим файлом
        if progress_bar is not None:
            progress_bar.set_description(f"Обработка: {os.path.basename(file_path)}")

        # Очистка кэша CUDA перед загрузкой новой аудиозаписи
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        result = model.transcribe(file_path)
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(result["text"])
        logging.info(f"Транскрибирован {file_path} и сохранён в {text_path}")

        # Обновляем прогресс-бар, если он предоставлен
        if progress_bar is not None:
            progress_bar.update(1)

        return True
    except Exception as e:
        logging.error(f"Ошибка при обработке файла {file_path}: {e}")
        # Обновляем прогресс-бар даже при ошибке
        if progress_bar is not None:
            progress_bar.update(1)
        return False

def transcribe_audio_files(input_folder, output_base_folder, base_input_folder, model):
    files_to_process = []

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(('.wav', '.m4a', '.mp3', '.ogg')):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, base_input_folder)
                output_folder = os.path.join(output_base_folder, relative_path)
                os.makedirs(output_folder, exist_ok=True)
                files_to_process.append((file_path, output_folder))

    if not files_to_process:
        logging.warning(f"Не найдено аудиофайлов для обработки в {input_folder}")
        return

    total_files = len(files_to_process)
    processed_files = 0
    skipped_files = 0
    logging.info(f"Найдено {total_files} файлов для обработки")

    start_time = time.time()

    # Используем tqdm для отображения прогресса и устанавливаем начальное описание
    with tqdm(total=total_files, desc="Проверка файлов") as progress_bar:
        for file_path, output_folder in files_to_process:
            result = transcribe_file(file_path, output_folder, progress_bar)
            if result:
                processed_files += 1
            else:
                skipped_files += 1

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Выводим итоговую статистику
    logging.info(f"Обработка завершена.")
    logging.info(f"Всего файлов: {total_files}")
    logging.info(f"Обработано новых файлов: {processed_files}")
    logging.info(f"Пропущено существующих файлов: {skipped_files}")
    logging.info(f"Затраченное время: {format_time(elapsed_time)}")

if __name__ == "__main__":
    # Если аргументы не указаны через командную строку, запросить их интерактивно
    if args.input_folder is None or args.output_folder is None:
        base_input_folder = "G:\\mediafiles\\audio\\audiobooks"
        output_base_folder = "G:\\mediafiles\\bigdata\\транскрипции"
        specific_folder = input("Введите название конкретной папки внутри Audiobooks: ")
        input_folder = os.path.join(base_input_folder, specific_folder)
    else:
        input_folder = args.input_folder
        output_base_folder = args.output_folder
        base_input_folder = os.path.dirname(input_folder)

    # Загружаем модель
    model = load_model()

    # Синхронизируем структуру папок перед транскрипцией
    sync_directories(base_input_folder, output_base_folder)

    # Засекаем общее время выполнения
    start_time = time.time()

    # Транскрибируем аудиофайлы
    transcribe_audio_files(input_folder, output_base_folder, base_input_folder, model)
    
    # Выводим общее время выполнения всего скрипта
    elapsed_time = time.time() - start_time
    print(f"\nОбщее время выполнения: {format_time(elapsed_time)}")