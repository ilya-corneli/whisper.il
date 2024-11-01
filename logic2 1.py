import os
import whisper

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

def sync_directories(src_dir, dest_dir):
    # Создание аналогичной структуры папок в папке назначения
    for root, dirs, _ in os.walk(src_dir):
        for dir in dirs:
            src_path = os.path.join(root, dir)
            relative_path = os.path.relpath(src_path, src_dir)
            dest_path = os.path.join(dest_dir, relative_path)
            os.makedirs(dest_path, exist_ok=True)

def transcribe_audio_files(input_folder, output_base_folder):
    sync_directories(input_folder, output_base_folder)

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(('.wav', '.m4a', '.mp3', '.ogg')):
                file_path = os.path.join(root, file)

                relative_path = os.path.relpath(root, input_folder)
                output_folder = os.path.join(output_base_folder, relative_path)
                os.makedirs(output_folder, exist_ok=True)  # Создание папки, если её нет

                text_filename = os.path.splitext(file)[0] + ".txt"
                text_path = os.path.join(output_folder, text_filename)

                if os.path.exists(text_path):
                    print(f"Транскрипция для {file} уже существует. Пропуск.")
                    continue

                try:
                    result = model.transcribe(file_path)
                    with open(text_path, "w", encoding="utf-8") as f:
                        f.write(result["text"])
                    print(f"Транскрибирован {file} и сохранён в {text_path}")
                except Exception as e:
                    print(f"Ошибка при обработке файла {file}: {e}")

if __name__ == "__main__":
    base_input_folder = "G:\\Mediafiles\\Audio\\Audiobooks"
    output_base_folder = "G:\\Mediafiles\\Bigdata\\Транскрипции"

    specific_folder = input("Введите название конкретной папки внутри Audiobooks: ")
    input_folder = os.path.join(base_input_folder, specific_folder)

    transcribe_audio_files(input_folder, output_base_folder)
