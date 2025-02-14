import os
import torch
import whisper
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm



logging.basicConfig(level=logging.INFO)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Используется устройство: {device}")

model = whisper.load_model("turbo", device=device)

def transcribe_file(args):
    file_path, text_path = args
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
    files_to_process = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(('.wav', '.m4a', '.mp3', '.ogg')):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, base_input_folder)
                output_folder = os.path.join(output_base_folder, relative_path)
                os.makedirs(output_folder, exist_ok=True)
                text_filename = os.path.splitext(file)[0] + ".txt"
                text_path = os.path.join(output_folder, text_filename)
                files_to_process.append((file_path, text_path))
    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(transcribe_file, files_to_process), total=len(files_to_process), desc="Транскрипция аудиофайлов"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Транскрипция аудиофайлов с помощью Whisper")
    parser.add_argument("input_folder", help="Путь к папке с аудиофайлами")
    parser.add_argument("output_folder", help="Путь к папке для сохранения транскрипций")
    args = parser.parse_args()

    input_folder = args.input_folder
    output_base_folder = args.output_folder
    base_input_folder = os.path.dirname(input_folder)

    transcribe_audio_files(input_folder, output_base_folder, base_input_folder)
