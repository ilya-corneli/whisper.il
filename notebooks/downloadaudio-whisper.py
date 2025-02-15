import yt_dlp
import os
import whisper
import logging
import torch
import re

logging.basicConfig(level=logging.INFO)

def sanitize_filename(filename):
    filename = filename.lower().replace(" ", "-")
    filename = re.sub(r'[^a-zа-я0-9-]', '', filename)
    filename = filename.strip('-')
    return filename

def download_and_transcribe(video_url: str, output_path: str = "C:\\files\\MEGA\\Transfer\\audio"):

    info_dict = yt_dlp.YoutubeDL().extract_info(video_url, download=False)
    video_title = info_dict.get('title', 'UnknownTitle')
    sanitized_title = sanitize_filename(video_title)
    audio_file_sanitized = os.path.join(output_path, f"{sanitized_title}.m4a")

    ydl_opts = {
        'format': '251',
        'extract_audio': False,
        'outtmpl': audio_file_sanitized,
        'keepvideo': False,
    }

    if os.path.exists(audio_file_sanitized):
        logging.info(f"Файл {audio_file_sanitized} уже существует. Пропуск загрузки.")
    else:
        logging.info(f"Начинаем загрузку с санированным именем: {audio_file_sanitized}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

    transcribe_audio(audio_file_sanitized, output_path)


def transcribe_audio(file_path: str, output_folder: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("base", device=device)

    text_filename = os.path.splitext(os.path.basename(file_path))[0] + ".txt"
    text_path = os.path.join(output_folder, text_filename)

    if os.path.exists(text_path):
        logging.info(f"Транскрипция для {file_path} уже существует. Пропуск.")
        return

    try:
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        if not os.path.exists(file_path):
            logging.error(f"Файл не существует: {file_path}")
            return

        result = model.transcribe(file_path)
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(result["text"])
        logging.info(f"Транскрибирован {file_path} и сохранён в {text_path}")
    except Exception as e:
        logging.error(f"Ошибка при обработке файла {file_path}: {e}")

if __name__ == "__main__":
    video_urls_input = input("Введите URL видео (через пробел, если несколько): ")
    video_urls = video_urls_input.split() # Разделяем введенную строку на список URL по пробелам

    for video_url in video_urls: # Итерируемся по списку URL
        logging.info(f"Обработка URL: {video_url}") # Добавим лог для отслеживания текущего URL
        download_and_transcribe(video_url)
        logging.info(f"Обработка URL {video_url} завершена.\n") # Лог после обработки каждого URL

    logging.info("Обработка всех URL завершена.")