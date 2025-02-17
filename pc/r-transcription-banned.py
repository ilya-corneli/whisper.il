import yt_dlp
import os
import whisper
import logging
import torch
import re
import subprocess
import argparse
import time  # Для повторных попыток и задержек

logging.basicConfig(level=logging.INFO)

def sanitize_filename(filename):
    filename = filename.lower().replace(" ", "-")
    filename = re.sub(r'[^a-zа-я0-9-]', '', filename)
    filename = filename.strip('-')
    return filename

def check_ffmpeg():
    """Проверяет, установлен ли ffmpeg и доступен ли в PATH."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False

def download_video(video_url: str, output_path: str, format_selector='hls-0', progress=False, retries=3, retry_delay=5):
    """Загружает видео с заданным URL с повторными попытками."""
    sanitized_title = sanitize_filename(yt_dlp.YoutubeDL().extract_info(video_url, download=False).get('title', 'UnknownTitle'))
    video_file_downloaded = os.path.join(output_path, f"{sanitized_title}.mp4")

    ydl_opts = {
        'format': format_selector,
        'outtmpl': video_file_downloaded,
        'keepvideo': True,
        'nopart': True,
        'noprogress': not progress, # Прогресс бар включается/выключается опцией
        'rm-cache-dir': True,
    }

    for attempt in range(retries + 1):
        try:
            logging.info(f"Начинаем загрузку видео (попытка {attempt + 1}/{retries + 1}): {video_url} в {video_file_downloaded}")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            return video_file_downloaded  # Успешная загрузка
        except yt_dlp.utils.DownloadError as e:
            if attempt < retries:
                logging.warning(f"Ошибка загрузки (попытка {attempt + 1}/{retries + 1}): {e}. Повторная попытка через {retry_delay} секунд...")
                time.sleep(retry_delay)
                retry_delay *= 2 # Экспоненциальная задержка
            else:
                logging.error(f"Не удалось загрузить видео после {retries + 1} попыток: {e}")
                return None # Не удалось загрузить
        except Exception as e: # Ловим другие непредвиденные ошибки загрузки
            logging.error(f"Непредвиденная ошибка при загрузке видео: {e}")
            return None

def extract_audio_ffmpeg(video_file: str, audio_file_sanitized: str):
    """Извлекает аудио из видеофайла с помощью ffmpeg."""
    logging.info(f"Извлекаем аудио из видео {video_file} в {audio_file_sanitized}")
    try:
        result = subprocess.run([
            "ffmpeg",
            "-i", video_file,
            "-vn",
            "-acodec", "copy",
            audio_file_sanitized
        ], check=True, capture_output=True)
        logging.info(f"Аудио успешно извлечено и сохранено в {audio_file_sanitized}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Ошибка ffmpeg при извлечении аудио (код возврата: {e.returncode}): {e.stderr.decode()}")
        return False
    except FileNotFoundError:
        logging.error("Ошибка: ffmpeg не найден. Убедитесь, что ffmpeg установлен и добавлен в PATH.")
        return False
    except Exception as e:
        logging.error(f"Непредвиденная ошибка при извлечении аудио ffmpeg: {e}")
        return False
    return True


def download_and_transcribe(video_url: str, output_path: str, whisper_model_name: str, progress=False):
    """Скачивает видео, извлекает аудио и транскрибирует."""

    info_dict = yt_dlp.YoutubeDL().extract_info(video_url, download=False)
    video_title = info_dict.get('title', 'UnknownTitle')
    sanitized_title = sanitize_filename(video_title)
    audio_file_sanitized = os.path.join(output_path, f"{sanitized_title}.m4a")
    text_path = os.path.join(output_path, f"{sanitized_title}.txt")


    if os.path.exists(text_path):
        logging.info(f"Транскрипция для {video_url} уже существует. Пропуск.")
        return

    video_file_downloaded = download_video(video_url, output_path, progress=progress)
    if not video_file_downloaded:
        return # Если загрузка не удалась, выходим

    if not extract_audio_ffmpeg(video_file_downloaded, audio_file_sanitized):
        return # Если извлечение аудио не удалось, выходим

    # Удаляем временный видеофайл после извлечения аудио
    if os.path.exists(video_file_downloaded):
        os.remove(video_file_downloaded)
        logging.info(f"Временный видеофайл {video_file_downloaded} удален.")

    transcribe_audio(audio_file_sanitized, output_path, whisper_model_name)


def transcribe_audio(file_path: str, output_folder: str, whisper_model_name: str):
    """Транскрибирует аудиофайл с использованием Whisper."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(whisper_model_name, device=device)

    text_filename = os.path.splitext(os.path.basename(file_path))[0] + ".txt"
    text_path = os.path.join(output_folder, text_filename) # <-- Исправлено здесь!

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
        logging.error(f"Ошибка при транскрипции Whisper файла {file_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Скачать аудио с URL и транскрибировать его с помощью Whisper.")
    group = parser.add_mutually_exclusive_group()  # Группа для URL или файла, теперь необязательная
    group.add_argument("-u", "--urls-flag", nargs='+', help="URL(ы) видео (через пробел), флаг для URL")  # Переименовали флаг, чтобы не конфликтовало с позиционным аргументом
    group.add_argument("-f", "--url-file", type=str, help="Путь к файлу со списком URL (каждый URL на новой строке)")

    parser.add_argument("urls", nargs='*', help="URL(ы) видео (через пробел), позиционный аргумент")  # Позиционный аргумент, nargs='*' чтобы был необязательным в плане наличия в командной строке
    parser.add_argument("-o", "--output-path", default="C:\\files\\MEGA\\Transfer\\audio", help="Путь для сохранения аудио и текстовых файлов")
    parser.add_argument("-m", "--model", default="turbo", choices=["turbo", "tiny", "base", "small", "medium", "large"], help="Модель Whisper для использования")
    parser.add_argument("-p", "--progress", action="store_true", help="Показывать прогресс загрузки")  # Флаг для прогресса

    args = parser.parse_args()

    if not check_ffmpeg():
        logging.error("ffmpeg не установлен или не найден в PATH. Пожалуйста, установите ffmpeg для извлечения аудио.")
        exit(1)

    video_urls = []
    if args.urls_flag:  # Проверяем флаг URLs
        video_urls = args.urls_flag
    elif args.url_file:
        try:
            with open(args.url_file, 'r', encoding='utf-8') as f:
                video_urls = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            logging.error(f"Файл с URL не найден: {args.url_file}")
            exit(1)
    elif args.urls:  # Если позиционные URL-ы указаны
        video_urls = args.urls
    else:  # Если ничего не указано в командной строке, запрашиваем ввод
        video_urls_input = input("Введите URL видео (через пробел, если несколько): ")
        video_urls = video_urls_input.split() # Разделяем введенную строку на список URL по пробелам
        if not video_urls: # Если пользователь ничего не ввел в input()
            print("URL-ы не введены.") # Сообщение, если input пустой
            parser.print_help() # Выводим справку argparse
            exit(1) # Можно убрать exit(1) если хотите просто завершить скрипт без ошибки, но лучше оставить для индикации ошибки


    output_path = args.output_path
    whisper_model_name = args.model
    show_progress = args.progress

    for video_url in video_urls:
        logging.info(f"Обработка URL: {video_url}")
        download_and_transcribe(video_url, output_path, whisper_model_name, progress=show_progress)
        logging.info(f"Обработка URL {video_url} завершена.\n")

    logging.info("Обработка всех URL завершена.")