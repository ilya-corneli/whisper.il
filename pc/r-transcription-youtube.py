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

def download_video(video_url: str, output_path: str, format_selector='bestaudio[abr<128]/bestaudio/best', progress=False, retries=3, retry_delay=5):
    """Загружает видео с заданным URL с повторными попытками."""
    sanitized_title = sanitize_filename(yt_dlp.YoutubeDL().extract_info(video_url, download=False).get('title', 'UnknownTitle'))
    audio_file_downloaded = os.path.join(output_path, f"{sanitized_title}.%(ext)s") # Save with original extension first
    audio_file_m4a = os.path.join(output_path, f"{sanitized_title}.m4a") # Final m4a file path


    ydl_opts = {
        'format': format_selector, # Используем динамический селектор форматов
        'outtmpl': audio_file_downloaded, # Save with original extension
        'extract_audio': True,    # Ensure audio extraction
        'audioformat': 'm4a',     # Convert to m4a (if needed, ffmpeg will handle if direct audio is not m4a)
        'keepvideo': False,        # We want only audio, no video
        'nopart': True,
        'noprogress': not progress, # Прогресс бар включается/выключается опцией
        'rm-cache-dir': True,
    }

    for attempt in range(retries + 1):
        try:
            logging.info(f"Начинаем загрузку аудио (попытка {attempt + 1}/{retries + 1}): {video_url} в {audio_file_downloaded}")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])

            # **TROUBLESHOOTING: Check file existence immediately after download**
            expected_downloaded_file = os.path.join(output_path, f"{sanitized_title}.webm") # Assuming webm based on previous log
            if os.path.exists(expected_downloaded_file):
                logging.info(f"TROUBLESHOOTING: Файл существует сразу после загрузки: {expected_downloaded_file}")
            else:
                logging.warning(f"TROUBLESHOOTING: Файл НЕ существует сразу после загрузки: {expected_downloaded_file}")


            # Find the downloaded file (yt-dlp might add format extension) and rename to .m4a
            downloaded_file = None
            for ext in ['m4a', 'webm', 'mp3', 'aac', 'opus']: # Common audio extensions
                temp_file_path = os.path.join(output_path, f"{sanitized_title}.{ext}")
                if os.path.exists(temp_file_path):
                    downloaded_file = temp_file_path
                    break

            # **TROUBLESHOOTING: List all files in output_path if not found by extension**
            if downloaded_file is None:
                all_files_in_output_path = os.listdir(output_path)
                logging.warning(f"TROUBLESHOOTING: Аудиофайл не найден по расширениям. Файлы в output_path: {all_files_in_output_path}")


            if downloaded_file: # Check if a file was actually downloaded
                if downloaded_file != audio_file_m4a: # Rename only if extensions are different
                    os.rename(downloaded_file, audio_file_m4a)
                    logging.info(f"Переименовано в {audio_file_m4a}")
                return audio_file_m4a # Return final .m4a path
            else:
                logging.error(f"Ошибка: аудиофайл не был загружен для URL: {video_url}")
                return None


        except yt_dlp.utils.DownloadError as e:
            if attempt < retries:
                logging.warning(f"Ошибка загрузки (попытка {attempt + 1}/{retries + 1}): {e}. Повторная попытка через {retry_delay} секунд...")
                time.sleep(retry_delay)
                retry_delay *= 2 # Экспоненциальная задержка
            else:
                logging.error(f"Не удалось загрузить аудио после {retries + 1} попыток: {e}")
                return None # Не удалось загрузить
        except Exception as e: # Ловим другие непредвиденные ошибки загрузки
            logging.error(f"Непредвиденная ошибка при загрузке аудио: {e}")
            return None


def extract_audio_ffmpeg(audio_file_input: str, audio_file_sanitized: str):
    """Конвертирует аудиофайл в m4a с помощью ffmpeg."""
    logging.info(f"Конвертируем аудио в m4a с помощью ffmpeg: {audio_file_input} -> {audio_file_sanitized}") # Изменено сообщение
    try:
        result = subprocess.run([
            "ffmpeg",
            "-i", audio_file_input, # Используем входной аудиофайл
            "-acodec", "aac",        # Явно кодируем в AAC (m4a)
            "-ab", "128k",         # Битрейт 128kbps (можно настроить)
            "-vn",                 # Нет видео
            "-y",                   # Перезаписать, если существует
            audio_file_sanitized   # Выходной m4a файл
        ], check=True, capture_output=True)
        logging.info(f"Аудио успешно конвертировано в m4a: {audio_file_sanitized}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Ошибка ffmpeg при конвертации аудио (код возврата: {e.returncode}): {e.stderr.decode()}")
        return False
    except FileNotFoundError:
        logging.error("Ошибка: ffmpeg не найден. Убедитесь, что ffmpeg установлен и добавлен в PATH.")
        return False
    except Exception as e:
        logging.error(f"Непредвиденная ошибка при конвертации аудио ffmpeg: {e}")
        return False
    finally:
        if os.path.exists(audio_file_input): # Check if input file exists before trying to delete
            os.remove(audio_file_input) # Remove original downloaded audio file after conversion
            logging.info(f"Временный аудиофайл {audio_file_input} удален после конвертации.")
    return True


def download_and_transcribe(video_url: str, output_path: str, whisper_model_name: str, progress=False):
    """Скачивает аудио, извлекает аудио и транскрибирует."""

    info_dict = yt_dlp.YoutubeDL().extract_info(video_url, download=False)
    video_title = info_dict.get('title', 'UnknownTitle')
    sanitized_title = sanitize_filename(video_title)
    audio_file_sanitized = os.path.join(output_path, f"{sanitized_title}.m4a")
    text_path = os.path.join(output_path, f"{sanitized_title}.txt")


    if os.path.exists(text_path):
        logging.info(f"Транскрипция для {video_url} уже существует. Пропуск.")
        return

    audio_file_downloaded = download_video(video_url, output_path, progress=progress) # Скачиваем аудио напрямую

    if not audio_file_downloaded:
        return # Если загрузка не удалась, выходим

    # No ffmpeg conversion needed anymore for YouTube, we download m4a directly (or similar)
    # if not extract_audio_ffmpeg(audio_file_downloaded, audio_file_sanitized): # Конвертируем в m4a, если необходимо
    #    return # Если конвертация аудио не удалась, выходим


    transcribe_audio(audio_file_sanitized, output_path, whisper_model_name)


def transcribe_audio(file_path: str, output_folder: str, whisper_model_name: str):
    """Транскрибирует аудиофайл с использованием Whisper."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(whisper_model_name, device=device)

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