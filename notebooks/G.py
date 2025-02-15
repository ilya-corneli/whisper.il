
import yt_dlp
import os
import whisper
import logging
import torch
import re
import google.generativeai as genai

logging.basicConfig(level=logging.INFO)

# ---  Настройка Gemini ---
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    logging.error("Ошибка: Не установлена переменная окружения GEMINI_API_KEY.")
    exit(1)  # Выход из программы, если ключ не найден

generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 2048,
}

model = genai.GenerativeModel(model_name="gemini-pro", generation_config=generation_config)

# ---  Функции для скачивания и транскрипции (из вашего кода) ---
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

    transcribe_audio(audio_file_sanitized, output_path)  # Транскрибируем
    return os.path.join(output_path, os.path.splitext(os.path.basename(audio_file_sanitized))[0] + ".txt") # Возвращаем путь к файлу с транскрипцией

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

# ---  Функция для взаимодействия с Gemini ---
def process_with_gemini(text_file_path: str, prompt_template: str):
    try:
        with open(text_file_path, "r", encoding="utf-8") as f:
            transcribed_text = f.read()

        prompt = prompt_template.format(transcribed_text=transcribed_text) # Вставляем транскрипцию в шаблон промпта

        response = model.generate_content(prompt, stream=True) # Используем generate_content, а не start_chat

        for chunk in response:
            try:
                print(chunk.text, end="", flush=True)
            except Exception as e:
                print(f"\nОшибка при обработке потока: {e}")
        print()

    except FileNotFoundError:
        logging.error(f"Файл транскрипции не найден: {text_file_path}")
    except Exception as e:
        logging.error(f"Ошибка при взаимодействии с Gemini: {e}")

# ---  Основная часть программы ---
if __name__ == "__main__":
    video_urls_input = input("Введите URL видео (через пробел, если несколько): ")
    video_urls = video_urls_input.split()

    #  Шаблон промпта (ВАЖНО: настройте его под ваши задачи)
    prompt_template = """
Вот транскрипция видео:

{transcribed_text}

Пожалуйста, сделайте следующее:
1. Кратко изложите основную тему видео.
2. Выделите 3-5 ключевых моментов.
3. Определите, есть ли в видео какие-либо призывы к действию.
"""

    for video_url in video_urls:
        logging.info(f"Обработка URL: {video_url}")
        text_file_path = download_and_transcribe(video_url) # получаем путь к транскрипции

        if text_file_path: # проверяем что путь получен
            process_with_gemini(text_file_path, prompt_template)
        else:
            logging.error("Не удалось получить путь к файлу транскрипции")

        logging.info(f"Обработка URL {video_url} завершена.\n")

    logging.info("Обработка всех URL завершена.")