import yt_dlp

def download_audio(video_urls: list, output_path: str = "C:\\files\\MEGA\\Transfer\\audio"):
    def format_output(info_dict):
        channel = info_dict.get('uploader', 'UnknownChannel')
        title = info_dict.get('title', 'UnknownTitle')
        return f"{output_path}/{channel} - {title}.%(ext)s"

    options = {
        'format': 'bestaudio/best',
        'extract_audio': True,
        'audio_format': 'mp3',
        'outtmpl': f'{output_path}/%(uploader)s - %(title)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'aac',
            'preferredquality': '192',
        }],
    }

    with yt_dlp.YoutubeDL(options) as ydl:
        ydl.download(video_urls)

if __name__ == "__main__":
    urls = input("Введите URL видео через пробел: ").split()
    download_audio(urls)
    