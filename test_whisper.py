import os
import tempfile
from dotenv import load_dotenv
import yt_dlp
import openai

# Load API key from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def download_audio(video_url: str, filename: str):
    """Download audio from YouTube as MP3 using yt-dlp + ffmpeg."""
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": filename.replace(".mp3", ""),  # yt-dlp adds .mp3
        "quiet": True,
        "noplaylist": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

    # Ensure .mp3 extension
    if not filename.endswith(".mp3"):
        filename = filename + ".mp3"
    return filename


def transcribe_audio_whisper(filepath: str):
    """Transcribe audio using Whisper (OpenAI API)."""
    with open(filepath, "rb") as audio_file:
        result = openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    return result.text


if __name__ == "__main__":
    url = input("🔗 Enter a YouTube URL: ")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        print("⬇️ Downloading audio...")
        audio_path = download_audio(url, tmp.name)

        print("🎙️ Transcribing with Whisper...")
        transcript = transcribe_audio_whisper(audio_path)

        print("\n✅ TRANSCRIPT RESULT:\n")
        print(transcript[:1000])  # print first 1000 chars
