import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
import openai
import tempfile
import os
from fpdf import FPDF
from datetime import timedelta
from dotenv import load_dotenv
from googleapiclient.discovery import build
import isodate

# ---------------------------
# CONFIG
# ---------------------------
load_dotenv()  # load keys from .env

openai.api_key = os.getenv("OPENAI_API_KEY")
YOUTUBE_API_KEY = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="YouTube Summarizer", layout="wide")

# ---------------------------
# HELPERS
# ---------------------------

def get_video_id(url: str) -> str:
    """Extract video ID from YouTube link."""
    if "watch?v=" in url:
        return url.split("watch?v=")[-1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[-1].split("?")[0]
    return url

def get_video_metadata(video_id: str):
    """Fetch video title, channel, and duration using YouTube Data API v3."""
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    request = youtube.videos().list(
        part="snippet,contentDetails",
        id=video_id
    )
    response = request.execute()
    if not response["items"]:
        return None
    
    item = response["items"][0]
    title = item["snippet"]["title"]
    channel = item["snippet"]["channelTitle"]
    duration_iso = item["contentDetails"]["duration"]  # e.g., PT1H3M22S
    duration = isodate.parse_duration(duration_iso).total_seconds()

    return {
        "title": title,
        "channel": channel,
        "duration": str(timedelta(seconds=int(duration)))
    }

def fetch_transcript(video_id: str):
    """Try to fetch transcript from YouTube API."""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        return transcript
    except TranscriptsDisabled:
        return None
    except Exception as e:
        st.warning(f"Transcript not available: {e}")
        return None

def transcribe_audio_whisper(filepath: str):
    """Transcribe audio using Whisper."""
    result = openai.audio.transcriptions.create(
        model="whisper-1",
        file=open(filepath, "rb")
    )
    return result["text"]

def chunk_transcript(transcript, chunk_size=500):
    """Split transcript into smaller chunks with timestamps."""
    chunks, current_chunk, current_time = [], [], None
    for entry in transcript:
        if not current_time:
            current_time = entry['start']
        current_chunk.append(entry['text'])
        if len(" ".join(current_chunk)) > chunk_size:
            chunks.append((current_time, " ".join(current_chunk)))
            current_chunk, current_time = [], None
    if current_chunk:
        chunks.append((current_time, " ".join(current_chunk)))
    return chunks

def summarize_chunk(text, timestamp):
    """Summarize a single transcript chunk."""
    prompt = f"""
    Summarize the following YouTube transcript section. 
    Provide:
    1. Bullet-point key ideas
    2. A short paragraph abstract
    Include the starting timestamp [{str(timedelta(seconds=int(timestamp)))}].

    Transcript:
    {text}
    """

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )

    return response.choices[0].message.content

def export_pdf(summary_text, title="YouTube Summary"):
    """Export summary to a PDF file."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, summary_text)
    temp_path = os.path.join(tempfile.gettempdir(), "summary.pdf")
    pdf.output(temp_path)
    return temp_path

# ---------------------------
# STREAMLIT UI
# ---------------------------

st.title("🎥 YouTube Summarizer")
st.write("Paste a YouTube link to get transcript + AI summary with timestamps.")

url = st.text_input("Enter YouTube URL")
if st.button("Summarize") and url:
    video_id = get_video_id(url)
    metadata = get_video_metadata(video_id)

    if metadata:
        st.subheader(metadata["title"])
        st.write(f"Channel: {metadata['chann]()
