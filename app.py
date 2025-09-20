# ---------------------------
# IMPORTS
# ---------------------------
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
import yt_dlp
from sqlalchemy import create_engine, Column, Integer, String, Text, TIMESTAMP, func
from sqlalchemy.orm import declarative_base, sessionmaker
from openai import APIConnectionError, APIStatusError
from pydub import AudioSegment
import math
import unicodedata


# ---------------------------
# CONFIG
# ---------------------------
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
YOUTUBE_API_KEY = os.getenv("GOOGLE_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")  # e.g. postgresql+psycopg2://user:pass@localhost/db

# Streamlit setup
st.set_page_config(page_title="YouTube Summarizer", layout="wide")

# ---------------------------
# STYLES
# ---------------------------
def load_css(file_name: str):
    import pathlib
    css_path = pathlib.Path(file_name)
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# ---------------------------
# DATABASE SETUP
# ---------------------------
from sqlalchemy import create_engine, Column, Integer, String, Text, TIMESTAMP, func
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

class Summary(Base):
    __tablename__ = "summaries"

    id = Column(Integer, primary_key=True)
    video_id = Column(String(50), nullable=False)
    title = Column(Text, nullable=False)
    channel = Column(Text)
    duration = Column(Text)
    transcript = Column(Text)
    summary = Column(Text)
    created_at = Column(TIMESTAMP, server_default=func.now())

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://summarizer:secret12345@localhost:5432/youtube_summarizer")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

# ✅ Auto-create tables if they don’t exist
Base.metadata.create_all(engine)

def save_summary(video_id, metadata, transcript, summary):
    session = SessionLocal()
    try:
        existing = session.query(Summary).filter_by(video_id=video_id).first()
        transcript_text = (
            " ".join([t["text"] for t in transcript])
            if isinstance(transcript, list) else str(transcript)
        )
        if existing:
            existing.title = metadata.get("title", "")
            existing.channel = metadata.get("channel", "")
            existing.duration = metadata.get("duration", "")
            existing.transcript = transcript_text
            existing.summary = summary
        else:
            new_entry = Summary(
                video_id=video_id,
                title=metadata.get("title", ""),
                channel=metadata.get("channel", ""),
                duration=metadata.get("duration", ""),
                transcript=transcript_text,
                summary=summary,
            )
            session.add(new_entry)
        session.commit()
    finally:
        session.close()

def get_summary(video_id):
    session = SessionLocal()
    try:
        record = session.query(Summary).filter_by(video_id=video_id).first()
        if record:
            return {
                "title": record.title,
                "channel": record.channel,
                "duration": record.duration,
                "transcript": record.transcript,
                "summary": record.summary,
            }
        return None
    finally:
        session.close()

def list_summaries(limit=20):
    session = SessionLocal()
    try:
        return (
            session.query(Summary)
            .order_by(Summary.created_at.desc())
            .limit(limit)
            .all()
        )
    finally:
        session.close()

# ---------------------------
# HELPERS
# ---------------------------
def get_video_id(url: str) -> str:
    if "watch?v=" in url:
        return url.split("watch?v=")[-1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[-1].split("?")[0]
    return url

def get_video_metadata(video_id: str):
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY, cache_discovery=False)
    request = youtube.videos().list(part="snippet,contentDetails", id=video_id)
    response = request.execute()
    if not response["items"]:
        return None
    item = response["items"][0]
    title = item["snippet"]["title"]
    channel = item["snippet"]["channelTitle"]
    duration_iso = item["contentDetails"]["duration"]
    duration = isodate.parse_duration(duration_iso).total_seconds()
    return {"title": title, "channel": channel, "duration": str(timedelta(seconds=int(duration)))}

def fetch_transcript(video_id: str):
    try:
        return YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
    except TranscriptsDisabled:
        return None
    except Exception as e:
        print(f"[DEBUG] Transcript not available: {e}")
        return None

def download_audio(video_url: str, filename: str):
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": filename.replace(".mp3", ""),
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
    return filename if filename.endswith(".mp3") else filename + ".mp3"

def split_audio(file_path, chunk_length_ms=5*60*1000):  # default 5 minutes
    """Split audio into smaller chunks to stay under Whisper API 25 MB limit."""
    audio = AudioSegment.from_file(file_path, format="mp3")
    chunks = []
    num_chunks = math.ceil(len(audio) / chunk_length_ms)

    for i in range(num_chunks):
        start = i * chunk_length_ms
        end = min((i+1) * chunk_length_ms, len(audio))
        chunk = audio[start:end]
        out_path = f"{file_path}_part{i}.mp3"
        chunk.export(out_path, format="mp3")
        chunks.append(out_path)

    return chunks


def transcribe_audio_whisper_large(filepath: str):
    """Handle large audio files by chunking before Whisper transcription."""
    transcripts = []
    chunks = split_audio(filepath)

    for chunk in chunks:
        with open(chunk, "rb") as audio_file:
            result = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        transcripts.append(result.text)

    return " ".join(transcripts)


def transcribe_audio_whisper(filepath: str):
    with open(filepath, "rb") as audio_file:
        result = openai.audio.transcriptions.create(model="whisper-1", file=audio_file)
    return result.text

def chunk_transcript(transcript, chunk_size=500):
    chunks, current_chunk, current_time = [], [], None
    for entry in transcript:
        if not current_time:
            current_time = entry.get("start", 0)
        current_chunk.append(entry["text"])
        if len(" ".join(current_chunk)) > chunk_size:
            chunks.append((current_time, " ".join(current_chunk)))
            current_chunk, current_time = [], None
    if current_chunk:
        chunks.append((current_time, " ".join(current_chunk)))
    return chunks

def summarize_chunk(text, timestamp):
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

def clean_text(text: str) -> str:
    """
    Normalize text so FPDF doesn't crash on Unicode characters.
    Converts fancy quotes, dashes, bullets, etc. into plain ASCII.
    """
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")

def export_pdf(summary_text, title="YouTube Summary"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # ✅ Clean the text before writing
    pdf.multi_cell(0, 10, clean_text(summary_text))

    temp_path = os.path.join(tempfile.gettempdir(), "summary.pdf")
    pdf.output(temp_path)
    return temp_path

# ---------------------------
# PROCESS VIDEO (PIPELINE)
# ---------------------------
@st.cache_data(show_spinner=False)
def process_video(url: str):
    """Full pipeline: fetch metadata, transcript, and generate summary with error handling."""
    try:
        video_id = get_video_id(url)
        metadata = get_video_metadata(video_id)

        # Try YouTube transcript
        transcript = fetch_transcript(video_id)

        # Fall back to Whisper if transcript unavailable
        if not transcript:
            with st.spinner("🎙️ Transcribing audio with Whisper..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                    audio_path = download_audio(url, tmp.name)
                    full_text = transcribe_audio_whisper_large(audio_path)
                    if not full_text:
                        st.error("❌ Could not transcribe the audio. Please try again.")
                        return None, None, None
                    transcript = [{"start": 0, "text": full_text}]

        # Summarization
        with st.spinner("📝 Processing transcript into chunks and generating summary..."):
            chunks = chunk_transcript(transcript)
            all_summaries = []
            for timestamp, chunk in chunks:
                summary = summarize_chunk(chunk, timestamp)
                all_summaries.append(summary)

            final_summary = "\n\n".join(all_summaries)

        # Save to Postgres if available
        if metadata and transcript and final_summary:
            save_summary(video_id, metadata, transcript, final_summary)

        return metadata, transcript, final_summary

    except APIConnectionError:
        st.error("⚠️ Connection error: Could not reach OpenAI servers. Disable your VPN or check internet connection.")
        return None, None, None
    except APIStatusError as e:
        st.error(f"⚠️ API error: {e}")
        return None, None, None
    except Exception as e:
        st.error(f"⚠️ Unexpected error: {e}")
        return None, None, None

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.markdown("""
<div style="background:#FF0000; padding:1rem; text-align:center; color:white; border-radius:8px;">
    <h1 style="margin:0;">🎥 YouTube Summarizer</h1>
    <p style="margin:0; font-size:1.1rem;">Paste a YouTube link to get transcript + AI summary with timestamps.</p>
</div>
""", unsafe_allow_html=True)


tab1, tab2 = st.tabs(["▶️ Summarize", "📚 History"])

# --- Tab 1: Summarize ---
with tab1:
    url = st.text_input("Enter YouTube URL")
    if st.button("Summarize") and url:
        with st.spinner("⚡ Processing video..."):
            metadata, transcript, final_summary = process_video(url)

        col1, col2 = st.columns([1, 1])
        with col1:
            with st.container():
                st.markdown("""
                <div style="background:#fff; border:1px solid #ddd; border-radius:10px; padding:1rem; margin-bottom:1rem;">
                    <h3>🎬 Video Info</h3>
                </div>
                """, unsafe_allow_html=True)
            st.write(f"**Title:** {metadata['title']}")
            st.write(f"**Channel:** {metadata['channel']}")
            st.write(f"**Length:** {metadata['duration']}")
            st.subheader("📜 Full Transcript")
            with st.expander("Click to view transcript", expanded=False):
                transcript_text = " ".join([t["text"] for t in transcript])
                st.markdown(f"""
                <div style="max-height:400px; overflow-y:auto; background:#f5f5f5; padding:1rem; border-radius:8px; border:1px solid #ddd;">
                    {transcript_text}
                </div>
                """, unsafe_allow_html=True)


        with col2:
            st.markdown("""
            <div style="background:#fff; border:1px solid #ddd; border-radius:10px; padding:1rem; margin-bottom:1rem;">
                <h3>📝 Summary</h3>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(final_summary)

            pdf_file = export_pdf(final_summary, metadata["title"] if metadata else "summary")
            with open(pdf_file, "rb") as f:
                st.download_button("📥 Download PDF", f, file_name="summary.pdf")

# --- Tab 2: History ---
with tab2:
    st.subheader("📚 Saved Summaries")
    summaries = list_summaries(limit=20)
    if summaries:
        for record in summaries:
            with st.expander(f"{record.title} ({record.channel}) — {record.duration}"):
                st.markdown(record.summary)
                st.caption(f"Saved on {record.created_at}")
    else:
        st.info("No summaries saved yet.")
