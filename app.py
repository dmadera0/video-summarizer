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

# ---------------------------
# CONFIG
# ---------------------------
# Load API keys from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
YOUTUBE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Streamlit page setup
st.set_page_config(page_title="YouTube Summarizer", layout="wide")

# ---------------------------
# STYLES
# ---------------------------
def load_css(file_name: str):
    """Inject custom CSS into the Streamlit app."""
    import pathlib
    css_path = pathlib.Path(file_name)
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load external CSS file
load_css("style.css")

# ---------------------------
# HELPERS (basic utilities)
# ---------------------------
def get_video_id(url: str) -> str:
    """Extract video ID from a YouTube URL."""
    if "watch?v=" in url:
        return url.split("watch?v=")[-1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[-1].split("?")[0]
    return url

def download_audio(video_url: str, filename: str):
    """Download audio from YouTube as MP3 using yt-dlp + ffmpeg."""
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

    if not filename.endswith(".mp3"):
        filename = filename + ".mp3"
    return filename

def transcribe_audio_whisper(filepath: str):
    """Transcribe audio file using OpenAI Whisper."""
    with open(filepath, "rb") as audio_file:
        result = openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    return result.text

def chunk_transcript(transcript, chunk_size=500):
    """Split transcript into chunks with timestamps."""
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
    """Summarize a transcript chunk using GPT."""
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
# CACHED HELPERS (API calls)
# ---------------------------
@st.cache_data(show_spinner=False)
def get_video_metadata(video_id: str):
    """Fetch video metadata (title, channel, duration)."""
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY, cache_discovery=False)
    request = youtube.videos().list(part="snippet,contentDetails", id=video_id)
    response = request.execute()
    if not response["items"]:
        return None
    item = response["items"][0]
    duration_iso = item["contentDetails"]["duration"]
    duration = isodate.parse_duration(duration_iso).total_seconds()
    return {
        "title": item["snippet"]["title"],
        "channel": item["snippet"]["channelTitle"],
        "duration": str(timedelta(seconds=int(duration)))
    }

@st.cache_data(show_spinner=False)
def fetch_transcript(video_id: str):
    """Try to fetch transcript directly from YouTube."""
    try:
        return YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
    except TranscriptsDisabled:
        return None
    except Exception as e:
        print(f"[DEBUG] Transcript not available: {e}")
        return None

# ---------------------------
# FULL PIPELINE (CACHED)
# ---------------------------
@st.cache_data(show_spinner=True)
def process_video(url: str):
    """End-to-end pipeline: metadata → transcript → summary (cached)."""
    video_id = get_video_id(url)
    metadata = get_video_metadata(video_id)

    # Transcript
    transcript = fetch_transcript(video_id)
    if not transcript:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            audio_path = download_audio(url, tmp.name)
            full_text = transcribe_audio_whisper(audio_path)
            transcript = [{"start": 0, "text": full_text}]

    # Summarization
    chunks = chunk_transcript(transcript)
    summaries = [summarize_chunk(chunk, ts) for ts, chunk in chunks]
    final_summary = "\n\n".join(summaries)

    return metadata, transcript, final_summary

# ---------------------------
# STREAMLIT UI
# ---------------------------
# Top bar with centered title
st.markdown("""
    <div class="top-bar">
        <h1>🎥 YouTube Summarizer</h1>
    </div>
""", unsafe_allow_html=True)

# Subtitle under the bar
st.markdown("<p style='text-align: center; font-size: 1.1rem; color: #606060;'>Paste a YouTube link to get transcript + AI summary with timestamps.</p>", unsafe_allow_html=True)



url = st.text_input("Enter YouTube URL")
if st.button("Summarize") and url:
    with st.spinner("⚡ Processing video..."):
        metadata, transcript, final_summary = process_video(url)

    # Two-column layout
    col1, col2 = st.columns([1, 1])

    # LEFT: Video Info + Transcript
    with col1:
        if metadata:
            st.subheader("🎬 Video Info")
            st.write(f"**Title:** {metadata['title']}")
            st.write(f"**Channel:** {metadata['channel']}")
            st.write(f"**Length:** {metadata['duration']}")
        if transcript:
            st.subheader("📜 Full Transcript")
            with st.expander("Click to view transcript", expanded=False):
                transcript_text = " ".join([entry["text"] for entry in transcript])
                st.text_area("Transcript", transcript_text, height=400)

    # RIGHT: Summary + Export
    with col2:
        st.subheader("📝 Summary")
        st.markdown(final_summary)

        pdf_file = export_pdf(final_summary, metadata["title"] if metadata else "summary")
        with open(pdf_file, "rb") as f:
            st.download_button("📥 Download PDF", f, file_name="summary.pdf")
