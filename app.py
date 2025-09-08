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

# Set up Streamlit page config
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
    else:
        st.error(f"❌ CSS file not found: {file_name}")

# Load styling
load_css("style.css")

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
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY, cache_discovery=False)
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
    duration_iso = item["contentDetails"]["duration"]
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
        # Log error silently instead of showing warning in UI
        print(f"[DEBUG] Transcript not available: {e}")
        return None

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
    """Transcribe audio using Whisper (OpenAI API)."""
    with open(filepath, "rb") as audio_file:
        result = openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    return result.text

def chunk_transcript(transcript, chunk_size=500):
    """Split transcript into smaller chunks with timestamps."""
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
    """Summarize a single transcript chunk using OpenAI GPT model."""
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

    # Try to get transcript (YouTube first, then Whisper fallback)
    transcript = fetch_transcript(video_id)

    if not transcript:
        with st.spinner("🎙️ Transcribing audio with Whisper..."):
         with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            audio_path = download_audio(url, tmp.name)
            full_text = transcribe_audio_whisper(audio_path)
            transcript = [{"start": 0, "text": full_text}]


    # Summarize transcript
    with st.spinner("📝 Processing transcript into chunks and generating summary..."):
        chunks = chunk_transcript(transcript)
        all_summaries = []
        for timestamp, chunk in chunks:
            summary = summarize_chunk(chunk, timestamp)
            all_summaries.append(summary)

    final_summary = "\n\n".join(all_summaries)

    # ---------------------------
    # TWO-COLUMN LAYOUT
    # ---------------------------
    col1, col2 = st.columns([1, 1])

    # LEFT: Video metadata + transcript
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if metadata:
            st.subheader("🎬 Video Info")
            st.write(f"**Title:** {metadata['title']}")
            st.write(f"**Channel:** {metadata['channel']}")
            st.write(f"**Length:** {metadata['duration']}")
        else:
            st.warning("⚠️ Could not fetch video metadata.")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        if transcript:
            st.subheader("📜 Full Transcript")
            with st.expander("Click to view transcript", expanded=False):
                if isinstance(transcript, list) and "text" in transcript[0]:
                    transcript_text = " ".join([entry["text"] for entry in transcript])
                else:
                    transcript_text = transcript[0]["text"] if isinstance(transcript, list) else str(transcript)
                st.text_area("Transcript", transcript_text, height=400)
        st.markdown('</div>', unsafe_allow_html=True)

    # RIGHT: Summary + export options
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📝 Summary")
        st.text_area("Summary Output", final_summary, height=400)

        pdf_file = export_pdf(final_summary, metadata["title"] if metadata else "summary")
        with open(pdf_file, "rb") as f:
            st.download_button("📥 Download PDF", f, file_name="summary.pdf")

        st.code(final_summary)
        st.markdown('</div>', unsafe_allow_html=True)