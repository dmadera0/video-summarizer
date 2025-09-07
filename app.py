import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from pytube import YouTube
import openai
import tempfile
import os
import subprocess
from fpdf import FPDF
from datetime import timedelta

# ---------------------------
# CONFIG
# ---------------------------
openai.api_key = os.getenv("OPENAI_API_KEY")

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

def download_audio(video_url: str, filename: str):
    """Download audio from YouTube video."""
    yt = YouTube(video_url)
    stream = yt.streams.filter(only_audio=True).first()
    stream.download(filename=filename)

def transcribe_audio_whisper(filepath: str):
    """Transcribe audio using Whisper."""
    result = openai.Audio.transcriptions.create(
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

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )

    return response["choices"][0]["message"]["content"]

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
    yt = YouTube(url)
    st.subheader(yt.title)
    st.write(f"Channel: {yt.author}")
    st.write(f"Length: {str(timedelta(seconds=yt.length))}")

    # Get transcript or fallback to Whisper
    transcript = fetch_transcript(video_id)

    if not transcript:
        st.info("Transcript not available. Downloading audio for Whisper transcription...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            download_audio(url, tmp.name)
            full_text = transcribe_audio_whisper(tmp.name)
            transcript = [{"start": 0, "text": full_text}]

    # Summarize in chunks
    st.info("Processing transcript into chunks and summarizing...")
    chunks = chunk_transcript(transcript)
    all_summaries = []
    for timestamp, chunk in chunks:
        summary = summarize_chunk(chunk, timestamp)
        all_summaries.append(summary)

    final_summary = "\n\n".join(all_summaries)

    # Display
    st.subheader("📝 Summary")
    st.text_area("Summary Output", final_summary, height=400)

    # PDF Export
    pdf_file = export_pdf(final_summary, yt.title)
    with open(pdf_file, "rb") as f:
        st.download_button("📥 Download PDF", f, file_name="summary.pdf")

    # Copy button
    st.code(final_summary)

