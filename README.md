# 🎥 YouTube Summarizer

[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![Postgres](https://img.shields.io/badge/Postgres-Database-336791?logo=postgresql)](https://www.postgresql.org/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?logo=openai)](https://openai.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An AI-powered web app that generates **summaries and transcripts** of YouTube videos.  
Built with **Streamlit**, **OpenAI GPT-4o-mini**, and **Postgres** for storage.

---

## ✨ Features
- 🔗 Paste any YouTube link to generate:
  - Transcript (from YouTube captions or Whisper fallback)
  - AI-powered summary with timestamps
- 📚 Save summaries to Postgres and browse them in the **History tab**
- 📥 Export summaries as PDF
- 🎨 YouTube-inspired UI with centered title + red top bar
- 🎙️ Whisper fallback for videos without transcripts (auto-chunked for >10min videos)

---

## 🛠️ Tech Stack
- **Frontend/UI**: [Streamlit](https://streamlit.io/)
- **AI Models**: [OpenAI GPT-4o-mini](https://platform.openai.com/)
- **Transcripts**: [YouTube Transcript API](https://github.com/jdepoix/youtube-transcript-api), [Whisper](https://openai.com/research/whisper)
- **Database**: PostgreSQL (via SQLAlchemy ORM)
- **Others**: `yt-dlp`, `pydub`, `fpdf`

---

## 🚀 Getting Started (Local)

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/video-summarizer.git
cd video-summarizer

### 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate


### 3. Install dependencies
pip install -r requirements.txt


### 4. Set up environment variables
Create a .env file in the project root:
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
DATABASE_URL=postgresql+psycopg2://user:password@localhost:5432/youtube_summarizer

### 5. Run Postgres locally
psql -U postgres
CREATE DATABASE youtube_summarizer;

### 6. Start the app
./run.sh
