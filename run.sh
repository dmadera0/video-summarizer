#!/bin/bash

# Simple run script for YouTube Summarizer
# Usage: ./run.sh

if [ ! -d "venv" ]; then
  echo "⚡ Virtual environment not found. Creating one..."
  python3 -m venv venv
  source venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
else
  source venv/bin/activate
fi

echo "🚀 Starting Streamlit app..."
streamlit run app.py
