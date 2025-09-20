# -------------------------
# 1. Base Image
# -------------------------
FROM python:3.11-slim

# -------------------------
# 2. Install system packages
# -------------------------
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# -------------------------
# 3. Set working directory
# -------------------------
WORKDIR /app

# -------------------------
# 4. Install dependencies
# -------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------
# 5. Copy app code
# -------------------------
COPY . .

# -------------------------
# 6. Expose Streamlit port
# -------------------------
EXPOSE 8501

# -------------------------
# 7. Run Streamlit
# -------------------------
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
