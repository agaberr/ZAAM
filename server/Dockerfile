FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download nltk data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy application code
COPY . .

# Make the startup script executable
RUN chmod +x run.sh

# Environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV OAUTHLIB_INSECURE_TRANSPORT=1
ENV PORT=10000

# Expose the port (Render will override with its own PORT)
EXPOSE 10000
EXPOSE $PORT

# Command to run the application
# Render will use the CMD defined here
CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers 4 --timeout 120 