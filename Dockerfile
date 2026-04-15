FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY templates/ templates/
COPY static/ static/
COPY models/ models/

# Create uploads directory
RUN mkdir -p static/uploads

# Expose port
# EXPOSE 8080
EXPOSE 7860

# Set environment variables
ENV FLASK_APP=app.py
ENV PYTHONUNBUFFERED=1

# Run with gunicorn
# CMD ["gunicorn", "-b", "0.0.0.0:8080", "-w", "2", "--timeout", "120", "app:app"]
# CMD ["sh", "-c", "gunicorn -b 0.0.0.0:${PORT:-8080} -w 1 --timeout 120 app:app"]
CMD ["sh", "-c", "gunicorn -b 0.0.0.0:${PORT:-7860} -w 1 app:app"]
