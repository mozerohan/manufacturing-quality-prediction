
FROM python:3.12-slim

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and trained model
COPY src/ ./src/
COPY models/ ./models/

# Copy automated tests
COPY tests/ ./tests/

# Expose the API port
EXPOSE 8000

# Start the FastAPI application
CMD ["python", "-m", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
