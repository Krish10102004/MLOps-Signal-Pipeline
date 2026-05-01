FROM python:3.9-slim

# Non-root user for security best practice
RUN useradd -m -u 1000 mluser

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and data
COPY run.py config.yaml data.csv ./

# Switch to non-root
USER mluser

# Default run: uses bundled files, writes output inside container
CMD ["python", "run.py", \
     "--input",    "data.csv", \
     "--config",   "config.yaml", \
     "--output",   "metrics.json", \
     "--log-file", "run.log"]
