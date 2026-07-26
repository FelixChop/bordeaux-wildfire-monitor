FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=5000
ENV NASA_FIRMS_MAP_KEY=${NASA_FIRMS_MAP_KEY}

EXPOSE 5000

# Single worker + threads: one shared in-memory cache (wind field, vegetation,
# simulation) and one background data-fetch thread. Plenty for this app's load.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "90", "app:app"]
