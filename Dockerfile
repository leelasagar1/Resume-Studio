FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY static ./static
COPY run.py .

# Inside the container the app must listen on all interfaces; docker-compose
# publishes it only on the host's 127.0.0.1.
ENV HOST=0.0.0.0
EXPOSE 8765
CMD ["python", "run.py"]
