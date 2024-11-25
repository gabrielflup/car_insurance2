FROM python:3.8-slim-buster

RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
