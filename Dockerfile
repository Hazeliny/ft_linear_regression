FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install matplotlib numpy
CMD ["python3", "train.py"]