FROM python:3.12-slim
WORKDIR /app
RUN pip install numpy
COPY sattva_engine.py .
COPY train.py .
CMD ["python", "train.py"]
