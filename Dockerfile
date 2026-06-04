FROM python:3.12-slim

WORKDIR /app

COPY . .

RUN pip install uv

RUN uv sync

EXPOSE 5000

CMD ["uv", "run", "python", "model_final/app.py"]