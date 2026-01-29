# Use an official Python runtime as a parent image
FROM python:3.10-slim

# set environment
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# create working directory
WORKDIR /app

# install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy code
COPY . .

# generate model during build (so image contains models/model.joblib)
RUN python scripts/train_model.py

# expose port
ARG PORT=8000
ENV PORT=${PORT}
EXPOSE ${PORT}

# default CMD
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]