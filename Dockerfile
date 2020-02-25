# FROM python:3.6-slim
FROM tensorflow/tensorflow:2.1.0-py3
WORKDIR /app

COPY . .
RUN pip install -e ./

ENTRYPOINT [ "tfrecord" ]
