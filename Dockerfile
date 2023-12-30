FROM python:3.10.12

COPY requirements.txt ./app

RUN pip install -r ./app/requirements.txt