
# syntax=docker/dockerfile:1

FROM python:3.9.6-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

RUN apt-get update ##[edited]
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY . .

#start web app
CMD [ "python", "-m" , "flask", "run", "--host=0.0.0.0"]
