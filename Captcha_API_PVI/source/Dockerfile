FROM python:3.6

RUN pip3 install --upgrade pip

COPY requirements.txt /

RUN apt-get update ##[edited]
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip3 install -r /requirements.txt

ADD . /code

WORKDIR /code


ENTRYPOINT python app.py

