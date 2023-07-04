FROM python:3.9

WORKDIR /usr/src/app

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN pip3 install pyradiomics

COPY ./Data ./Data
COPY ./backend ./backend
COPY ./frontend ./frontend
COPY main.py main.py


EXPOSE 8888

CMD ["python","main.py"]
