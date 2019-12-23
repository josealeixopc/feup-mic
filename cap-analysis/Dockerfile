FROM python:3.7.5

RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install -y libopenmpi-dev


COPY ./requirements.txt /

RUN pip install -r requirements.txt

COPY . /

CMD [ "python", "./run_training.py" ]