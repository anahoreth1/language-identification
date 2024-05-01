#Deriving the latest base image
FROM python:latest



COPY ./requirements.txt /root/requirements.txt

RUN pip install --upgrade pip && \
    pip install --upgrade "jax[cpu]" && \
    pip install --ignore-installed -r /root/requirements.txt


WORKDIR /root/docker_test

COPY . /root/docker_test

CMD [ "python", "inference.py"]
