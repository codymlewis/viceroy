FROM ubuntu
RUN apt update && apt install -y python3 python3-pip python-is-python3 git

COPY requirements.txt /requirements.txt
RUN pip install jax && pip install -r requirements.txt

COPY . /app
WORKDIR /app
ENTRYPOINT ["./experiments.sh"]