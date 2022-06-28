FROM ubuntu
RUN apt update && apt install -y python3 python3-pip python-is-python3 git

COPY requirements.txt /requirements.txt
RUN pip install jax && pip install -r requirements.txt

COPY dataloader/ /dataloader/
RUN python -c "\
import dataloader; \
import logging; \
logging.basicConfig(level=logging.INFO); \
dataloader.download('/app/data', 'mnist'); \
dataloader.download('/app/data', 'cifar10'); \
dataloader.download('/app/data', 'kddcup99'); \
"

COPY . /app
WORKDIR /app
ENTRYPOINT ["./experiments.sh"]
