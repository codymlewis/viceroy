FROM ubuntu
RUN apt update && apt install -y python3 python3-pip python-is-python3 git

COPY requirements.txt /requirements.txt
RUN pip install jax && pip install -r requirements.txt

RUN python -c "\
import app.dataloader; \
import logging; \
logging.basicConfig(level=logging.INFO); \
app.dataloader.download('/app/data', 'mnist'); \
app.dataloader.download('/app/data', 'cifar10'); \
app.dataloader.download('/app/data', 'kddcup99'); \
"

COPY . /app
WORKDIR /app
ENTRYPOINT ["./experiments.sh"]
