FROM docker.io/tensorflow/tensorflow:2.17.0-gpu

USER root

RUN groupadd -g 4001 appuser && \
    useradd -m -u 4001 -g 4001 appuser && \
    mkdir /{app,llm} && \
    chown appuser:appuser /{app,llm}

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    tini && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt

RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt

COPY run_gpt2.py ./run_gpt2.py

USER appuser
ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["python3", "/app/run_gpt2.py"]
