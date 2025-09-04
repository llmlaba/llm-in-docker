FROM docker.io/pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

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
COPY environment.yml ./environment.yml

RUN conda env update -n base -f environment.yml

COPY run_mistral.py ./run_mistral.py

USER appuser
ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["python3", "/app/run_mistral.py"]
