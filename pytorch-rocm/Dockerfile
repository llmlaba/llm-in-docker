FROM docker.io/rocm/pytorch:rocm6.2.2_ubuntu22.04_py3.10_pytorch_release_2.3.0

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

RUN conda env update -n py_3.10 -f environment.yml

COPY run_mistral.py ./run_mistral.py

USER appuser
ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["python3", "/app/run_mistral.py"]
