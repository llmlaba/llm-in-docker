FROM docker.io/rocm/tensorflow:rocm6.2.2-py3.9-tf2.16.2-dev AS builder

USER root

WORKDIR /build

COPY meta-tensorflow.toml /build/tensorflow/pyproject.toml

RUN pip3 install --upgrade pip wheel setuptools build && \
    python3 -m build --wheel /build/tensorflow

FROM docker.io/rocm/tensorflow:rocm6.2.2-py3.9-tf2.16.2-dev

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

COPY --from=builder /build/tensorflow/dist/tensorflow-2.16.2.post1-py3-none-any.whl ./tensorflow-2.16.2.post1-py3-none-any.whl

RUN pip3 install --upgrade pip && \
    pip3 install ./tensorflow-2.16.2.post1-py3-none-any.whl && \
    pip3 install -r requirements.txt

COPY run_gpt2.py ./run_gpt2.py

USER appuser
ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["python3", "/app/run_gpt2.py"]
