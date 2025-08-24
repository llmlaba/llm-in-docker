# LLM in docker examples

## Reqirments
- AMD Mi50/MI100 32Gb VRAM
- Workstation 40 GB RAM, 200GB SSD, 750W Power supply 
- Ubuntu 24.04 LTS
- Docker CE and docker compose

## Test environment
- My test environment: HP Z440 + AMD Mi50

## Preparation
- Get the Mistral
```bash
git lfs install
git clone https://huggingface.co/mistralai/Mistral-7B-v0.1 mistral
```

- Get the GPT2
```bash
git lfs install
git clone https://huggingface.co/openai-community/gpt2 gpt2
```

## Use cases

### llama.cpp in docker with CPU

### PyTorch in docker with AMD ROCm

- Run container
```bash
cd pytorch-rocm
./docker_env.sh up
```

- Check logs
```bash
docker container logs pytorch-rocm_pytorch-rocm.local_1
```

- Test request 
```bash
curl -s http://localhost:8080/v1/completion \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What you know about sun?",
    "max_tokens": 60,
    "temperature": 0.7,
    "top_p": 0.95,
    "stop": "eof"
  }' | jq

```

- Stop container
```bash
./docker_env.sh down
```

### TensorFlow in docker with AMD ROCm

- Run container
```bash
cd tensorflow-rocm
./docker_env.sh up
```

- Check logs
```bash
docker container logs tensorflow-rocm_tensorflow-rocm.local_1
```

- Test request 
```bash
curl -s http://localhost:8080/v1/completion \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What you know about sun?",
    "max_tokens": 60,
    "temperature": 0.7,
    "top_p": 0.95,
    "stop": "eof"
  }' | jq

```

- Stop container
```bash
./docker_env.sh down
```

### PyTorch in docker with NVIDIA CUDA

### TensorFlow in docker with NVIDIA CUDA
