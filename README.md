# LLM in docker examples

## Reqirments
- AMD Mi50/MI100 32Gb VRAM
- Workstation 40 GB RAM, 1TB SSD, 750W Power supply 
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

- Get the Mathstral GGUF
```bash
git lfs install
git clone https://huggingface.co/lmstudio-community/mathstral-7B-v0.1-GGUF mathstral
```

## Use cases

### llama.cpp in docker with CPU
> Notes:
> - Tested CPU Intel(R) Xeon(R) CPU E5-2630

- Run container
```bash
cd llama.cpp-cpu
./docker_env.sh up
```

- Check logs
```bash
docker container logs llamacpp-cpu_llamacpp-cpu.local_1
```

- Test request 
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "",
    "messages": [{"role": "user", "content": "Continue this text: What you know about sun?"}],
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

### llama.cpp in docker with Vulkan
> Notes:
> - Tested GPU AMD Mi50

- Run container
```bash
cd llama.cpp-vulkan
./docker_env.sh up
```

- Check logs
```bash
docker docker container logs llamacpp-vulkan_llamacpp-vulkan.local_1
```

- Test request 
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "",
    "messages": [{"role": "user", "content": "Continue this text: What you know about sun?"}],
    "max_tokens": 360,
    "temperature": 0.7,
    "top_p": 0.95,
    "stop": "eof"
  }' | jq

```

- Stop container
```bash
./docker_env.sh down
```

### PyTorch in docker with AMD ROCm
> Notes:
> - Tested GPU AMD Mi50

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
> Notes:
> - Tested GPU AMD Mi50

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
> Notes:
> - Tested GPU Nvidia Tesla V100

- Run container
```bash
cd pytorch-cuda
./docker_env.sh up
```

- Check logs
```bash
docker container logs pytorch-cuda_pytorch-cuda.local_1
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

### TensorFlow in docker with NVIDIA CUDA
> Notes:
> - Tested GPU Nvidia Tesla V100

- Run container
```bash
cd tensorflow-cuda
./docker_env.sh up
```

- Check logs
```bash
docker container logs tensorflow-cuda_tensorflow-rocm.local_1
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
