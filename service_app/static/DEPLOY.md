# Deploy Standalone `service_app` Demo

This guide is for running the demo on a remote Linux machine.

## 1. Clone the repository and enter it

```bash
git clone https://github.com/IINemo/thinkbooster.git
cd thinkbooster
```

Then follow environment basics from [README.md](README.md) (create/activate Python env, copy `.env`, add API keys).

## 2. Remove Ubuntu's `python3-blinker` package

```bash
sudo apt remove python3-blinker
```

Why: distro-level `python3-blinker` can shadow the `pip` version inside your Python environment and cause dependency/runtime conflicts. Removing it avoids that package resolution clash.

## 3. Run project setup

```bash
./setup.sh
```

## 4. Start a persistent terminal session

```bash
tmux new -t thinkbooster
```

Inside that `tmux` session, start the service.

### Basic mode (API-only strategies, no GPU needed)

```bash
python service_app/main.py
```

This supports `self_consistency` via external API providers (OpenRouter, OpenAI, etc.).

### PRM-only mode (single GPU, API-based generation + PRM scoring)

Uses an external API (OpenRouter/OpenAI) for generation and a local PRM model for scoring. This is the recommended single-GPU setup:

```bash
export CUDA_VISIBLE_DEVICES=0                             # GPU for the PRM model
export PRM_MODEL_PATH=Qwen/Qwen2.5-Math-PRM-7B           # HF model id or local path
export PRM_DEVICE=cuda:0                                  # device the scorer runs on
export PRM_GPU_MEMORY_UTILIZATION=0.8                     # fraction of GPU mem for PRM
export VLLM_WORKER_MULTIPROC_METHOD=spawn                 # avoids CUDA init errors
python service_app/main.py
```

### Full mode with PRM scorer (requires GPU)

To enable the PRM scorer (`tts_scorer=prm`) alongside a local vLLM generation model, you need to allocate GPU(s) and point the service at the PRM model weights:

```bash
export CUDA_VISIBLE_DEVICES=0                             # GPU for the PRM model
export PRM_MODEL_PATH=Qwen/Qwen2.5-Math-PRM-7B           # HF model id or local path
export PRM_DEVICE=cuda:0                                  # device the scorer runs on
export PRM_GPU_MEMORY_UTILIZATION=0.8                     # fraction of GPU mem for PRM
export VLLM_WORKER_MULTIPROC_METHOD=spawn                 # avoids CUDA init errors
python service_app/main.py
```

If you also run a local vLLM model alongside PRM on the **same** GPU, make sure the combined `VLLM_GPU_MEMORY_UTILIZATION + PRM_GPU_MEMORY_UTILIZATION` stays below `0.95` to avoid OOM. Alternatively, put each on a separate GPU:

```bash
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_MODEL_PATH=Qwen/Qwen2.5-Coder-7B-Instruct    # main model on cuda:0
export VLLM_GPU_MEMORY_UTILIZATION=0.9
export PRM_MODEL_PATH=Qwen/Qwen2.5-Math-PRM-7B            # PRM scorer on cuda:1
export PRM_DEVICE=cuda:1
export PRM_GPU_MEMORY_UTILIZATION=0.9
python service_app/main.py
```

Service URLs (default):
- Home: `http://<server-ip>:8080/`
- API docs: `http://<server-ip>:8080/docs`
- Deploy guide route: `http://<server-ip>:8080/deploy`

## 5. Logs

Each service run creates a timestamped log directory:

```
logs/<date>/<time>/service.log
```

For example: `logs/2026-03-04/15-20-14/service.log`. To tail the latest log:

```bash
tail -f logs/$(ls -t logs/ | head -1)/$(ls -t logs/$(ls -t logs/ | head -1) | head -1)/service.log
```

**Note:** You can change it to port 80/443 if you want user to access it directly through HTTP/HTTPS.

## Common Problems

### CUDA init error fallback

If you hit:

```text
RuntimeError: CUDA driver initialization failed, you might not have a CUDA gpu.
```

run:

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

Then re-run:

```bash
python service_app/main.py
```
