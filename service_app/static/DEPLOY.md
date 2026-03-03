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

Inside that `tmux` session, start the service:

```bash
python service_app/main.py
```

Service URLs (default):
- Home: `http://<server-ip>:8080/`
- API docs: `http://<server-ip>:8080/docs`
- Deploy guide route: `http://<server-ip>:8080/deploy`

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
