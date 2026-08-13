"""Launch and manage an SGLang server process.

SGLang gives ~150-200 tok/sec on A100 80GB for Qwen2.5-VL-7B.
Supports custom n-gram logit processor — critical for OCR repetition suppression.
"""
import subprocess
import sys
import time

import requests


def is_ready(url: str) -> bool:
    try:
        return requests.get(f"{url}/health", timeout=3).status_code == 200
    except Exception:
        return False


def launch(
    model_path: str,
    host: str = "0.0.0.0",
    port: int = 10000,
    served_model_name: str = "enclave-scribe",
    mem_fraction_static: float = 0.80,
    context_length: int = 32768,
    attention_backend: str = "fa3",
    timeout: int = 300,
    log_file: str = "log/sglang.log",
) -> subprocess.Popen | None:
    url = f"http://127.0.0.1:{port}"

    if is_ready(url):
        print(f"SGLang server already running at {url}")
        return None

    import os
    os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)

    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model",                  model_path,
        "--served-model-name",      served_model_name,
        "--host",                   host,
        "--port",                   str(port),
        "--attention-backend",      attention_backend,
        "--page-size",              "1",
        "--mem-fraction-static",    str(mem_fraction_static),
        "--context-length",         str(context_length),
        "--enable-custom-logit-processor",
        "--disable-overlap-schedule",
        "--skip-server-warmup",
    ]

    print(f"Starting SGLang server (port={port})...")
    log_f = open(log_file, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
    proc._log_file = log_f

    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            log_f.flush()
            raise RuntimeError(f"SGLang server exited early. Check {log_file}")
        if is_ready(url):
            print(f"SGLang ready ({time.time() - start:.0f}s) — {url}")
            return proc
        time.sleep(3)

    stop(proc)
    raise TimeoutError(f"SGLang server did not start within {timeout}s. Check {log_file}")


def stop(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    if hasattr(proc, "_log_file"):
        proc._log_file.close()
