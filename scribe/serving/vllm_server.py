"""Launch and manage a vLLM server process.

vLLM gives ~120-160 tok/sec on A100 80GB for Qwen2.5-VL-7B.
Uses OpenAI-compatible /v1/chat/completions endpoint.
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
    port: int = 8000,
    served_model_name: str = "enclave-scribe",
    gpu_memory_utilization: float = 0.90,
    max_model_len: int = 32768,
    tensor_parallel_size: int = 1,
    timeout: int = 300,
    log_file: str = "log/vllm.log",
) -> subprocess.Popen | None:
    url = f"http://127.0.0.1:{port}"

    if is_ready(url):
        print(f"vLLM server already running at {url}")
        return None

    import os
    os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model",                    model_path,
        "--served-model-name",        served_model_name,
        "--host",                     host,
        "--port",                     str(port),
        "--dtype",                    "bfloat16",
        "--gpu-memory-utilization",   str(gpu_memory_utilization),
        "--max-model-len",            str(max_model_len),
        "--tensor-parallel-size",     str(tensor_parallel_size),
        "--trust-remote-code",
        "--enable-chunked-prefill",
        "--limit-mm-per-prompt",      "image=20",   # allow up to 20 images per request
    ]

    print(f"Starting vLLM server (TP={tensor_parallel_size}, port={port})...")
    log_f = open(log_file, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
    proc._log_file = log_f

    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            log_f.flush()
            raise RuntimeError(f"vLLM server exited early. Check {log_file}")
        if is_ready(url):
            print(f"vLLM ready ({time.time() - start:.0f}s) — {url}")
            return proc
        time.sleep(3)

    stop(proc)
    raise TimeoutError(f"vLLM server did not start within {timeout}s. Check {log_file}")


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
