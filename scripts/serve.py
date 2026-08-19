#!/usr/bin/env python3
"""Unified serve launcher: starts the chosen inference backend then the FastAPI gateway.

Usage:
    python scripts/serve.py --backend sglang --model /path/to/model
    python scripts/serve.py --backend vllm   --model /path/to/model
    python scripts/serve.py --backend local  --model /path/to/model

Environment variables set automatically:
    SCRIBE_BACKEND   — propagated to the FastAPI app
    SCRIBE_SERVER_URL — propagated to the FastAPI app
    MODEL_PATH       — propagated to the FastAPI app (local backend)
"""
import argparse
import os
import signal
import sys

import uvicorn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["local", "vllm", "sglang"], default="sglang")
    p.add_argument("--model", required=True, help="Path or HF hub name of the model")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--api-port", type=int, default=8080, help="FastAPI port")
    p.add_argument("--backend-port", type=int, default=None, help="Override inference backend port")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90, help="vLLM only")
    p.add_argument("--mem-fraction-static", type=float, default=0.80, help="SGLang only")
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM only")
    p.add_argument("--timeout", type=int, default=300, help="Backend startup timeout (s)")
    p.add_argument("--config", default=None, help="Optional YAML config (overrides defaults)")
    return p.parse_args()


def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()

    if args.config:
        cfg = load_config(args.config)
        for k, v in cfg.items():
            if not hasattr(args, k.replace("-", "_")):
                continue
            setattr(args, k.replace("-", "_"), v)

    backend_proc = None

    if args.backend == "vllm":
        from scribe.serving.vllm_server import launch, stop
        port = args.backend_port or 8000
        server_url = f"http://127.0.0.1:{port}"
        backend_proc = launch(
            model_path=args.model,
            host=args.host,
            port=port,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            timeout=args.timeout,
        )

    elif args.backend == "sglang":
        from scribe.serving.sglang_server import launch, stop
        port = args.backend_port or 10000
        server_url = f"http://127.0.0.1:{port}"
        backend_proc = launch(
            model_path=args.model,
            host=args.host,
            port=port,
            mem_fraction_static=args.mem_fraction_static,
            context_length=args.max_model_len,
            timeout=args.timeout,
        )

    else:  # local
        server_url = ""
        stop = None

    os.environ["SCRIBE_BACKEND"] = args.backend
    os.environ["MODEL_PATH"] = args.model
    if server_url:
        os.environ["SCRIBE_SERVER_URL"] = server_url

    def _shutdown(sig, frame):
        if stop and backend_proc:
            stop(backend_proc)
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print(f"Starting FastAPI gateway on {args.host}:{args.api_port} (backend={args.backend})")
    try:
        uvicorn.run(
            "api.server:app",
            host=args.host,
            port=args.api_port,
            log_level="info",
        )
    finally:
        if stop and backend_proc:
            stop(backend_proc)


if __name__ == "__main__":
    main()
