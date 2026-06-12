from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check whether JAX can see the requested accelerator.")
    parser.add_argument("--require-cuda", action="store_true", help="Exit non-zero unless JAX is using a CUDA/GPU backend.")
    args = parser.parse_args(argv)
    try:
        import jax
    except ImportError as exc:
        print(json.dumps({"ok": False, "error": f"jax import failed: {exc}"}), file=sys.stderr)
        return 2
    backend = jax.default_backend()
    devices = [str(device) for device in jax.devices()]
    ok = (not args.require_cuda) or backend in {"gpu", "cuda"}
    payload = {"ok": ok, "backend": backend, "devices": devices, "jax_version": getattr(jax, "__version__", "unknown")}
    stream = sys.stdout if ok else sys.stderr
    print(json.dumps(payload, indent=2, sort_keys=True), file=stream)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
