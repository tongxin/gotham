"""Local HTTP server: serves the UI and exposes /api/simulate over the C++ core."""

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from . import core, data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MIME = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".json": "application/json; charset=utf-8",
}


def _safe_path(path):
    candidate = os.path.realpath(os.path.join(ROOT, path.lstrip("/")))
    if candidate == ROOT or candidate.startswith(ROOT + os.sep):
        return candidate
    return None


def _build_ceilings(gpu, cfg, compute_scale):
    bw = gpu["bandwidth_GBps"] * 1e9 * float(cfg.get("bandwidthScale", 1.0))
    peak = core.peak_flops(gpu, cfg["precision"], compute_scale)
    prec_label = {p["id"]: p["label"] for p in data.PRECISIONS}
    ceilings = [{
        "kind": "primary",
        "id": cfg["precision"],
        "peak": peak,
        "bandwidth": bw,
        "label": f"{gpu['name']} · {prec_label[cfg['precision']]}",
        "primary": True,
    }]
    if cfg.get("showPrecisionCeilings", True):
        for p in data.PRECISIONS:
            if p["id"] == cfg["precision"]:
                continue
            if p.get("needsFp8") and not gpu["fp8_TFLOPS"]:
                continue
            ceilings.append({
                "kind": "precision",
                "id": p["id"],
                "peak": core.peak_flops(gpu, p["id"], compute_scale),
                "bandwidth": bw,
                "label": prec_label[p["id"]],
                "primary": False,
            })
    if cfg.get("showOtherCeilings", True):
        for other in data.GPUS:
            if other["id"] == gpu["id"]:
                continue
            ceilings.append({
                "kind": "gpu",
                "id": other["id"],
                "peak": other["fp16_TFLOPS"] * 1e12,
                "bandwidth": other["bandwidth_GBps"] * 1e9,
                "label": other["name"],
                "primary": False,
            })
    return ceilings


def _simulate_payload(payload):
    cfg = payload.get("cfg") or {}
    gpu_id = payload.get("gpu") or "h100"
    gpu = data.get_gpu(gpu_id)
    model_ids = payload.get("models") or []
    if not model_ids:
        raise ValueError("no models selected")
    models = [data.get_model(mid) for mid in model_ids]

    gpus = int(cfg.get("gpus", 1))
    compute_scale = float(cfg.get("computeScale", 1.0))

    results = []
    for m in models:
        sim = core.simulate(m, gpu, cfg)
        mem = core.memory_footprint(m, cfg)
        results.append({
            "model": m,
            "prefill": sim["prefill"],
            "decode": sim["decode"],
            "memory_per_gpu": {
                "weights": mem["weights"] / gpus,
                "kv": mem["kv"] / gpus,
                "act": mem["act"] / gpus,
                "total": mem["total"] / gpus,
            },
        })

    sweeps = []
    if cfg.get("sweep") and cfg.get("phase", "both") in ("decode", "both"):
        for m in models:
            sweeps.append({"model": m["id"], "points": core.decode_sweep(m, gpu, cfg)})

    sim0 = core.simulate(models[0], gpu, cfg)
    return {
        "gpu": gpu,
        "peak": sim0["peak"],
        "bw": sim0["bw"],
        "ridge": sim0["ridge"],
        "coreVersion": core.version(),
        "ceilings": _build_ceilings(gpu, cfg, compute_scale),
        "results": results,
        "sweeps": sweeps,
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "GothamSim/0.1"

    def log_message(self, fmt, *args):
        pass

    def _send(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj, status=200):
        self._send(status, json.dumps(obj).encode(), "application/json; charset=utf-8")

    def _send_error_json(self, status, message):
        self._send_json({"error": message}, status)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/data":
            self._send_json(data.catalog())
            return
        if path in ("", "/"):
            path = "/index.html"
        target = _safe_path(path)
        if not target or not os.path.isfile(target):
            self._send_error_json(404, "not found")
            return
        ext = os.path.splitext(target)[1].lower()
        with open(target, "rb") as fh:
            self._send(200, fh.read(), MIME.get(ext, "application/octet-stream"))

    def do_POST(self):
        if urlparse(self.path).path != "/api/simulate":
            self._send_error_json(404, "not found")
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            self._send_json(_simulate_payload(payload))
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            self._send_error_json(400, str(exc))
        except Exception as exc:  # noqa: BLE001 - report any server-side failure
            self._send_error_json(500, str(exc))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Gotham roofline simulator server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    print(f"Gotham simulator (core: {core.version()}) → http://{args.host}:{args.port}")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
