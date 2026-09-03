"""Sanity checks for the C++ core via the Python binding.

Run from the project root:  python3 simulator/test_core.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator import core, data  # noqa: E402


def approx(actual, expected, tol=0.02, label=""):
    if abs(actual - expected) / expected > tol:
        raise AssertionError(f"{label}: {actual} != {expected} (±{tol:.0%})")


def main():
    print(f"C++ core version: {core.version()}")

    gpu = data.get_gpu("h100")
    m8b = data.get_model("llama3-8b")
    m70b = data.get_model("llama3-70b")
    mixtral = data.get_model("mixtral-8x7b")
    cfg = {"phase": "both", "B": 1, "S": 2048, "gpus": 1,
           "precision": "fp16", "kvPrecision": "fp16",
           "computeScale": 1.0, "bandwidthScale": 1.0}

    # Peaks and ridge point
    ridge = core.peak_flops(gpu, "fp16", 1.0) / (gpu["bandwidth_GBps"] * 1e9)
    approx(ridge, 295.4, 0.01, "ridge")
    approx(core.peak_flops(gpu, "fp8", 1.0), 1979e12, 0.01, "fp8 peak")
    approx(core.peak_flops(gpu, "int4", 1.0), 4 * 989.5e12, 0.01, "int4 peak")
    approx(core.peak_flops(gpu, "fp32", 1.0), 67e12, 0.01, "fp32 peak")

    # LLaMA-3 8B prefill: compute-bound at peak, ~57.7k tok/s
    s = core.simulate(m8b, gpu, cfg)
    approx(s["prefill"]["intensity"], 2081, 0.02, "prefill intensity")
    approx(s["prefill"]["achieved"], 989.5e12, 0.01, "prefill achieved")
    assert s["prefill"]["bound"] == "compute", "prefill should be compute-bound"
    approx(s["prefill"]["throughput"], 57752, 0.05, "prefill tok/s")

    # LLaMA-3 8B decode at B=1: memory-bound, ~196 tok/s
    approx(s["decode"]["intensity"], 1.0, 0.03, "decode intensity")
    assert s["decode"]["bound"] == "memory", "decode should be memory-bound"
    approx(s["decode"]["achieved"], 3433e9, 0.02, "decode achieved")
    approx(s["decode"]["throughput"], 196, 0.06, "decode tok/s")

    # 70B does not fit one H100; fits two
    mem1 = core.memory_footprint(m70b, cfg)
    assert mem1["total"] > gpu["memory_GB"] * 1e9, "70B should OOM one H100"
    cfg2 = dict(cfg, gpus=2)
    mem2 = core.memory_footprint(m70b, cfg2)
    assert mem2["total"] / 2 < gpu["memory_GB"] * 1e9, "70B should fit two H100s"

    # Mixtral: total params drive weight bytes, active params drive FLOPs
    sm = core.simulate(mixtral, gpu, cfg)
    assert sm["decode"]["bound"] == "memory"
    approx(sm["wBytes"], 46.7e9 * 2, 0.01, "mixtral weights")
    # MoE decode streams the per-token active weight set, not all 46.7B
    approx(sm["decodeWBytes"], 12.9e9 * 2, 0.03, "mixtral decode stream weights")
    approx(sm["decode"]["intensity"], 1.0, 0.06, "mixtral decode intensity")
    approx(sm["decode"]["throughput"], 3350e9 / (12.9e9 * 2), 0.06, "mixtral decode tok/s")

    # Dense models still stream the full weight set per decode step
    s8 = core.simulate(m8b, gpu, cfg)
    approx(s8["decodeWBytes"], s8["wBytes"], 1e-9, "dense decode stream weights")

    # Sweep: 44 points, monotonic intensity, first ~1 FLOP/B
    pts = core.decode_sweep(m8b, gpu, cfg)
    assert len(pts) == 44
    assert pts[0]["B"] == 1 and pts[-1]["B"] == 1024
    assert all(pts[i]["x"] <= pts[i + 1]["x"] for i in range(len(pts) - 1))
    approx(pts[0]["x"], 1.0, 0.03, "sweep first intensity")

    # GQA KV-cache savings: Mistral kv_dim == H/4
    mistral = data.get_model("mistral-7b")
    smi = core.simulate(mistral, gpu, cfg)
    assert abs(smi["kvDim"] - 1024) < 1e-9

    print("All core checks passed.")


if __name__ == "__main__":
    main()
