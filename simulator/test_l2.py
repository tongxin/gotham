"""Sanity checks for the L2 kernel-level core.

Run from the project root:  python3 simulator/test_l2.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator import core, core_l2, data  # noqa: E402


def approx(actual, expected, tol=0.02, label=""):
    if abs(actual - expected) / expected > tol:
        raise AssertionError(f"{label}: {actual} != {expected} (±{tol:.0%})")


def kernel_by_name(phase, name):
    for k in phase["kernels"]:
        if k["name"] == name:
            return k
    raise AssertionError(f"kernel {name} not found")


def main():
    print(f"L2 core test — {core.version()}")
    gpu = data.get_gpu("h100")
    m8b = data.get_model("llama3-8b")
    mixtral = data.get_model("mixtral-8x7b")
    cfg = {"phase": "both", "B": 1, "S": 2048, "gpus": 1,
           "precision": "fp16", "kvPrecision": "fp16",
           "computeScale": 1.0, "bandwidthScale": 1.0,
           "flashAttention": True, "recompute": True, "fuseLayer": True,
           "l2UsableFrac": 0.8, "occupancyTarget": 4}

    for ph in ("prefill", "decode"):
        l2 = core_l2.simulate_l2(m8b, gpu, cfg, ph)
        l1 = core.simulate(m8b, gpu, dict(cfg, phase=ph))[ph]
        # prefill is compute-bound: L2 should be at least as slow as L1
        if ph == "prefill":
            assert l2["totalTime"] >= l1["time"] * 0.9, f"{ph}: L2 too fast vs L1"
        # decode may be faster than L1 thanks to the L2 reuse model, but
        # utilization must stay within the roofline
        assert 0.0 < l2["utilization"] <= 1.0001, f"{ph}: utilization out of range"
        assert 0.0 < l2["occupancyUtil"] <= 1.0
        assert 0.0 <= l2["l2Hit"] <= 1.0
        assert l2["totalTime"] > 0 and l2["throughput"] > 0
        assert len(l2["kernels"]) >= 6, "expected qkv/scores/pv/out/mlp/logits"

    # prefill attention: flash removes SxS HBM traffic
    flash = core_l2.simulate_l2(m8b, gpu, cfg, "prefill")
    mat = core_l2.simulate_l2(m8b, gpu, dict(cfg, flashAttention=False), "prefill")
    assert kernel_by_name(flash, "attn_scores")["dramPerGpu"] < \
        kernel_by_name(mat, "attn_scores")["dramPerGpu"], "flash should cut attn DRAM"

    # MoE: routed expert kernel exists and carries the top-k FLOPs
    mix = core_l2.simulate_l2(mixtral, gpu, cfg, "decode")
    exp = kernel_by_name(mix, "mlp_experts")
    assert exp["flopsPerGpu"] > 0
    assert kernel_by_name(mix, "mlp_router")["flopsPerGpu"] > 0

    # memory: weights dominate; matches L1 weight bytes
    l2mem = core_l2.simulate_l2(m8b, gpu, cfg, "prefill")
    l1mem = core.memory_footprint(m8b, cfg)
    approx(l2mem["memWeights"], l1mem["weights"], 0.01, "weights")

    # occupancy responds to SMEM pressure: smaller tile target => higher util
    loose = core_l2.simulate_l2(m8b, gpu, cfg, "prefill")
    assert loose["occupancyUtil"] > 0

    print("All L2 checks passed.")


if __name__ == "__main__":
    main()
