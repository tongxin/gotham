"""ctypes binding to the L2 (kernel-level) C++ core in cpp/l2.cpp."""

import ctypes

from .core import _load_library


class L2Model(ctypes.Structure):
    _fields_ = [
        ("params_b", ctypes.c_double),
        ("active_b", ctypes.c_double),
        ("layers", ctypes.c_int),
        ("hidden", ctypes.c_int),
        ("heads", ctypes.c_int),
        ("kv_heads", ctypes.c_int),
        ("vocab", ctypes.c_int),
        ("ffn", ctypes.c_int),
        ("experts", ctypes.c_int),
        ("topk", ctypes.c_int),
        ("shared_experts", ctypes.c_int),
    ]


class L2Gpu(ctypes.Structure):
    _fields_ = [
        ("fp16_tflops", ctypes.c_double),
        ("fp8_tflops", ctypes.c_double),
        ("fp32_tflops", ctypes.c_double),
        ("bandwidth_gbps", ctypes.c_double),
        ("memory_gb", ctypes.c_double),
        ("sm_count", ctypes.c_int),
        ("clock_ghz", ctypes.c_double),
        ("smem_per_sm", ctypes.c_double),
        ("regs_per_sm", ctypes.c_double),
        ("l1_per_sm", ctypes.c_double),
        ("l2_bytes", ctypes.c_double),
        ("l2_bw_gbps", ctypes.c_double),
        ("tmem_per_sm", ctypes.c_double),
        ("smem_bw_per_clk", ctypes.c_double),
        ("max_threads_per_sm", ctypes.c_int),
        ("max_warps_per_sm", ctypes.c_int),
        ("max_blocks_per_sm", ctypes.c_int),
    ]


class L2Workload(ctypes.Structure):
    _fields_ = [
        ("phase", ctypes.c_int),
        ("B", ctypes.c_int),
        ("S", ctypes.c_int),
        ("gpus", ctypes.c_int),
        ("precision", ctypes.c_int),
        ("kv_precision", ctypes.c_int),
        ("compute_scale", ctypes.c_double),
        ("bandwidth_scale", ctypes.c_double),
        ("flash_attention", ctypes.c_int),
        ("recompute", ctypes.c_int),
        ("fuse_layer", ctypes.c_int),
        ("q_tile", ctypes.c_int),
        ("k_tile", ctypes.c_int),
        ("l2_usable_frac", ctypes.c_double),
        ("occupancy_target_blocks", ctypes.c_int),
        ("threads_per_block", ctypes.c_int),
        ("regs_per_thread", ctypes.c_int),
    ]


class L2Kernel(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char * 32),
        ("flops_g", ctypes.c_double),
        ("dram_g", ctypes.c_double),
        ("smem_g", ctypes.c_double),
        ("t_compute", ctypes.c_double),
        ("t_dram", ctypes.c_double),
        ("t_smem", ctypes.c_double),
        ("t_l2", ctypes.c_double),
        ("t_total", ctypes.c_double),
        ("achieved", ctypes.c_double),
        ("occupancy_util", ctypes.c_double),
        ("bound", ctypes.c_int),
        ("valid", ctypes.c_int),
    ]


class L2Result(ctypes.Structure):
    _fields_ = [
        ("kernels", L2Kernel * 24),
        ("n_kernels", ctypes.c_int),
        ("layer_flops", ctypes.c_double),
        ("layer_dram_g", ctypes.c_double),
        ("layer_smem_g", ctypes.c_double),
        ("layer_time", ctypes.c_double),
        ("total_time", ctypes.c_double),
        ("tokens", ctypes.c_double),
        ("throughput", ctypes.c_double),
        ("latency_per_token", ctypes.c_double),
        ("achieved", ctypes.c_double),
        ("utilization", ctypes.c_double),
        ("p_eff", ctypes.c_double),
        ("t_compute_sum", ctypes.c_double),
        ("t_dram_sum", ctypes.c_double),
        ("t_smem_sum", ctypes.c_double),
        ("occupancy_util", ctypes.c_double),
        ("l2_hit", ctypes.c_double),
        ("mem_weights", ctypes.c_double),
        ("mem_kv", ctypes.c_double),
        ("mem_act", ctypes.c_double),
        ("mem_total", ctypes.c_double),
        ("ridge", ctypes.c_double),
        ("peak", ctypes.c_double),
        ("bw", ctypes.c_double),
        ("valid", ctypes.c_int),
    ]


from .core import PRECISION_ENUM, KV_ENUM  # noqa: E402

_lib = _load_library()
_lib.gotham_l2_peak_flops.argtypes = [ctypes.POINTER(L2Gpu), ctypes.c_int, ctypes.c_double]
_lib.gotham_l2_peak_flops.restype = ctypes.c_double
_lib.gotham_l2_simulate.argtypes = [
    ctypes.POINTER(L2Model),
    ctypes.POINTER(L2Gpu),
    ctypes.POINTER(L2Workload),
    ctypes.POINTER(L2Result),
]
_lib.gotham_l2_simulate.restype = ctypes.c_int


def _model(model):
    return L2Model(
        params_b=float(model["params"]),
        active_b=float(model.get("active") or 0.0),
        layers=int(model["layers"]),
        hidden=int(model["hidden"]),
        heads=int(model["heads"]),
        kv_heads=int(model.get("kv_heads") or 0),
        vocab=int(model.get("vocab") or 32000),
        ffn=int(model.get("ffn") or 0),
        experts=int(model.get("experts") or 0),
        topk=int(model.get("topk") or 0),
        shared_experts=int(model.get("shared") or 0),
    )


def _gpu(gpu):
    return L2Gpu(
        fp16_tflops=float(gpu["fp16_TFLOPS"]),
        fp8_tflops=float(gpu.get("fp8_TFLOPS") or 0.0),
        fp32_tflops=float(gpu["fp32_TFLOPS"]),
        bandwidth_gbps=float(gpu["bandwidth_GBps"]),
        memory_gb=float(gpu["memory_GB"]),
        sm_count=int(gpu.get("sm_count", 132)),
        clock_ghz=float(gpu.get("clock_ghz", 1.8)),
        smem_per_sm=float(gpu.get("smem_per_sm_kb", 228)) * 1024,
        regs_per_sm=float(gpu.get("regs_per_sm_kb", 256)) * 1024,
        l1_per_sm=float(gpu.get("l1_per_sm_kb", 256)) * 1024,
        l2_bytes=float(gpu.get("l2_mb", 50)) * 1024 * 1024,
        l2_bw_gbps=float(gpu.get("l2_bw_gbps", 7000)),
        tmem_per_sm=float(gpu.get("tmem_per_sm_kb", 0)) * 1024,
        smem_bw_per_clk=float(gpu.get("smem_bw_b_per_clk", 128)),
        max_threads_per_sm=int(gpu.get("max_threads_per_sm", 2048)),
        max_warps_per_sm=int(gpu.get("max_warps_per_sm", 64)),
        max_blocks_per_sm=int(gpu.get("max_blocks_per_sm", 32)),
    )


def _cfg(cfg, phase):
    return L2Workload(
        phase=0 if phase == "prefill" else 1,
        B=int(cfg.get("B", 1)),
        S=int(cfg.get("S", 2048)),
        gpus=int(cfg.get("gpus", 1)),
        precision=PRECISION_ENUM[cfg["precision"]],
        kv_precision=KV_ENUM[cfg.get("kvPrecision", "fp16")],
        compute_scale=float(cfg.get("computeScale", 1.0)),
        bandwidth_scale=float(cfg.get("bandwidthScale", 1.0)),
        flash_attention=1 if cfg.get("flashAttention", True) else 0,
        recompute=1 if cfg.get("recompute", True) else 0,
        fuse_layer=1 if cfg.get("fuseLayer", True) else 0,
        q_tile=int(cfg.get("qTile", 64)),
        k_tile=int(cfg.get("kTile", 64)),
        l2_usable_frac=float(cfg.get("l2UsableFrac", 0.8)),
        occupancy_target_blocks=int(cfg.get("occupancyTarget", 4)),
        threads_per_block=int(cfg.get("threadsPerBlock", 256)),
        regs_per_thread=int(cfg.get("regsPerThread", 128)),
    )


def _kernel(k):
    return {
        "name": k.name.decode(),
        "flopsPerGpu": k.flops_g,
        "dramPerGpu": k.dram_g,
        "smemPerGpu": k.smem_g,
        "tCompute": k.t_compute,
        "tDram": k.t_dram,
        "tSmem": k.t_smem,
        "tL2": k.t_l2,
        "tTotal": k.t_total,
        "achieved": k.achieved,
        "occupancyUtil": k.occupancy_util,
        "bound": ("compute", "dram", "smem", "l2")[k.bound] if 0 <= k.bound < 4 else "compute",
    }


def simulate_l2(model, gpu, cfg, phase):
    ms = _model(model)
    gs = _gpu(gpu)
    wc = _cfg(cfg, phase)
    res = L2Result()
    if not _lib.gotham_l2_simulate(ctypes.byref(ms), ctypes.byref(gs), ctypes.byref(wc), ctypes.byref(res)):
        raise RuntimeError("C++ L2 simulate call failed")
    return {
        "kernels": [_kernel(res.kernels[i]) for i in range(res.n_kernels)],
        "layerFlops": res.layer_flops,
        "layerDramPerGpu": res.layer_dram_g,
        "layerSmemPerGpu": res.layer_smem_g,
        "layerTime": res.layer_time,
        "totalTime": res.total_time,
        "tokens": res.tokens,
        "throughput": res.throughput,
        "latencyPerToken": res.latency_per_token,
        "achieved": res.achieved,
        "utilization": res.utilization,
        "pEff": res.p_eff,
        "tComputeSum": res.t_compute_sum,
        "tDramSum": res.t_dram_sum,
        "tSmemSum": res.t_smem_sum,
        "occupancyUtil": res.occupancy_util,
        "l2Hit": res.l2_hit,
        "memWeights": res.mem_weights,
        "memKv": res.mem_kv,
        "memAct": res.mem_act,
        "memTotal": res.mem_total,
        "ridge": res.ridge,
        "peak": res.peak,
        "bw": res.bw,
    }
