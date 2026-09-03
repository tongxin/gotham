"""ctypes binding to the C++ roofline core (cpp/sim.cpp)."""

import ctypes
import os


class ModelSpec(ctypes.Structure):
    _fields_ = [
        ("params_b", ctypes.c_double),
        ("active_b", ctypes.c_double),
        ("layers", ctypes.c_int),
        ("hidden", ctypes.c_int),
        ("heads", ctypes.c_int),
        ("kv_heads", ctypes.c_int),
        ("experts", ctypes.c_int),
        ("topk", ctypes.c_int),
    ]


class GpuSpec(ctypes.Structure):
    _fields_ = [
        ("fp16_tflops", ctypes.c_double),
        ("fp8_tflops", ctypes.c_double),
        ("fp32_tflops", ctypes.c_double),
        ("bandwidth_gbps", ctypes.c_double),
        ("memory_gb", ctypes.c_double),
    ]


class WorkloadCfg(ctypes.Structure):
    _fields_ = [
        ("phase", ctypes.c_int),
        ("B", ctypes.c_int),
        ("S", ctypes.c_int),
        ("gpus", ctypes.c_int),
        ("precision", ctypes.c_int),
        ("kv_precision", ctypes.c_int),
        ("compute_scale", ctypes.c_double),
        ("bandwidth_scale", ctypes.c_double),
    ]


class PhaseResult(ctypes.Structure):
    _fields_ = [
        ("flops", ctypes.c_double),
        ("bytes", ctypes.c_double),
        ("tokens", ctypes.c_double),
        ("flops_per_gpu", ctypes.c_double),
        ("bytes_per_gpu", ctypes.c_double),
        ("intensity", ctypes.c_double),
        ("achieved", ctypes.c_double),
        ("utilization", ctypes.c_double),
        ("time", ctypes.c_double),
        ("throughput", ctypes.c_double),
        ("latency_per_token", ctypes.c_double),
        ("bound", ctypes.c_int),
        ("valid", ctypes.c_int),
    ]


class SimResult(ctypes.Structure):
    _fields_ = [
        ("prefill", PhaseResult),
        ("decode", PhaseResult),
        ("w_bytes", ctypes.c_double),
        ("decode_w_bytes", ctypes.c_double),
        ("decode_streamed_b", ctypes.c_double),
        ("kv_write_per_token", ctypes.c_double),
        ("kv_read_per_token", ctypes.c_double),
        ("act_per_token", ctypes.c_double),
        ("kv_dim", ctypes.c_double),
        ("peak", ctypes.c_double),
        ("bw", ctypes.c_double),
        ("ridge", ctypes.c_double),
        ("valid", ctypes.c_int),
    ]


PRECISION_ENUM = {"fp32": 0, "fp16": 1, "fp8": 2, "int8": 3, "int4": 4}
KV_ENUM = {"fp16": 0, "fp8": 1}
PHASE_ENUM = {"prefill": 0, "decode": 1, "both": 2}


def _candidate_paths():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    names = ["libgotham.dylib", "libgotham.so", "libgotham.dll"]
    paths = []
    env = os.environ.get("GOTHAM_LIB")
    if env:
        paths.append(env)
    for name in names:
        paths.append(os.path.join(root, "cpp", "build", name))
        paths.append(os.path.join(root, "cpp", name))
    return paths


def _load_library():
    for path in _candidate_paths():
        if os.path.exists(path):
            try:
                return ctypes.CDLL(path)
            except OSError:
                continue
    raise RuntimeError(
        "C++ core library not found. Build it first with: make -C cpp"
    )


_lib = _load_library()
_lib.gotham_version.restype = ctypes.c_char_p
_lib.gotham_peak_flops.argtypes = [ctypes.POINTER(GpuSpec), ctypes.c_int, ctypes.c_double]
_lib.gotham_peak_flops.restype = ctypes.c_double
_lib.gotham_simulate.argtypes = [
    ctypes.POINTER(ModelSpec),
    ctypes.POINTER(GpuSpec),
    ctypes.POINTER(WorkloadCfg),
    ctypes.POINTER(SimResult),
]
_lib.gotham_simulate.restype = ctypes.c_int
_lib.gotham_memory.argtypes = [
    ctypes.POINTER(ModelSpec),
    ctypes.POINTER(WorkloadCfg),
    ctypes.POINTER(ctypes.c_double),
]
_lib.gotham_memory.restype = ctypes.c_int


def _model_spec(model):
    return ModelSpec(
        params_b=float(model["params"]),
        active_b=float(model.get("active") or 0.0),
        layers=int(model["layers"]),
        hidden=int(model["hidden"]),
        heads=int(model["heads"]),
        kv_heads=int(model.get("kv_heads") or 0),
        experts=int(model.get("experts") or 0),
        topk=int(model.get("topk") or 0),
    )


def _gpu_spec(gpu):
    return GpuSpec(
        fp16_tflops=float(gpu["fp16_TFLOPS"]),
        fp8_tflops=float(gpu.get("fp8_TFLOPS") or 0.0),
        fp32_tflops=float(gpu["fp32_TFLOPS"]),
        bandwidth_gbps=float(gpu["bandwidth_GBps"]),
        memory_gb=float(gpu["memory_GB"]),
    )


def _workload_cfg(cfg):
    return WorkloadCfg(
        phase=PHASE_ENUM[cfg["phase"]],
        B=int(cfg.get("B", 1)),
        S=int(cfg.get("S", 2048)),
        gpus=int(cfg.get("gpus", 1)),
        precision=PRECISION_ENUM[cfg["precision"]],
        kv_precision=KV_ENUM[cfg.get("kvPrecision", "fp16")],
        compute_scale=float(cfg.get("computeScale", 1.0)),
        bandwidth_scale=float(cfg.get("bandwidthScale", 1.0)),
    )


def _phase_dict(ph):
    if not ph.valid:
        return None
    return {
        "flops": ph.flops,
        "bytes": ph.bytes,
        "tokens": ph.tokens,
        "flopsPerGpu": ph.flops_per_gpu,
        "bytesPerGpu": ph.bytes_per_gpu,
        "intensity": ph.intensity,
        "achieved": ph.achieved,
        "utilization": ph.utilization,
        "time": ph.time,
        "throughput": ph.throughput,
        "latencyPerToken": ph.latency_per_token,
        "bound": "memory" if ph.bound == 0 else "compute",
    }


def peak_flops(gpu, precision, scale=1.0):
    return _lib.gotham_peak_flops(ctypes.byref(_gpu_spec(gpu)), PRECISION_ENUM[precision], float(scale))


def simulate(model, gpu, cfg):
    ms = _model_spec(model)
    gs = _gpu_spec(gpu)
    wc = _workload_cfg(cfg)
    res = SimResult()
    if not _lib.gotham_simulate(ctypes.byref(ms), ctypes.byref(gs), ctypes.byref(wc), ctypes.byref(res)):
        raise RuntimeError("C++ simulate call failed")
    return {
        "prefill": _phase_dict(res.prefill),
        "decode": _phase_dict(res.decode),
        "wBytes": res.w_bytes,
        "decodeWBytes": res.decode_w_bytes,
        "decodeStreamedB": res.decode_streamed_b,
        "kvWritePerToken": res.kv_write_per_token,
        "kvReadPerToken": res.kv_read_per_token,
        "actPerToken": res.act_per_token,
        "kvDim": res.kv_dim,
        "peak": res.peak,
        "bw": res.bw,
        "ridge": res.ridge,
    }


def memory_footprint(model, cfg):
    ms = _model_spec(model)
    wc = _workload_cfg(cfg)
    buf = (ctypes.c_double * 4)()
    if not _lib.gotham_memory(ctypes.byref(ms), ctypes.byref(wc), buf):
        raise RuntimeError("C++ memory call failed")
    weights, kv, act, total = buf
    return {"weights": weights, "kv": kv, "act": act, "total": total}


def decode_sweep(model, gpu, cfg, steps=44):
    points = []
    for i in range(steps):
        b = max(1, round(1024 ** (i / (steps - 1))))
        c = dict(cfg)
        c["phase"] = "decode"
        c["B"] = b
        res = simulate(model, gpu, c)
        points.append({"B": b, "x": res["decode"]["intensity"], "y": res["decode"]["achieved"]})
    return points


def version():
    return _lib.gotham_version().decode()
