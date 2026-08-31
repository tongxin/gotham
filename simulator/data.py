"""Model, GPU, and precision catalogs (single source of truth for the UI too)."""

MODELS = [
    {"id": "gpt2-124m", "name": "GPT-2 124M", "params": 0.124, "layers": 12, "hidden": 768, "heads": 12, "vocab": 50257, "note": "Dense baseline"},
    {"id": "gpt2-1p5b", "name": "GPT-2 1.5B", "params": 1.542, "layers": 48, "hidden": 1600, "heads": 25, "vocab": 50257},
    {"id": "llama2-7b", "name": "LLaMA-2 7B", "params": 6.7, "layers": 32, "hidden": 4096, "heads": 32, "vocab": 32000},
    {"id": "mistral-7b", "name": "Mistral 7B", "params": 7.3, "layers": 32, "hidden": 4096, "heads": 32, "kv_heads": 8, "vocab": 32000, "note": "GQA"},
    {"id": "llama3-8b", "name": "LLaMA-3 8B", "params": 8.03, "layers": 32, "hidden": 4096, "heads": 32, "vocab": 128256},
    {"id": "gemma-2b", "name": "Gemma 2B", "params": 2.6, "layers": 26, "hidden": 2304, "heads": 8, "kv_heads": 1, "vocab": 256128, "note": "MQA"},
    {"id": "phi3-mini", "name": "Phi-3 mini 3.8B", "params": 3.8, "layers": 32, "hidden": 3072, "heads": 32, "vocab": 32064},
    {"id": "qwen2p5-14b", "name": "Qwen2.5 14B", "params": 14.8, "layers": 48, "hidden": 5120, "heads": 40, "kv_heads": 8, "vocab": 151936},
    {"id": "mixtral-8x7b", "name": "Mixtral 8x7B", "params": 46.7, "active": 12.9, "layers": 32, "hidden": 4096, "heads": 32, "kv_heads": 8, "vocab": 32000, "moe": True, "note": "8 experts, top-2"},
    {"id": "llama2-70b", "name": "LLaMA-2 70B", "params": 68.98, "layers": 80, "hidden": 8192, "heads": 64, "kv_heads": 8, "vocab": 32000},
    {"id": "llama3-70b", "name": "LLaMA-3 70B", "params": 70.6, "layers": 80, "hidden": 8192, "heads": 64, "kv_heads": 8, "vocab": 128256},
    {"id": "gpt3-175b", "name": "GPT-3 175B", "params": 175, "layers": 96, "hidden": 12288, "heads": 96, "vocab": 50257},
    {"id": "deepseek-v3", "name": "DeepSeek-V3 671B", "params": 671, "active": 37, "layers": 61, "hidden": 7168, "heads": 128, "vocab": 129280, "moe": True, "note": "MLA, 256 experts, top-8"},
    {"id": "glm-5.2", "name": "GLM-5.2 743B", "params": 743, "active": 39, "layers": 78, "hidden": 6144, "heads": 64, "kv_heads": 8, "vocab": 151936, "moe": True, "note": "DSA, 256 experts, 1M ctx"},
    {"id": "glm-5.3", "name": "GLM-5.3 744B", "params": 744, "active": 40, "layers": 78, "hidden": 6144, "heads": 64, "kv_heads": 8, "vocab": 151936, "moe": True, "note": "Same base as 5.2, ~40B active"},
    {"id": "deepseek-v4-flash", "name": "DeepSeek-V4 Flash", "params": 284, "active": 13, "layers": 43, "hidden": 4096, "heads": 64, "kv_heads": 1, "vocab": 129280, "moe": True, "note": "DSA, 256 experts, top-6, 1M ctx"},
    {"id": "deepseek-v4-pro", "name": "DeepSeek-V4 Pro", "params": 1600, "active": 49, "layers": 61, "hidden": 7168, "heads": 128, "kv_heads": 1, "vocab": 129280, "moe": True, "note": "DSA, 384 experts, top-6, 1M ctx"},
    {"id": "qwen3-30b-a3b", "name": "Qwen3 30B-A3B", "params": 30.5, "active": 3, "layers": 48, "hidden": 2048, "heads": 32, "kv_heads": 8, "vocab": 151936, "moe": True, "note": "64 experts, top-8"},
]

GPUS = [
    {"id": "h100", "name": "NVIDIA H100 SXM", "memory_GB": 80, "bandwidth_GBps": 3350, "fp16_TFLOPS": 989.5, "fp8_TFLOPS": 1979, "fp32_TFLOPS": 67, "sram_MB": 50, "note": "Hopper, HBM3"},
    {"id": "h200", "name": "NVIDIA H200", "memory_GB": 141, "bandwidth_GBps": 4800, "fp16_TFLOPS": 989.5, "fp8_TFLOPS": 1979, "fp32_TFLOPS": 67, "sram_MB": 50, "note": "Hopper, HBM3e"},
    {"id": "b200", "name": "NVIDIA B200", "memory_GB": 192, "bandwidth_GBps": 8000, "fp16_TFLOPS": 2250, "fp8_TFLOPS": 4500, "fp32_TFLOPS": 80, "sram_MB": 126, "note": "Blackwell, dual-die"},
    {"id": "b300", "name": "NVIDIA B300", "memory_GB": 288, "bandwidth_GBps": 8000, "fp16_TFLOPS": 2800, "fp8_TFLOPS": 5600, "fp32_TFLOPS": 100, "sram_MB": 126, "note": "Blackwell Ultra, HBM3e"},
    {"id": "a100-80", "name": "NVIDIA A100 80GB", "memory_GB": 80, "bandwidth_GBps": 2039, "fp16_TFLOPS": 312, "fp8_TFLOPS": None, "fp32_TFLOPS": 19.5, "sram_MB": 40, "note": "Ampere, HBM2e"},
    {"id": "a100-40", "name": "NVIDIA A100 40GB", "memory_GB": 40, "bandwidth_GBps": 1555, "fp16_TFLOPS": 312, "fp8_TFLOPS": None, "fp32_TFLOPS": 19.5, "sram_MB": 40, "note": "Ampere, HBM2"},
    {"id": "mi300x", "name": "AMD MI300X", "memory_GB": 192, "bandwidth_GBps": 5300, "fp16_TFLOPS": 1307, "fp8_TFLOPS": 2614, "fp32_TFLOPS": 81.7, "sram_MB": 256, "note": "CDNA3, HBM3"},
    {"id": "l40s", "name": "NVIDIA L40S", "memory_GB": 48, "bandwidth_GBps": 864, "fp16_TFLOPS": 362, "fp8_TFLOPS": 724, "fp32_TFLOPS": 91.6, "sram_MB": 96, "note": "Ada, GDDR6"},
    {"id": "rtx4090", "name": "NVIDIA RTX 4090", "memory_GB": 24, "bandwidth_GBps": 1008, "fp16_TFLOPS": 165, "fp8_TFLOPS": 330, "fp32_TFLOPS": 82.6, "sram_MB": 72, "note": "Ada, GDDR6X"},
    {"id": "rtx3090", "name": "NVIDIA RTX 3090", "memory_GB": 24, "bandwidth_GBps": 936, "fp16_TFLOPS": 71, "fp8_TFLOPS": None, "fp32_TFLOPS": 35.6, "sram_MB": 6, "note": "Ampere, GDDR6X"},
    {"id": "t4", "name": "NVIDIA T4", "memory_GB": 16, "bandwidth_GBps": 320, "fp16_TFLOPS": 65, "fp8_TFLOPS": None, "fp32_TFLOPS": 8.1, "sram_MB": 4, "note": "Turing, GDDR6"},
]

PRECISIONS = [
    {"id": "fp16", "label": "FP16 / BF16", "bytes": 2},
    {"id": "fp8", "label": "FP8 (E4M3 / E5M2)", "bytes": 1, "needsFp8": True},
    {"id": "int8", "label": "INT8 (W8A8)", "bytes": 1},
    {"id": "int4", "label": "INT4 (W4A16)", "bytes": 0.5},
    {"id": "fp32", "label": "FP32", "bytes": 4},
]

KV_PRECISIONS = [
    {"id": "fp16", "label": "FP16 / BF16 (2 B/elem)"},
    {"id": "fp8", "label": "FP8 (1 B/elem)"},
]


def get_model(model_id):
    for m in MODELS:
        if m["id"] == model_id:
            return m
    raise KeyError(f"unknown model: {model_id}")


def get_gpu(gpu_id):
    for g in GPUS:
        if g["id"] == gpu_id:
            return g
    raise KeyError(f"unknown GPU: {gpu_id}")


def catalog():
    return {"models": MODELS, "gpus": GPUS, "precisions": PRECISIONS, "kvPrecisions": KV_PRECISIONS}
