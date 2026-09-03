"""Measured L1 validation records.

Every entry pins an independently published measurement to one catalog model
and GPU plus the exact workload configuration the source used.  The L1 runner
(simulator/validate.py) replays each record through the roofline core and
computes signed errors:  error = (predicted - measured) / measured.

`comparable` controls whether the row is fed into the aggregate MAPE:
  "decode"              -> decode-only steady-state row (direct comparison)
  "composite_serial"    -> end-to-end run modeled as serial prefill+decode
                           waves at a published concurrency (documented
                           approximation of engine scheduling)
  "reference"           -> engine/scheduler-level result; no L1 prediction is
                           meaningful without unpublished concurrency details.

All numbers below were re-read from the linked pages on 2026-09-03.
"""

BENCHMARKS = [
    {
        "id": "vllm-fp8kv-llama31-8b-h100-bf16kv",
        "source": "vLLM FP8 KV-cache validation (official vLLM project blog)",
        "url": "https://github.com/vllm-project/vllm-project.github.io/blob/main/_posts/2026-04-22-fp8-kvcache.md",
        "published": "2026-04-22",
        "accessed": "2026-09-03",
        "model": "llama3-8b",
        "gpu": "h100",
        "precision": "fp16",
        "kvPrecision": "fp16",
        "phase": "decode",
        "B": 1,
        "S": 778,
        "gpus": 1,
        "engine": "vLLM",
        "engineVersion": "v0.19.1 + FlashAttention-3",
        "metric": "interTokenLatencyMs",
        "measured": 6.474,
        "measuredThroughputTokPerSec": 154.5,
        "fitInterceptMs": 6.44,
        "fitSlopeMsPerTok": 4.37e-05,
        "comparable": "decode",
        "notes": (
            "Single-request (concurrency 1) decode on one H100; 128 output "
            "tokens, input length swept 256-125k. ITL is the fitted line "
            "evaluated at S=778 (MLPerf CNN-DailyMail average input length): "
            "6.44 + 4.37e-5*778 ms. Model weights are BF16; KV cache is BF16."
        ),
    },
    {
        "id": "vllm-fp8kv-llama31-8b-h100-fp8kv",
        "source": "vLLM FP8 KV-cache validation (official vLLM project blog)",
        "url": "https://github.com/vllm-project/vllm-project.github.io/blob/main/_posts/2026-04-22-fp8-kvcache.md",
        "published": "2026-04-22",
        "accessed": "2026-09-03",
        "model": "llama3-8b",
        "gpu": "h100",
        "precision": "fp16",
        "kvPrecision": "fp8",
        "phase": "decode",
        "B": 1,
        "S": 778,
        "gpus": 1,
        "engine": "vLLM",
        "engineVersion": "v0.19.1 + FlashAttention-3",
        "metric": "interTokenLatencyMs",
        "measured": 6.598,
        "measuredThroughputTokPerSec": 151.6,
        "fitInterceptMs": 6.58,
        "fitSlopeMsPerTok": 2.37e-05,
        "comparable": "decode",
        "notes": (
            "Same sweep as the BF16-KV row with --kv-cache-dtype fp8. The "
            "measured ITL slope (2.37e-5 ms/token) is almost exactly the FP8 "
            "KV read traffic the L1 model predicts, validating the KV-read "
            "term; the intercept (6.58 ms) is dominated by software overheads "
            "at concurrency 1."
        ),
    },
    {
        "id": "vllm-fp8kv-llama31-8b-b200-fp8kv",
        "source": "vLLM FP8 KV-cache validation (official vLLM project blog)",
        "url": "https://github.com/vllm-project/vllm-project.github.io/blob/main/_posts/2026-04-22-fp8-kvcache.md",
        "published": "2026-04-22",
        "accessed": "2026-09-03",
        "model": "llama3-8b",
        "gpu": "b200",
        "precision": "fp16",
        "kvPrecision": "fp8",
        "phase": "decode",
        "B": 1,
        "S": 778,
        "gpus": 1,
        "engine": "vLLM",
        "engineVersion": "vLLM + FlashInfer (same post)",
        "metric": "interTokenLatencyMs",
        "measured": 3.968,
        "measuredThroughputTokPerSec": 252.0,
        "fitInterceptMs": 3.96,
        "fitSlopeMsPerTok": 9.72e-06,
        "comparable": "decode",
        "notes": (
            "Single-request decode on one B200 with the FlashInfer backend; "
            "ITL fit evaluated at S=778. The intercept is well above the "
            "pure weight-streaming bound, so the implied bandwidth "
            "efficiency is lower than on H100 for this measurement."
        ),
    },
    {
        "id": "glm53-flash-fp8-h200-tp4-single-stream",
        "source": "GLM-5.3-Flash FP8 model card (dealignai), cross-checked with SGLang docs and zai-org config.json",
        "url": "https://huggingface.co/dealignai/GLM-5.3-Flash-UNCENSORED-FP8",
        "published": "2026-08-28",
        "accessed": "2026-09-03",
        "model": "glm-5.3-flash",
        "gpu": "h200",
        "precision": "fp8",
        "kvPrecision": "fp16",
        "phase": "decode",
        "B": 1,
        "S": 128,
        "gpus": 4,
        "engine": "vLLM",
        "engineVersion": "vLLM (TP4)",
        "metric": "interTokenLatencyMs",
        "measured": 6.135,
        "measuredThroughputTokPerSec": 163.0,
        "comparable": "decode",
        "notes": (
            "Single-stream decode of GLM-5.3-Flash FP8 on 4x H200 (TP4). "
            "Model card reports 163 tok/s decode without MTP speculative "
            "decoding (211 tok/s with MTP). S is unpublished; S=128 is "
            "assumed for the tiny KV term. Architecture from config.json: "
            "320B total / 18B active, 45 text layers, 288 experts, top-8."
        ),
    },
    {
        "id": "vllm-fp8kv-llama31-8b-h100-c8-fp8kv",
        "source": "vLLM FP8 KV-cache validation (official vLLM project blog)",
        "url": "https://github.com/vllm-project/vllm-project.github.io/blob/main/_posts/2026-04-22-fp8-kvcache.md",
        "published": "2026-04-22",
        "accessed": "2026-09-03",
        "model": "llama3-8b",
        "gpu": "h100",
        "precision": "fp16",
        "kvPrecision": "fp8",
        "phase": "composite",
        "B": 8,
        "S": 20000,
        "gpus": 1,
        "N": 150,
        "O": 2000,
        "engine": "vLLM",
        "engineVersion": "v0.19.1 + FlashAttention-3",
        "metric": "totalDurationS",
        "measured": 585.2,
        "measuredOutputTokPerSec": 517.5,
        "comparable": "composite_serial",
        "notes": (
            "150 requests, ~20k input / ~2k output tokens each, concurrency "
            "8 on one H100, FP8 KV cache, BF16 weights. L1 replay treats the "
            "run as ceil(150/8)=19 serial waves of one prefill (B=8) followed "
            "by 2000 decode steps (B=8) - a documented approximation that "
            "ignores prefill/decode overlap and scheduler gaps."
        ),
    },
    {
        "id": "redhat-mlperf-v51-llama31-8b-h100-offline",
        "source": "Red Hat: MLPerf Inference v5.1 results",
        "url": "https://www.redhat.com/en/blog/efficient-and-reproducible-llm-inference-red-hat-mlperf-inference-v51-results",
        "published": "2025-10-31",
        "accessed": "2026-09-03",
        "model": "llama3-8b",
        "gpu": "h100",
        "precision": "fp8",
        "kvPrecision": "fp8",
        "phase": "composite",
        "B": None,
        "S": 778,
        "gpus": 1,
        "N": 13368,
        "O": 73,
        "engine": "vLLM",
        "engineVersion": "v0.10.0 (MLPerf Offline scenario)",
        "metric": "throughputTokPerSec",
        "measured": 5777.08,
        "comparable": "reference",
        "notes": (
            "Official MLPerf v5.1 small-LLM result: FP8 Llama-3.1-8B on one "
            "H100, CNN-DailyMail subset (avg ISL 778 / OSL 73). Offline "
            "throughput is achieved at an unpublished, scheduler-controlled "
            "concurrency, so no L1 prediction is computed; listed as a "
            "reference point for the scheduler-level L2/L3 stages."
        ),
    },
    {
        "id": "redhat-mlperf-v51-llama31-8b-h100-server",
        "source": "Red Hat: MLPerf Inference v5.1 results",
        "url": "https://www.redhat.com/en/blog/efficient-and-reproducible-llm-inference-red-hat-mlperf-inference-v51-results",
        "published": "2025-10-31",
        "accessed": "2026-09-03",
        "model": "llama3-8b",
        "gpu": "h100",
        "precision": "fp8",
        "kvPrecision": "fp8",
        "phase": "composite",
        "B": None,
        "S": 778,
        "gpus": 1,
        "N": 13368,
        "O": 73,
        "engine": "vLLM",
        "engineVersion": "v0.10.0 (MLPerf Server scenario)",
        "metric": "throughputTokPerSec",
        "measured": 5103.99,
        "comparable": "reference",
        "notes": (
            "Same submission, Server scenario (Poisson load, TTFT/TPOT "
            "bounds). Concurrency is load-dependent; reference only at L1."
        ),
    },
    {
        "id": "redhat-mlperf-v51-llama31-8b-l40s-offline",
        "source": "Red Hat: MLPerf Inference v5.1 results",
        "url": "https://www.redhat.com/en/blog/efficient-and-reproducible-llm-inference-red-hat-mlperf-inference-v51-results",
        "published": "2025-10-31",
        "accessed": "2026-09-03",
        "model": "llama3-8b",
        "gpu": "l40s",
        "precision": "fp8",
        "kvPrecision": "fp8",
        "phase": "composite",
        "B": None,
        "S": 778,
        "gpus": 1,
        "N": 13368,
        "O": 73,
        "engine": "vLLM",
        "engineVersion": "v0.10.0 (MLPerf Offline scenario)",
        "metric": "throughputTokPerSec",
        "measured": 1642.0,
        "comparable": "reference",
        "notes": "L40S Offline; reference only at L1 (concurrency unpublished).",
    },
    {
        "id": "redhat-mlperf-v51-llama31-8b-l40s-server",
        "source": "Red Hat: MLPerf Inference v5.1 results",
        "url": "https://www.redhat.com/en/blog/efficient-and-reproducible-llm-inference-red-hat-mlperf-inference-v51-results",
        "published": "2025-10-31",
        "accessed": "2026-09-03",
        "model": "llama3-8b",
        "gpu": "l40s",
        "precision": "fp8",
        "kvPrecision": "fp8",
        "phase": "composite",
        "B": None,
        "S": 778,
        "gpus": 1,
        "N": 13368,
        "O": 73,
        "engine": "vLLM",
        "engineVersion": "v0.10.0 (MLPerf Server scenario)",
        "metric": "throughputTokPerSec",
        "measured": 1207.0,
        "comparable": "reference",
        "notes": "L40S Server; reference only at L1 (concurrency unpublished).",
    },
    {
        "id": "nvidia-trtllm-llama2-70b-8xh100-single",
        "source": "NVIDIA TensorRT-LLM: batch-1 latency study (Dec 2023)",
        "url": "https://developer.nvidia.com/blog/tensorrt-llm-improves-large-language-model-latency-by-up-to-8x-on-hopper-gpus/",
        "published": "2023-12-14",
        "accessed": "2026-09-03",
        "model": "llama2-70b",
        "gpu": "h100",
        "precision": "fp8",
        "kvPrecision": "fp8",
        "phase": "composite",
        "B": 1,
        "S": 2048,
        "gpus": 8,
        "O": 128,
        "engine": "TensorRT-LLM",
        "engineVersion": "TRT-LLM (TP8)",
        "metric": "queriesPerSec",
        "measured": 0.75,
        "comparable": "reference",
        "notes": (
            "Single request, ISL 2048 / OSL 128, 8x H100, ~1.7 s per "
            "inference. Batch-1 decode latency is dominated by per-step "
            "software and pipeline overheads rather than DRAM streaming, so "
            "the row is reference-only for L1 and motivates L2/L3 latency "
            "modeling."
        ),
    },
]


def get_benchmark(bench_id):
    for b in BENCHMARKS:
        if b["id"] == bench_id:
            return b
    raise KeyError(f"unknown validation record: {bench_id}")
