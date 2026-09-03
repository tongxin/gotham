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
    {
        "id": "sglang-glm52-h200-c64",
        "source": "SGLang GLM-5.2 deployment cookbook, benchmark cards",
        "url": "https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.2",
        "published": "2026-09-03",
        "accessed": "2026-09-03",
        "model": "glm-5.2",
        "gpu": "h200",
        "precision": "fp8",
        "kvPrecision": "fp16",
        "phase": "decode",
        "B": 64,
        "S": 8192,
        "gpus": 8,
        "engine": "SGLang",
        "engineVersion": "v0.5.14 @ 49e384ce (balanced strategy)",
        "metric": "interTokenLatencyMs",
        "measured": 23.49,
        "measuredThroughputTokPerSec": None,
        "comparable": "decode",
        "notes": (
            "GLM-5.2 FP8 on one 8xH200 node, random dataset ISL 8192 / OSL 1024, "
            "max_concurrency 64 (TTFT 7473 ms). Cookbook card reports TPOT 23.49 ms "
            "and total throughput 2343 tok/s/GPU; decode-only comparison uses TPOT. "
            "The node's TP/DP split is not printed on the benchmark card; TPOT is only "
            "weakly G-dependent at L1 for MoE decode."
        ),
    },
    {
        "id": "sglang-glm52-h200-c256",
        "source": "SGLang GLM-5.2 deployment cookbook, benchmark cards",
        "url": "https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.2",
        "published": "2026-09-03",
        "accessed": "2026-09-03",
        "model": "glm-5.2",
        "gpu": "h200",
        "precision": "fp8",
        "kvPrecision": "fp16",
        "phase": "decode",
        "B": 256,
        "S": 8192,
        "gpus": 8,
        "engine": "SGLang",
        "engineVersion": "v0.5.14 @ 49e384ce (balanced strategy)",
        "metric": "interTokenLatencyMs",
        "measured": 28.08,
        "comparable": "decode",
        "notes": (
            "Same GLM-5.2/H200 card at max_concurrency 256 (TTFT 80562 ms), "
            "TPOT 28.08 ms, 2391 tok/s/GPU. Saturating MoE expert traffic makes "
            "decode per-step cost grow toward a full weight pass at this batch."
        ),
    },
    {
        "id": "sglang-glm52-b200-c64",
        "source": "SGLang GLM-5.2 deployment cookbook, benchmark cards",
        "url": "https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.2",
        "published": "2026-09-03",
        "accessed": "2026-09-03",
        "model": "glm-5.2",
        "gpu": "b200",
        "precision": "fp8",
        "kvPrecision": "fp8",
        "phase": "decode",
        "B": 64,
        "S": 8192,
        "gpus": 8,
        "engine": "SGLang",
        "engineVersion": "main @ 09ca4fc (balanced strategy)",
        "metric": "interTokenLatencyMs",
        "measured": 17.65,
        "comparable": "decode",
        "notes": (
            "GLM-5.2 FP8 on 8xB200, ISL 8192 / OSL 1024, max_concurrency 64 "
            "(TTFT 5742 ms), TPOT 17.65 ms, 3078 tok/s/GPU. Blackwell auto-selects "
            "an FP8 DSA KV cache."
        ),
    },
    {
        "id": "sglang-glm52-b200-c256",
        "source": "SGLang GLM-5.2 deployment cookbook, benchmark cards",
        "url": "https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.2",
        "published": "2026-09-03",
        "accessed": "2026-09-03",
        "model": "glm-5.2",
        "gpu": "b200",
        "precision": "fp8",
        "kvPrecision": "fp8",
        "phase": "decode",
        "B": 256,
        "S": 8192,
        "gpus": 8,
        "engine": "SGLang",
        "engineVersion": "main @ 09ca4fc (balanced strategy)",
        "metric": "interTokenLatencyMs",
        "measured": 32.61,
        "comparable": "decode",
        "notes": (
            "Same GLM-5.2/B200 card at max_concurrency 256 (TTFT 18744 ms), "
            "TPOT 32.61 ms, 5022 tok/s/GPU."
        ),
    },
    {
        "id": "sglang-glm53-b300-bf16-c64",
        "source": "SGLang GLM-5.3 deployment cookbook, benchmark cards",
        "url": "https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.3",
        "published": "2026-09-03",
        "accessed": "2026-09-03",
        "model": "glm-5.3",
        "gpu": "b300",
        "precision": "fp16",
        "kvPrecision": "fp8",
        "phase": "decode",
        "B": 64,
        "S": 8192,
        "gpus": 8,
        "engine": "SGLang",
        "engineVersion": "main @ 20a491d1d311 (balanced strategy)",
        "metric": "interTokenLatencyMs",
        "measured": 22.30,
        "comparable": "decode",
        "notes": (
            "GLM-5.3 BF16 weights on B300, ISL 8192 / OSL 1024, max_concurrency 64 "
            "(TTFT 9512 ms), TPOT 22.30 ms, 2279 tok/s/GPU. Balanced card has no "
            "EAGLE/MTP speculative decoding; Blackwell DSA KV cache is FP8."
        ),
    },
    {
        "id": "sglang-glm53-b300-bf16-c256",
        "source": "SGLang GLM-5.3 deployment cookbook, benchmark cards",
        "url": "https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.3",
        "published": "2026-09-03",
        "accessed": "2026-09-03",
        "model": "glm-5.3",
        "gpu": "b300",
        "precision": "fp16",
        "kvPrecision": "fp8",
        "phase": "decode",
        "B": 256,
        "S": 8192,
        "gpus": 8,
        "engine": "SGLang",
        "engineVersion": "main @ 20a491d1d311 (balanced strategy)",
        "metric": "interTokenLatencyMs",
        "measured": 25.84,
        "comparable": "decode",
        "notes": (
            "Same GLM-5.3/B300 BF16 card at max_concurrency 256 (TTFT 75250 ms), "
            "TPOT 25.84 ms, 2333 tok/s/GPU."
        ),
    },
    {
        "id": "trtllm-dsv32exp-b200-tp4-b1",
        "source": "NVIDIA TensorRT-LLM blog: Optimizing DeepSeek-V3.2 on NVIDIA Blackwell GPUs",
        "url": "https://nvidia.github.io/TensorRT-LLM/1.2.1/blogs/tech_blog/blog15_Optimizing_DeepSeek_V32_on_NVIDIA_Blackwell_GPUs.html",
        "published": "2026-08-01",
        "accessed": "2026-09-03",
        "model": "deepseek-v3.2-exp",
        "gpu": "b200",
        "precision": "fp8",
        "kvPrecision": "fp8",
        "phase": "decode",
        "B": 1,
        "S": 8192,
        "gpus": 4,
        "engine": "TensorRT-LLM (PyTorch backend)",
        "engineVersion": "blog benchmark: batch 1, ISL 8K, OSL 1K, 10 requests, TP4, MTP3",
        "metric": "interTokenLatencyMs",
        "measured": 3.2344,
        "measuredThroughputTokPerSec": 309.2,
        "comparable": "decode",
        "notes": (
            "DeepSeek-V3.2-Exp FP8 min-latency run on B200 (TP4, concurrency 1, "
            "ISL 8192): TPOT 3.2344 ms with 3-layer MTP speculative decoding. "
            "L1 predicts the non-speculative weight-streaming floor, so part of the "
            "gap is draft verification, not just engine overhead."
        ),
    },
    {
        "id": "sglang-dsv3-tp16-16xh800",
        "source": "SGLang issue #3812: Expert Parallelism benchmarks for DeepSeek-V3/R1",
        "url": "https://github.com/sgl-project/sglang/issues/3812",
        "published": "2025-02-23",
        "accessed": "2026-09-03",
        "model": "deepseek-v3",
        "gpu": "h800",
        "precision": "fp8",
        "kvPrecision": "fp8",
        "phase": "composite",
        "B": None,
        "S": 93,
        "gpus": 16,
        "N": 600,
        "O": 201,
        "engine": "SGLang",
        "engineVersion": "TP16 on 2x8 H800 (request rate 40)",
        "metric": "totalTokenThroughputTokPerSec",
        "measured": 1606.80,
        "measuredOutputTokPerSec": 1099.36,
        "comparable": "reference",
        "notes": (
            "DeepSeek-V3 671B FP8, 600 prompts (avg ISL 93 / OSL 201), avg "
            "concurrency 390; TTFT mean 12 s shows scheduler queueing dominates, "
            "so L1 does not compute a prediction (listed for L2/L3 scheduler work)."
        ),
    },
    {
        "id": "sglang-dsv3-tp16-ep16-16xh800",
        "source": "SGLang issue #3812: Expert Parallelism benchmarks for DeepSeek-V3/R1",
        "url": "https://github.com/sgl-project/sglang/issues/3812",
        "published": "2025-02-23",
        "accessed": "2026-09-03",
        "model": "deepseek-v3",
        "gpu": "h800",
        "precision": "fp8",
        "kvPrecision": "fp8",
        "phase": "composite",
        "B": None,
        "S": 93,
        "gpus": 16,
        "N": 600,
        "O": 201,
        "engine": "SGLang",
        "engineVersion": "TP16 + EP16 on 2x8 H800 (request rate 40)",
        "metric": "totalTokenThroughputTokPerSec",
        "measured": 1687.81,
        "measuredOutputTokPerSec": 1154.79,
        "comparable": "reference",
        "notes": (
            "Same workload with --enable-ep-moe (avg concurrency 371); output "
            "throughput 1154.79 tok/s. Reference-only at L1: queueing-dominated."
        ),
    },
    {
        "id": "redhat-vllm-qwen3-30b-a3b-h100",
        "source": "vLLM blog / Red Hat: TurboQuant study (decode measurement quoted in the TurboQuant FAQ)",
        "url": "https://blog.vllm.ai/2026/05/11/turboquant.html",
        "published": "2026-05-11",
        "accessed": "2026-09-03",
        "model": "qwen3-30b-a3b",
        "gpu": "h100",
        "precision": "fp16",
        "kvPrecision": "fp8",
        "phase": "decode",
        "B": None,
        "S": 1024,
        "gpus": 1,
        "engine": "vLLM",
        "engineVersion": "v0.20.2 (Red Hat measurement, May 2026)",
        "metric": "decodeTokPerSec",
        "measured": 4510.0,
        "comparable": "reference",
        "notes": (
            "Qwen3-30B-A3B on H100: 4510 decode tok/s with FP8 KV vs 4520 BF16 KV "
            "(Red Hat table in the TurboQuant FAQ). The concurrency / batch of the "
            "measurement is not published, so the row is reference-only; it is the "
            "source quoted for Qwen3-30B-A3B coverage."
        ),
    },
]


def get_benchmark(bench_id):
    for b in BENCHMARKS:
        if b["id"] == bench_id:
            return b
    raise KeyError(f"unknown validation record: {bench_id}")
