# Gotham — GPU Roofline Simulator for LLM Inference

An interactive web app that models LLM prefill and decode on a roofline chart.
The simulation math runs in a **C++ core** (`cpp/sim.cpp`), wrapped by **Python**
through ctypes (`simulator/core.py`), and served to a dependency-free HTML/JS UI
by a local API server (`simulator/server.py`).

## Run it

```sh
make -C cpp                 # build libgotham.{dylib,so,dll} from the C++ core
python3 -m simulator.server # serves the UI + API at http://127.0.0.1:8765
```

Then open http://127.0.0.1:8765. No pip installs, no npm, no build step for the UI.

## What you can do

- **Compare models** — select multiple models (dense and MoE: LLaMA, GPT, Mistral,
  Mixtral, Qwen, DeepSeek-V3, …) and plot prefill and decode points on the same
  roofline.
- **Pick hardware** — H100/H200/B200, A100, MI300X, L40S, RTX 4090/3090, T4,
  with optional ghost ceilings for the other GPUs.
- **Vary precision** — FP32 / FP16-BF16 / FP8 / INT8 / INT4 weights, plus a
  separate KV-cache dtype; overlay all precision ceilings for the selected GPU.
- **Shape the workload** — batch size, sequence length, number of GPUs
  (idealized tensor-parallel sharding), and prefill / decode / both phases.
- **Explore the ridge** — the batch-size sweep shows how decode moves from the
  memory-bound regime toward the compute ceiling as the batch grows.
- **Check capacity** — stacked memory chart shows weights, KV cache, and
  activations vs. GPU HBM, flagging configurations that don't fit.

Hover any point on the roofline for intensity, achieved TFLOPS, utilization,
throughput, and time.

## Architecture

```text
cpp/sim.cpp          C++ roofline math, exported through a C ABI
cpp/sim.hpp          C structs for model / GPU / workload / results
cpp/Makefile         builds cpp/build/libgotham.{dylib,so,dll}
simulator/data.py    model + GPU + precision catalogs (single source of truth)
simulator/core.py    ctypes binding to the C++ core
simulator/server.py  stdlib HTTP server: static UI + /api/data + /api/simulate
simulator/cli.py     command-line access to the same core
simulator/test_core.py  sanity checks (python3 simulator/test_core.py)
js/app.js            UI state + fetch/render (no simulation math)
js/chart.js          SVG chart renderers (roofline, memory, throughput)
```

There is deliberately **no math in JavaScript** — the browser only formats and
plots numbers returned by `POST /api/simulate`. The same core is available from
the command line:

```sh
python3 -m simulator.cli --model llama3-8b --gpu h100 --phase both \
  --batch 1 --seq 2048 --precision fp16
```

## The model

An idealized roofline analysis: performance is bounded by the minimum of the
tensor-core compute peak and HBM bandwidth (`min(peak, bandwidth × I)`), and the
ridge point separates the memory-bound and compute-bound regimes.

```text
FLOPs(prefill)   = B·S·2·N_active + 2·B·S²·L·(H + H_kv)
FLOPs(decode/token) = 2·N_active + 2·L·S·(H + H_kv)
DRAM bytes(prefill) = W + B·S·(2·L·H_kv·b_kv + 16·H)
DRAM bytes(decode/step) = W + B·(2·L·S·H_kv·b_kv + 2·L·H_kv·b_kv + 16·H)
I = FLOPs / DRAM bytes
Achieved = min(compute peak, bandwidth × I)
```

`N_active` is the active parameter count (MoE models only pay for their
per-token experts), `W` is weight bytes, `H_kv` is the KV-head dimension
(GQA/MQA shrink the KV cache), and the `16·H` term is a rough activation-traffic
estimate. Memory capacity is weights + KV cache + a FlashAttention-style
activation estimate (`16·B·S·H`). Multi-GPU results assume ideal tensor-parallel
scaling: weights and KV sharded evenly, communication ignored.

This is an upper-bound model — it deliberately ignores kernel launches,
communication overhead, and software efficiency. For the next step — how data,
tensor, pipeline, and expert parallelism move bytes between chips and when
communication becomes the bottleneck — see the companion interactive
visualization: [How to Parallelize a Transformer for Training](https://ezyang.github.io/interactive-parallelize-transformer/).
