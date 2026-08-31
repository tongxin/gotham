# Gotham — GPU Roofline Simulator for LLM Inference

An interactive, dependency-free web app that models LLM prefill and decode on a
roofline chart. Pick a model (or several to compare), a GPU, a precision, and a
workload — the simulator computes arithmetic intensity, achieved performance,
memory footprint, and throughput, then plots everything against the GPU's
compute and memory ceilings.

## Run it

Open `index.html` in any modern browser. No build step, no server, no
dependencies.

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

## Project layout

```text
index.html      UI shell
css/style.css   dark theme
js/data.js      model + GPU + precision catalogs
js/sim.js       roofline math (pure, node-testable)
js/chart.js     SVG chart renderers (roofline, memory, throughput)
js/app.js       state + wiring
```

`sim.js` and `data.js` are UMD modules, so the math can be exercised from Node:

```sh
node -e 'const d = require("./js/data.js"); const S = require("./js/sim.js");
const g = d.gpus.find(x => x.id === "h100");
const m = d.models.find(x => x.id === "llama3-8b");
console.log(S.simulate(m, g, {precision:"fp16", kvPrecision:"fp16", phase:"both", B:1, S:2048, gpus:1, computeScale:1, bandwidthScale:1}));'
```
