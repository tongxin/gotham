# Gotham Simulation Methodology

**GPU Roofline Simulator for LLM Inference** — how the simulator works, what every
knob does, and how each knob and constraint moves the roofline, in equations.

Companion to the code: the math lives in `cpp/sim.cpp`, exposed through a C ABI,
bound by `simulator/core.py` via ctypes, and served to the UI by
`simulator/server.py`. The JavaScript renders only; it computes nothing.

---

## 1. The roofline model in one paragraph

A roofline chart has two ceilings. The **compute ceiling** is the GPU's
tensor-core peak for the selected precision, `P`. The **memory ceiling** is the
effective HBM bandwidth, `b_eff`, times the arithmetic intensity `I` (FLOPs per
byte of DRAM traffic). Any workload achieves at most the lower of the two:

```text
A = min(P, b_eff · I)          A  = achieved FLOP/s
I = FLOPs / bytes              R  = P / b_eff  (ridge point)
```

Left of the ridge (`I < R`) the workload is **memory-bound** and adding compute
peak does nothing; right of the ridge (`I > R`) it is **compute-bound** and
adding bandwidth does nothing. The whole simulator is just the machinery for
computing `I` honestly for a given model, GPU, precision, and workload shape.

## 2. Symbols

| Symbol | Meaning | Source |
|---|---|---|
| `N` | total parameters (converted from B to units) | model catalog |
| `N_active` | per-token active parameters (MoE); `= N` for dense | model catalog |
| `L` | transformer layers | model catalog |
| `H` | hidden size | model catalog |
| `h` | attention heads | model catalog |
| `h_kv` | KV heads (`= h` if unspecified) | model catalog |
| `H_kv` | KV dimension `= H · h_kv / h` | derived |
| `S` | sequence length | user knob |
| `B` | batch size (tokens per decode step; `B·S` tokens in prefill) | user knob |
| `G` | GPU count (tensor-parallel shards) | user knob |
| `prec` | weight precision | user knob |
| `kv_prec` | KV cache precision | user knob |
| `c_s` | compute-scale multiplier (derating) | user knob |
| `w_s` | bandwidth-scale multiplier (derating) | user knob |
| `b_elem` | bytes per weight element | precision table |
| `q` | quantization metadata overhead | precision table |
| `b_kv` | bytes per KV-cache element | KV precision |
| `W` | total weight bytes | derived |
| `P` | tensor-core peak for `prec` (× `c_s`) | GPU catalog + precision |
| `b_eff` | effective DRAM bandwidth `= b × w_s` | GPU catalog + knob |
| `R` | ridge point `= P / b_eff` | derived |
| `M` | per-GPU HBM capacity | GPU catalog |

## 3. Precision: bytes and compute peak

The weight precision selects both how many bytes each parameter occupies and
which compute peak applies:

| `prec` | `b_elem` (B/param) | `q` | Compute peak `P` |
|---|---|---|---|
| FP32 | 4 | 1 | scalar `P_fp32` |
| FP16 / BF16 | 2 | 1 | `P_fp16` |
| FP8 | 1 | 1 | `P_fp8` (falls back to `P_fp16` if unsupported) |
| INT8 | 1 | 1.03 | `P_fp8` or `2·P_fp16` |
| INT4 | 0.5 | 1.06 | `4·P_fp16` |

**Constraint:** FP8 is only offered when the GPU has FP8 tensor cores
(A100, RTX 3090, and T4 do not — the option is filtered out of the UI, and the
peak falls back to `P_fp16`). The `q` factors account for scale/zero-point
metadata that real quantized checkpoints carry.

The KV cache precision is separate (`b_kv = 2` for FP16, `1` for FP8) because in
practice KV caches are often kept in FP16 even for quantized weights.

```text
W = N · b_elem(prec) · q(prec)
```

## 4. Weighted FLOPs and DRAM traffic

### FLOPs

```text
F_token        = 2 · N_active                       # forward pass per token
F_attn_prefill = 2 · B · S² · L · (H + H_kv)        # QK^T + PV over all tokens
F_attn_decode  = 2 · B · S · L · (H + H_kv)         # per step, against cached KV

FLOPs_prefill = B · S · F_token + F_attn_prefill
FLOPs_decode  = B · (F_token + 2 · S · L · (H + H_kv))   # per decode step
```

The `2·N_active` term covers the QKV/MLP projections; the attention terms are the
sequence-dependent part. MoE models pay `N_active` (e.g. GLM-5.2: 39B of 743B),
while the **total** `N` still governs weight bytes.

### DRAM traffic

```text
KV_write_token = 2 · L · H_kv · b_kv     # store K and V once per token
KV_read_token  = 2 · L · S · H_kv · b_kv # read the full cache during decode
ACT_token      = 16 · H                  # rough activation traffic per token

bytes_prefill = W + B · S · (KV_write_token + ACT_token)
bytes_decode  = W + B · (KV_read_token + KV_write_token + ACT_token)                     # per step
```

**Constraint / assumption:** in decode, the full weight set `W` is read from
DRAM once per step and amortized over the batch `B`; no L2/weight-reuse credit is
taken. In prefill, `W` is read once for the whole `B·S` batch, so it is fully
amortized there. The `16·H` activation term is an acknowledged estimate
(FlashAttention-style, per token), not a cycle-accurate activation budget.

## 5. Tensor-parallel sharding (`G` GPUs)

Multi-GPU results assume **ideal tensor parallelism**: weights, KV cache, and
FLOPs are all sharded evenly, and inter-GPU communication is ignored.

```text
FLOPs_G = FLOPs / G        bytes_G = bytes / G
I_G     = FLOPs_G / bytes_G = FLOPs / bytes        # G cancels!
A_G     = min(P, b_eff · I_G)
t       = FLOPs_G / A_G                             # time per phase / step
```

Two consequences worth internalizing:

1. **The roofline point is `G`-invariant.** Because both FLOPs and bytes divide
   by `G`, the intensity and the achieved FLOP/s per GPU do not move when you
   change GPU count. What scales is aggregate throughput (`G ×`) and, critically,
   memory per GPU.
2. **Capacity is the real `G` constraint.** Weights, KV cache, and activations
   all divide by `G`, so increasing `G` is what lets a 743B model fit at all
   (see §7).

## 6. Throughput, latency, utilization

```text
A_G      = min(P, b_eff · I_G)
t        = FLOPs_G / A_G
tokens/s = tokens / t          # prefill: B·S / t;  decode: B / t
latency  = t / tokens          # per-token latency
util     = A_G / P
bound    = memory if I_G < R else compute
```

## 7. Memory footprint and the capacity constraint

```text
weights_G = W / G
KV_G      = 2 · L · H_kv · B · S · b_kv / G
ACT_G     = 16 · B · S · H / G
total_G   = weights_G + KV_G + ACT_G

feasible  ⟺  total_G ≤ M
```

If `total_G > M` the UI flags **OOM**. This is where GQA/MQA earns its keep
(`h_kv < h` shrinks `H_kv`, hence the KV cache), and where quantized weights plus
GPU count decide whether a frontier MoE (GLM-5.2, DeepSeek-V4 Pro) can run at all.
The activation estimate `16·B·S·H` is the FlashAttention-era rule of thumb, not a
checkpointing-aware exact budget.

## 8. Knob-by-knob effect on the roofline

| Knob | Equations it touches | Roofline effect |
|---|---|---|
| **Model** (`N`, `N_active`, `L`, `H`, `h_kv`) | §4, §7 | Picks the point: bigger `N` → lower `I` (more weight traffic); GQA shrinks KV traffic; MoE active params cut FLOPs |
| **GPU** (`P`, `b`, `M`) | §1, §7 | Sets both ceilings, the ridge `R`, and the capacity line |
| **Weight precision** | `W`, `P` | Smaller `b_elem` → less traffic → `I` moves right; higher tensor peak → ridge moves right |
| **KV precision** | `KV_G`, `bytes_*` | Small shift in `I` and capacity; mostly invisible on the chart |
| **Phase (prefill/decode)** | §4 | Two different points: prefill is usually compute-bound, decode usually memory-bound at small `B` |
| **Batch `B`** | `bytes` amortization, `KV_G` | Decode: `B↑` amortizes `W` → `I` rises toward the ridge. Prefill: `W` already amortized; `I` barely moves. KV cache grows linearly |
| **Sequence `S`** | attention terms, KV read | Prefill: `S²` FLOPs vs linear bytes → `I` climbs steeply. Decode: per-token FLOPs and KV read both grow linearly in `S` → `I` saturates, latency grows linearly |
| **GPU count `G`** | §5, §7 | Point stays put; aggregate throughput ×`G`; per-GPU memory ↓ (fixes OOM) |
| **Compute scale `c_s`** | `P = P_prec · c_s` | Lowers the compute ceiling; ridge moves left; compute-bound points drop |
| **Bandwidth scale `w_s`** | `b_eff = b · w_s` | Rotates the memory ceiling down; ridge moves right; memory-bound points drop |
| **Sweep toggle** | none (diagnostic) | Traces decode points for `B = 1 … 1024`, showing the memory→compute transition |

### Why decode `I` rises with batch

For decode at batch `B`, with `W` amortized:

```text
I_decode ≈ (F_token + 2·S·L·(H + H_kv)) / (W/B + 2·L·S·H_kv·b_kv + 2·L·H_kv·b_kv + 16·H)
```

At `B = 1`, `W` dominates the denominator and `I ≈ 1 FLOP/B` for a typical FP16
model — deep in the memory-bound regime, which is why single-stream decode on
H100 runs at roughly DRAM bandwidth. As `B → 1024`, `W/B → 0` and `I` approaches
the compute-side ratio; the sweep curve on the chart shows this trajectory.

## 9. Derived ceilings drawn on the chart

The primary ceiling is the selected GPU + precision. The UI can overlay:

- **Other precision ceilings** for the same GPU — `P(prec')` at the same
  `b_eff`, each with its own ridge (`fp32`, `fp8/int8`, `int4`).
- **Other GPU ceilings** — each other GPU's `P_fp16` and `b` (unscaled by the
  derating knobs), drawn as ghost lines for comparison.

All are the same `min(P, b_eff · I)` curves with different parameters; no
separate model is needed.

## 10. Worked example — LLaMA-3 8B on H100, FP16, `B=1, S=2048`

```text
N=8.03e9  N_active=8.03e9  L=32  H=4096  h=h_kv=32 → H_kv=4096
W = 8.03e9 · 2 = 16.06 GB
P = 989.5 TFLOPS   b_eff = 3.35 TB/s   R = 295.4 FLOP/B
```

**Prefill** (2048 tokens):

```text
FLOPs = 2048·2·8.03e9 + 2·2048²·32·(4096+4096) = 35.09e12
bytes = 16.06e9 + 2048·(2·32·4096·2 + 16·4096) = 17.27 GB
I     = 2032 FLOP/B  →  I > R  →  compute-bound
A     = 989.5 TFLOPS
t     = 35.09e12 / 989.5e12 = 35.5 ms
t/s   = 2048 / 35.5 ms ≈ 57.7k tok/s
```

**Decode** (one step, 1 token):

```text
FLOPs = 2·8.03e9 + 2·2048·32·8192 = 17.13e9
bytes = 16.06e9 + (2·2048·32·4096·2 + 2·32·4096·2 + 16·4096) ≈ 17.13 GB
I     = 1.0 FLOP/B  →  I < R  →  memory-bound
A     = 3.35 TFLOPS (bandwidth)
t     = 17.13e9 / 3.35e12 = 5.11 ms/step → ≈ 196 tok/s
```

**Memory:** `16.06 GB (W) + 1.07 GB (KV) + 0.13 GB (ACT) = 17.3 GB ≤ 80 GB` ✓

This is the canonical pattern: prefill sits on the compute ceiling, decode sits
on the memory ceiling, and only batch scaling moves decode off the wall.

## 11. Deliberate limitations

- **No communication model.** Tensor-parallel scaling is idealized; all-gather /
  reduce-scatter / expert-parallel traffic is ignored. See the companion
  visualization [How to Parallelize a Transformer for Training](https://ezyang.github.io/interactive-parallelize-transformer/)
  for that layer of analysis.
- **No occupancy or microarchitecture.** SM count, register file, L1/SMEM/L2,
  TMEM, and SM bandwidth do not enter the equations; `sram_MB` is display-only.
  The roofline answers "best possible", not "what a kernel achieves".
- **No L2/SMEM reuse credit.** All weight traffic is charged to DRAM, which makes
  the model conservative for high-reuse workloads.
- **Activation traffic is an estimate** (`16·H` per token, `16·B·S·H` buffer),
  not a per-layer checkpointing analysis.
- **Peak-vs-sustained.** Spec-sheet peaks are used; the compute/bandwidth derating
  sliders exist precisely to approximate sustained performance (e.g. ~70–75% MFU
  on Hopper).

## 12. Reproducing any chart value

Every number on screen is produced by the C++ core from the JSON config sent to
`POST /api/simulate`; the same computation is available offline:

```sh
python3 -m simulator.cli --model glm-5.2 --gpu b300 --phase both \
  --batch 1 --seq 4096 --precision fp8 --gpus 8
```
