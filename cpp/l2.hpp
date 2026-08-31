#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/* L2-level model: coarse architecture plus the inner workings of a layer
   (SwiGLU width, expert count, routing, vocab). */
typedef struct {
  double params_b;
  double active_b;
  int layers;
  int hidden;
  int heads;
  int kv_heads;
  int vocab;
  int ffn;             /* SwiGLU intermediate width; 0 => round(8/3*H) */
  int experts;         /* routed experts; 1 for dense */
  int topk;            /* experts active per token */
  int shared_experts;  /* always-on experts (0 for none) */
} L2Model;

/* L2-level GPU: the coarse ceilings plus per-SM execution resources. */
typedef struct {
  double fp16_tflops;
  double fp8_tflops;
  double fp32_tflops;
  double bandwidth_gbps;
  double memory_gb;
  int sm_count;
  double clock_ghz;
  double smem_per_sm;      /* bytes */
  double regs_per_sm;      /* bytes */
  double l1_per_sm;        /* bytes */
  double l2_bytes;
  double l2_bw_gbps;       /* L2 bandwidth estimate */
  double tmem_per_sm;      /* bytes (0 => no tensor memory) */
  double smem_bw_per_clk;  /* bytes/clk/SM */
  int max_threads_per_sm;
  int max_warps_per_sm;
  int max_blocks_per_sm;
} L2Gpu;

/* phase: 0 = prefill, 1 = decode (one phase per call; Python drives "both") */
typedef struct {
  int phase;
  int B;
  int S;
  int gpus;
  int precision;      /* 0 fp32, 1 fp16, 2 fp8, 3 int8, 4 int4 */
  int kv_precision;   /* 0 fp16, 1 fp8 */
  double compute_scale;
  double bandwidth_scale;
  int flash_attention;      /* 1 = tiled FA-style (no SxS in HBM) */
  int recompute;            /* 1 = activation checkpointing (1 layer buffered) */
  int fuse_layer;           /* 1 = activations stay on-chip within a layer */
  int q_tile;               /* FA query tile rows */
  int k_tile;               /* FA key tile cols */
  double l2_usable_frac;    /* fraction of L2 treated as usable (0.8 default) */
  int occupancy_target_blocks; /* CTAs/SM needed to saturate tensor cores */
  int threads_per_block;
  int regs_per_thread;
} L2Workload;

typedef struct {
  char name[32];
  double flops_g;      /* per-GPU FLOPs */
  double dram_g;       /* per-GPU DRAM bytes (after L2 hit) */
  double smem_g;       /* per-GPU SMEM traffic */
  double t_compute;
  double t_dram;
  double t_smem;
  double t_l2;
  double t_total;
  double achieved;     /* FLOP/s for this kernel */
  double occupancy_util;
  int bound;           /* 0 compute, 1 dram, 2 smem, 3 l2 */
  int valid;
} L2Kernel;

typedef struct {
  L2Kernel kernels[24];
  int n_kernels;
  double layer_flops;
  double layer_dram_g;
  double layer_smem_g;
  double layer_time;
  double total_time;         /* prefill: whole batch; decode: one step */
  double tokens;
  double throughput;         /* tokens/s */
  double latency_per_token;
  double achieved;
  double utilization;
  double p_eff;
  double t_compute_sum;
  double t_dram_sum;
  double t_smem_sum;
  double occupancy_util;     /* flops-weighted mean across kernels */
  double l2_hit;             /* fraction of layer traffic served by L2 */
  double mem_weights;        /* per GPU */
  double mem_kv;
  double mem_act;
  double mem_total;
  double ridge;
  double peak;
  double bw;
  int valid;
} L2Result;

int gotham_l2_simulate(const L2Model* model, const L2Gpu* gpu,
                       const L2Workload* cfg, L2Result* out);
double gotham_l2_peak_flops(const L2Gpu* gpu, int precision, double scale);

#ifdef __cplusplus
}
#endif
