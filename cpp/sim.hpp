#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/* Model architecture. active_b = per-token active params for MoE models
   (0 means "same as params_b"). kv_heads = 0 means "same as heads". */
typedef struct {
  double params_b;
  double active_b;
  int layers;
  int hidden;
  int heads;
  int kv_heads;
} GothamModel;

/* GPU compute/memory characteristics. fp8_tflops = 0 means "not supported". */
typedef struct {
  double fp16_tflops;
  double fp8_tflops;
  double fp32_tflops;
  double bandwidth_gbps;
  double memory_gb;
} GothamGpu;

/* Workload configuration.
   phase:       0 = prefill, 1 = decode, 2 = both
   precision:   0 = fp32, 1 = fp16/bf16, 2 = fp8, 3 = int8, 4 = int4
   kv_precision: 0 = fp16 (2 B/elem), 1 = fp8 (1 B/elem) */
typedef struct {
  int phase;
  int B;
  int S;
  int gpus;
  int precision;
  int kv_precision;
  double compute_scale;
  double bandwidth_scale;
} GothamWorkload;

/* Roofline results for one phase. bound: 0 = memory, 1 = compute. */
typedef struct {
  double flops;
  double bytes;
  double tokens;
  double flops_per_gpu;
  double bytes_per_gpu;
  double intensity;
  double achieved;
  double utilization;
  double time;
  double throughput;
  double latency_per_token;
  int bound;
  int valid;
} GothamPhase;

typedef struct {
  GothamPhase prefill;
  GothamPhase decode;
  double w_bytes;
  double kv_write_per_token;
  double kv_read_per_token;
  double act_per_token;
  double kv_dim;
  double peak;
  double bw;
  double ridge;
  int valid;
} GothamResult;

double gotham_peak_flops(const GothamGpu* gpu, int precision, double scale);
int gotham_simulate(const GothamModel* model, const GothamGpu* gpu,
                    const GothamWorkload* cfg, GothamResult* out);
int gotham_memory(const GothamModel* model, const GothamWorkload* cfg,
                  double out[4]);  /* weights, kv, act, total (bytes, all GPUs) */
const char* gotham_version(void);

#ifdef __cplusplus
}
#endif
