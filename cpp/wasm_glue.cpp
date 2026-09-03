/* WebAssembly boundary for the Gotham simulator.
   Structs are passed as flat double arrays to avoid ABI marshaling, plus a
   char buffer for kernel names. Layouts are documented in js/simapi.js. */

#include "sim.hpp"
#include "l2.hpp"

#include <cstring>

extern "C" {

/* L1: model[11], gpu[5], cfg[8] -> out[37]
   model: params_b active_b layers hidden heads kv_heads vocab ffn experts topk shared
   gpu:   fp16 fp8 fp32 bandwidth_gbps memory_gb
   cfg:   phase B S gpus precision kv_precision compute_scale bandwidth_scale
   out:   prefill[13] decode[13] w_bytes kv_write kv_read act kv_dim peak bw ridge valid
          decode_w_bytes decode_streamed_b
   phase out[13]: flops bytes tokens flops_g bytes_g intensity achieved utilization
                  time throughput latency bound valid */
int gotham_wasm_simulate(const double* m, const double* g, const double* c,
                         double* out) {
  if (!m || !g || !c || !out) return 0;
  GothamModel model;
  model.params_b = m[0];
  model.active_b = m[1];
  model.layers = static_cast<int>(m[2]);
  model.hidden = static_cast<int>(m[3]);
  model.heads = static_cast<int>(m[4]);
  model.kv_heads = static_cast<int>(m[5]);
  model.experts = static_cast<int>(m[8]);
  model.topk = static_cast<int>(m[9]);
  GothamGpu gpu;
  gpu.fp16_tflops = g[0];
  gpu.fp8_tflops = g[1];
  gpu.fp32_tflops = g[2];
  gpu.bandwidth_gbps = g[3];
  gpu.memory_gb = g[4];
  GothamWorkload cfg;
  cfg.phase = static_cast<int>(c[0]);
  cfg.B = static_cast<int>(c[1]);
  cfg.S = static_cast<int>(c[2]);
  cfg.gpus = static_cast<int>(c[3]);
  cfg.precision = static_cast<int>(c[4]);
  cfg.kv_precision = static_cast<int>(c[5]);
  cfg.compute_scale = c[6];
  cfg.bandwidth_scale = c[7];
  GothamResult res;
  if (!gotham_simulate(&model, &gpu, &cfg, &res)) return 0;

  const auto copy_phase = [&](const GothamPhase& ph, double* dst) {
    dst[0] = ph.flops;
    dst[1] = ph.bytes;
    dst[2] = ph.tokens;
    dst[3] = ph.flops_per_gpu;
    dst[4] = ph.bytes_per_gpu;
    dst[5] = ph.intensity;
    dst[6] = ph.achieved;
    dst[7] = ph.utilization;
    dst[8] = ph.time;
    dst[9] = ph.throughput;
    dst[10] = ph.latency_per_token;
    dst[11] = ph.bound;
    dst[12] = ph.valid;
  };
  copy_phase(res.prefill, out);
  copy_phase(res.decode, out + 13);
  out[26] = res.w_bytes;
  out[27] = res.kv_write_per_token;
  out[28] = res.kv_read_per_token;
  out[29] = res.act_per_token;
  out[30] = res.kv_dim;
  out[31] = res.peak;
  out[32] = res.bw;
  out[33] = res.ridge;
  out[34] = res.valid;
  out[35] = res.decode_w_bytes;
  out[36] = res.decode_streamed_b;
  return 1;
}

/* L1 memory: model[11], cfg[8] -> out[4] (weights, kv, act, total) */
void gotham_wasm_memory(const double* m, const double* c, double* out) {
  if (!m || !c || !out) return;
  GothamModel model;
  model.params_b = m[0];
  model.active_b = m[1];
  model.layers = static_cast<int>(m[2]);
  model.hidden = static_cast<int>(m[3]);
  model.heads = static_cast<int>(m[4]);
  model.kv_heads = static_cast<int>(m[5]);
  model.experts = static_cast<int>(m[8]);
  model.topk = static_cast<int>(m[9]);
  GothamWorkload cfg;
  cfg.phase = 0;
  cfg.B = static_cast<int>(c[1]);
  cfg.S = static_cast<int>(c[2]);
  cfg.gpus = static_cast<int>(c[3]);
  cfg.precision = static_cast<int>(c[4]);
  cfg.kv_precision = static_cast<int>(c[5]);
  cfg.compute_scale = c[6];
  cfg.bandwidth_scale = c[7];
  gotham_memory(&model, &cfg, out);
}

double gotham_wasm_peak_flops(const double* g, double precision, double scale) {
  GothamGpu gpu;
  gpu.fp16_tflops = g[0];
  gpu.fp8_tflops = g[1];
  gpu.fp32_tflops = g[2];
  gpu.bandwidth_gbps = g[3];
  gpu.memory_gb = g[4];
  return gotham_peak_flops(&gpu, static_cast<int>(precision), scale);
}

/* L2: model[11], gpu[17], cfg[17] -> out[313], names buffer 24*32 */
int gotham_wasm_l2_simulate(const double* m, const double* g, const double* c,
                            double* out, char* names, int names_cap) {
  if (!m || !g || !c || !out) return 0;
  L2Model model;
  model.params_b = m[0];
  model.active_b = m[1];
  model.layers = static_cast<int>(m[2]);
  model.hidden = static_cast<int>(m[3]);
  model.heads = static_cast<int>(m[4]);
  model.kv_heads = static_cast<int>(m[5]);
  model.vocab = static_cast<int>(m[6]);
  model.ffn = static_cast<int>(m[7]);
  model.experts = static_cast<int>(m[8]);
  model.topk = static_cast<int>(m[9]);
  model.shared_experts = static_cast<int>(m[10]);
  L2Gpu gpu;
  gpu.fp16_tflops = g[0];
  gpu.fp8_tflops = g[1];
  gpu.fp32_tflops = g[2];
  gpu.bandwidth_gbps = g[3];
  gpu.memory_gb = g[4];
  gpu.sm_count = static_cast<int>(g[5]);
  gpu.clock_ghz = g[6];
  gpu.smem_per_sm = g[7];
  gpu.regs_per_sm = g[8];
  gpu.l1_per_sm = g[9];
  gpu.l2_bytes = g[10];
  gpu.l2_bw_gbps = g[11];
  gpu.tmem_per_sm = g[12];
  gpu.smem_bw_per_clk = g[13];
  gpu.max_threads_per_sm = static_cast<int>(g[14]);
  gpu.max_warps_per_sm = static_cast<int>(g[15]);
  gpu.max_blocks_per_sm = static_cast<int>(g[16]);
  L2Workload cfg;
  cfg.phase = static_cast<int>(c[0]);
  cfg.B = static_cast<int>(c[1]);
  cfg.S = static_cast<int>(c[2]);
  cfg.gpus = static_cast<int>(c[3]);
  cfg.precision = static_cast<int>(c[4]);
  cfg.kv_precision = static_cast<int>(c[5]);
  cfg.compute_scale = c[6];
  cfg.bandwidth_scale = c[7];
  cfg.flash_attention = static_cast<int>(c[8]);
  cfg.recompute = static_cast<int>(c[9]);
  cfg.fuse_layer = static_cast<int>(c[10]);
  cfg.q_tile = static_cast<int>(c[11]);
  cfg.k_tile = static_cast<int>(c[12]);
  cfg.l2_usable_frac = c[13];
  cfg.occupancy_target_blocks = static_cast<int>(c[14]);
  cfg.threads_per_block = static_cast<int>(c[15]);
  cfg.regs_per_thread = static_cast<int>(c[16]);
  L2Result res;
  if (!gotham_l2_simulate(&model, &gpu, &cfg, &res)) return 0;

  for (int i = 0; i < 24; ++i) {
    double* dst = out + i * 12;
    if (i < res.n_kernels) {
      const L2Kernel& k = res.kernels[i];
      dst[0] = k.flops_g;
      dst[1] = k.dram_g;
      dst[2] = k.smem_g;
      dst[3] = k.t_compute;
      dst[4] = k.t_dram;
      dst[5] = k.t_smem;
      dst[6] = k.t_l2;
      dst[7] = k.t_total;
      dst[8] = k.achieved;
      dst[9] = k.occupancy_util;
      dst[10] = k.bound;
      dst[11] = k.valid;
    } else {
      for (int j = 0; j < 12; ++j) dst[j] = 0.0;
    }
  }
  out[288] = res.n_kernels;
  out[289] = res.layer_flops;
  out[290] = res.layer_dram_g;
  out[291] = res.layer_smem_g;
  out[292] = res.layer_time;
  out[293] = res.total_time;
  out[294] = res.tokens;
  out[295] = res.throughput;
  out[296] = res.latency_per_token;
  out[297] = res.achieved;
  out[298] = res.utilization;
  out[299] = res.p_eff;
  out[300] = res.t_compute_sum;
  out[301] = res.t_dram_sum;
  out[302] = res.t_smem_sum;
  out[303] = res.occupancy_util;
  out[304] = res.l2_hit;
  out[305] = res.mem_weights;
  out[306] = res.mem_kv;
  out[307] = res.mem_act;
  out[308] = res.mem_total;
  out[309] = res.ridge;
  out[310] = res.peak;
  out[311] = res.bw;
  out[312] = res.valid;

  if (names && names_cap >= 24 * 32) {
    char* p = names;
    for (int i = 0; i < 24; ++i) {
      if (i < res.n_kernels) {
        std::strncpy(p, res.kernels[i].name, 31);
        p[31] = '\0';
      } else {
        p[0] = '\0';
      }
      p += 32;
    }
  }
  return 1;
}

const char* gotham_wasm_version(void) {
  return gotham_version();
}

}  // extern "C"
