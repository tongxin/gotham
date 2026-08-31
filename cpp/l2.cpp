/* L2 simulation: per-kernel execution model for one transformer layer.
   Three resource clocks per kernel — tensor compute (occupancy-limited),
   HBM bandwidth (L2-reuse-adjusted), and SMEM bandwidth — plus model
   inner workings (attention, SwiGLU/MoE, vocab logits). */

#include "l2.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace {

double bytes_per_elem(int precision) {
  switch (precision) {
    case 0: return 4.0;
    case 1: return 2.0;
    case 2: return 1.0;
    case 3: return 1.0;
    case 4: return 0.5;
    default: return 2.0;
  }
}

double quant_overhead(int precision) {
  if (precision == 4) return 1.06;
  if (precision == 3) return 1.03;
  return 1.0;
}

double kv_bytes(int kv_precision) {
  return kv_precision == 1 ? 1.0 : 2.0;
}

double kv_dim(const L2Model* m) {
  const int heads = m->heads > 0 ? m->heads : 1;
  const int kv_heads = m->kv_heads > 0 ? m->kv_heads : heads;
  return static_cast<double>(m->hidden) * kv_heads / heads;
}

double active_params(const L2Model* m) {
  const double b = m->active_b > 0 ? m->active_b : m->params_b;
  return b * 1e9;
}

}  // namespace

double gotham_l2_peak_flops(const L2Gpu* g, int precision, double scale) {
  const double s = scale > 0 ? scale : 1.0;
  double p;
  switch (precision) {
    case 0: p = g->fp32_tflops; break;
    case 2: p = g->fp8_tflops > 0 ? g->fp8_tflops : g->fp16_tflops; break;
    case 3: p = g->fp8_tflops > 0 ? g->fp8_tflops : g->fp16_tflops * 2.0; break;
    case 4: p = g->fp16_tflops * 4.0; break;
    default: p = g->fp16_tflops; break;
  }
  return p * 1e12 * s;
}

int gotham_l2_simulate(const L2Model* m, const L2Gpu* g, const L2Workload* w,
                       L2Result* out) {
  if (!m || !g || !w || !out) return 0;
  *out = L2Result{};

  const int B = w->B > 0 ? w->B : 1;
  const int S = w->S > 0 ? w->S : 1;
  const int G = w->gpus > 0 ? w->gpus : 1;
  const double cscale = w->compute_scale > 0 ? w->compute_scale : 1.0;
  const double bscale = w->bandwidth_scale > 0 ? w->bandwidth_scale : 1.0;
  const int flash = w->flash_attention;
  const int fuse = w->fuse_layer;
  const double l2_usable = g->l2_bytes * (w->l2_usable_frac > 0 ? w->l2_usable_frac : 0.8);
  const int tpb = w->threads_per_block > 0 ? w->threads_per_block : 512;
  const int rpt = w->regs_per_thread > 0 ? w->regs_per_thread : 64;
  const int occ_target = w->occupancy_target_blocks > 0 ? w->occupancy_target_blocks : 4;
  const int qt = w->q_tile > 0 ? w->q_tile : 64;
  const int kt = w->k_tile > 0 ? w->k_tile : 64;

  const double b_elem = bytes_per_elem(w->precision);
  const double qov = quant_overhead(w->precision);
  const double kvb = kv_bytes(w->kv_precision);
  const double H = m->hidden;
  const double Hkv = kv_dim(m);
  const double L = m->layers;
  const double F = m->ffn > 0 ? m->ffn : std::round(8.0 / 3.0 * H);
  const int E = m->experts > 0 ? m->experts : 1;
  const int K = m->topk > 0 ? m->topk : 1;
  const int SH = m->shared_experts > 0 ? m->shared_experts : 0;
  const double V = m->vocab > 0 ? m->vocab : 32000.0;
  const double N = m->params_b * 1e9;
  const double W = N * b_elem * qov;
  const double P = gotham_l2_peak_flops(g, w->precision, cscale);
  const double beff = g->bandwidth_gbps * 1e9 * bscale;
  const double smem_bw = g->sm_count * g->clock_ghz * 1e9 * g->smem_bw_per_clk;
  const bool tmem_used = g->tmem_per_sm > 0 &&
                         (w->precision == 2 || w->precision == 3 || w->precision == 4);
  const double tokens = w->phase == 0 ? static_cast<double>(B) * S : B;
  const double T = tokens;

  out->peak = P;
  out->bw = beff;
  out->ridge = P / beff;

  const double matmul_tile_smem = 64.0 * 64.0 * (2.0 * b_elem + 4.0);
  const double head_dim = m->heads > 0 ? H / m->heads : H;

  struct KRec {
    char name[32];
    double flops;
    double dram_raw;
    double smem;
    double tile_smem;
    double occ;
  };
  KRec ks[24];
  int nk = 0;

  auto add = [&](const char* name, double flops, double dram, double smem,
                 double tile_smem) {
    if (nk >= 24) return;
    KRec& k = ks[nk++];
    std::strncpy(k.name, name, 31);
    k.name[31] = '\0';
    k.flops = flops;
    k.dram_raw = dram;
    k.smem = smem;
    k.tile_smem = tile_smem;

    const double blocks_smem = (g->smem_per_sm > 0 && tile_smem > 0)
        ? std::floor(g->smem_per_sm / tile_smem) : 1e9;
    const double blocks_regs = (rpt > 0)
        ? std::floor(g->regs_per_sm / (rpt * 4.0 * tpb)) : 1e9;
    const double blocks_threads = std::floor(static_cast<double>(g->max_threads_per_sm) / tpb);
    double blocks_tmem = 1e9;
    if (tmem_used && g->tmem_per_sm > 0) {
      blocks_tmem = std::floor(g->tmem_per_sm / (128.0 * 128.0 * 4.0));
    }
    const double blocks = std::min({blocks_smem, blocks_regs, blocks_threads,
                                    static_cast<double>(g->max_blocks_per_sm), blocks_tmem});
    k.occ = std::min(1.0, blocks / occ_target);
  };

  auto wgt = [&](double rows, double cols) { return rows * cols * b_elem * qov; };
  const double act_in = T * H * b_elem;
  const double act_out = T * H * b_elem;

  /* ---- layer kernels ---- */
  /* QKV projection */
  add("qkv_proj", 2.0 * T * H * (H + 2.0 * Hkv),
      wgt(H, H + 2.0 * Hkv) + (fuse ? 0.0 : act_in + act_out),
      0.0, matmul_tile_smem);

  /* attention scores */
  if (w->phase == 0) {
    const double kv_write = T * 2.0 * Hkv * kvb;
    if (flash) {
      const double n_tiles = std::ceil(static_cast<double>(S) / qt) *
                             std::ceil(static_cast<double>(S) / kt);
      const double per_tile = qt * head_dim * b_elem +
                              2.0 * kt * head_dim * b_elem + 2.0 * qt * kt * 4.0;
      const double tile = per_tile;
      double smem = m->heads * n_tiles * per_tile;
      if (tmem_used) smem *= 0.25;  /* FA4-style: scores/accumulate in TMEM */
      add("attn_scores", 2.0 * T * S * H, kv_write, smem, tile);
      add("attn_pv", 2.0 * T * S * Hkv, 0.0, 0.0, matmul_tile_smem);
    } else {
      add("attn_scores", 2.0 * T * S * H,
          kv_write + 4.0 * T * S * b_elem, 0.0, matmul_tile_smem);
      add("attn_pv", 2.0 * T * S * Hkv, 2.0 * T * S * b_elem, 0.0, matmul_tile_smem);
    }
  } else {
    const double kv_read = B * 2.0 * S * Hkv * kvb;
    if (flash) {
      const double n_tiles = std::ceil(static_cast<double>(S) / kt);
      const int qeff = std::max(1, std::min(qt, B));
      const double per_tile = qeff * head_dim * b_elem +
                              2.0 * kt * head_dim * b_elem + 2.0 * qeff * kt * 4.0;
      const double tile = per_tile;
      double smem = m->heads * n_tiles * per_tile;
      if (tmem_used) smem *= 0.25;
      add("attn_scores", 2.0 * B * S * H, kv_read, smem, tile);
      add("attn_pv", 2.0 * B * S * Hkv, 0.0, 0.0, matmul_tile_smem);
    } else {
      add("attn_scores", 2.0 * B * S * H, kv_read + 4.0 * B * S * b_elem, 0.0, matmul_tile_smem);
      add("attn_pv", 2.0 * B * S * Hkv, 2.0 * B * S * b_elem, 0.0, matmul_tile_smem);
    }
    add("kv_write", 0.0, B * 2.0 * Hkv * kvb, 0.0, 1.0);
  }

  /* output projection */
  add("out_proj", 2.0 * T * H * H, wgt(H, H) + (fuse ? 0.0 : act_in + act_out),
      0.0, matmul_tile_smem);

  /* MLP: dense or MoE */
  if (E <= 1) {
    add("mlp", 6.0 * T * H * F, wgt(3.0 * H, F) + (fuse ? 0.0 : act_in + act_out),
        0.0, matmul_tile_smem);
  } else {
    add("mlp_router", 2.0 * T * H * E, wgt(H, E), 0.0, matmul_tile_smem);
    const double routed_flops = K * 6.0 * T * H * F;
    const double routed_wgt = K * wgt(3.0 * H, F);
    const double shared_flops = SH * 6.0 * T * H * F;
    const double shared_wgt = SH * wgt(3.0 * H, F);
    add("mlp_experts", routed_flops + shared_flops,
        routed_wgt + shared_wgt + (fuse ? 0.0 : act_in + act_out), 0.0, matmul_tile_smem);
  }

  /* ---- logits (outside layer loop, once per phase) ---- */
  {
    double flops = 2.0 * T * V * H;
    double dram = wgt(V, H) + (fuse ? 0.0 : act_in + T * V * b_elem);
    add("logits", flops, dram, 0.0, matmul_tile_smem);
  }

  /* ---- L2 reuse model (per GPU: compare sharded bytes to per-GPU L2) ---- */
  double layer_raw = 0.0;
  for (int i = 0; i < nk; ++i) layer_raw += ks[i].dram_raw;
  /* 95% cap: cache conflicts and tag overhead always leave some DRAM traffic */
  const double hit = std::min(0.95, std::min(1.0, l2_usable / (layer_raw / G)));
  out->l2_hit = hit;
  const double l2_bw = g->l2_bw_gbps * 1e9;

  /* ---- per-kernel clocks ---- */
  double layer_time = 0.0;
  double flops_weighted_occ = 0.0;
  double total_flops_g = 0.0;
  double occ_wsum = 0.0;
  double sum_c = 0.0, sum_d = 0.0, sum_s = 0.0;
  for (int i = 0; i < nk; ++i) {
    KRec& k = ks[i];
    L2Kernel* rk = &out->kernels[out->n_kernels++];
    std::strncpy(rk->name, k.name, 31);
    rk->name[31] = '\0';
    rk->flops_g = k.flops / G;
    const double bytes_g = k.dram_raw / G;
    rk->dram_g = bytes_g * (1.0 - hit);
    const double l2_bytes_g = bytes_g * hit;
    rk->smem_g = k.smem / G;
    rk->occupancy_util = k.occ;
    const double p_eff = P * k.occ;
    rk->t_compute = rk->flops_g / p_eff;
    rk->t_dram = rk->dram_g / beff;
    rk->t_l2 = l2_bytes_g / l2_bw;
    rk->t_smem = rk->smem_g / smem_bw;
    rk->t_total = std::max({rk->t_compute, rk->t_dram, rk->t_l2, rk->t_smem});
    rk->achieved = rk->t_total > 0 ? rk->flops_g / rk->t_total : 0.0;
    const double t_max = rk->t_total;
    rk->bound = rk->t_compute >= t_max - 1e-12 ? 0
              : (rk->t_dram >= t_max - 1e-12 ? 1
              : (rk->t_l2 >= t_max - 1e-12 ? 3 : 2));
    rk->valid = 1;
    layer_time += rk->t_total;
    total_flops_g += rk->flops_g;
    occ_wsum += rk->flops_g * k.occ;
    sum_c += rk->t_compute;
    sum_d += rk->t_dram;
    sum_s += rk->t_smem;
  }
  flops_weighted_occ = total_flops_g > 0 ? occ_wsum / total_flops_g : 1.0;
  out->occupancy_util = flops_weighted_occ;
  out->t_compute_sum = sum_c;
  out->t_dram_sum = sum_d;
  out->t_smem_sum = sum_s;
  out->layer_flops = total_flops_g * G;
  out->layer_dram_g = 0.0;
  for (int i = 0; i < nk; ++i) out->layer_dram_g += out->kernels[i].dram_g;
  out->layer_smem_g = 0.0;
  for (int i = 0; i < nk; ++i) out->layer_smem_g += out->kernels[i].smem_g;

  /* logits kernel is the last one; layer time excludes it */
  const double logits_t = out->kernels[nk - 1].t_total;
  const double logits_flops_g = out->kernels[nk - 1].flops_g;
  out->layer_time = layer_time - logits_t;
  out->total_time = L * out->layer_time + logits_t;
  out->tokens = w->phase == 0 ? T : B;
  out->throughput = out->total_time > 0 ? out->tokens / out->total_time : 0.0;
  out->latency_per_token = out->throughput > 0 ? 1.0 / out->throughput : 0.0;
  const double total_flops = L * (total_flops_g - logits_flops_g) + logits_flops_g;
  out->achieved = out->total_time > 0 ? total_flops / out->total_time : 0.0;
  out->utilization = P > 0 ? out->achieved / P : 0.0;
  out->p_eff = P * flops_weighted_occ;

  /* ---- memory per GPU ---- */
  const double kv_total = 2.0 * Hkv * L * B * S * kvb;
  double act_per_layer = 0.0;
  if (w->phase == 0) {
    act_per_layer = (flash ? 2.0 : 4.0) * B * S * H * b_elem +
                    (flash ? 0.0 : 2.0 * B * S * S * b_elem);
  } else {
    act_per_layer = (flash ? 2.0 : 4.0) * B * H * b_elem +
                    (flash ? 0.0 : 2.0 * B * S * b_elem);
  }
  const double act_total = act_per_layer * (w->recompute ? 1.0 : L);
  out->mem_weights = W / G;
  out->mem_kv = kv_total / G;
  out->mem_act = act_total / G;
  out->mem_total = out->mem_weights + out->mem_kv + out->mem_act;
  out->valid = 1;
  return 1;
}
