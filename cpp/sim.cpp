#include "sim.hpp"

#include <algorithm>
#include <cmath>

namespace {

const char* kVersion = "gotham-core-0.1.0";

double bytes_per_elem(int precision) {
  switch (precision) {
    case 0: return 4.0;    /* fp32 */
    case 1: return 2.0;    /* fp16 / bf16 */
    case 2: return 1.0;    /* fp8 */
    case 3: return 1.0;    /* int8 */
    case 4: return 0.5;    /* int4 */
    default: return 2.0;
  }
}

double quant_overhead(int precision) {
  if (precision == 4) return 1.06;  /* int4 scale/zero-point metadata */
  if (precision == 3) return 1.03;  /* int8 metadata */
  return 1.0;
}

double kv_bytes(int kv_precision) {
  return kv_precision == 1 ? 1.0 : 2.0;
}

double kv_dim(const GothamModel* m) {
  const int heads = m->heads > 0 ? m->heads : 1;
  const int kv_heads = m->kv_heads > 0 ? m->kv_heads : heads;
  return static_cast<double>(m->hidden) * kv_heads / heads;
}

double active_params(const GothamModel* m) {
  const double b = m->active_b > 0 ? m->active_b : m->params_b;
  return b * 1e9;
}

/* Weight bytes streamed per decode step.

   Dense models read the whole weight set once per step, independent of batch.
   MoE models with expert-parallel weight placement only need the experts that
   the current batch actually routes to. For a batch of B tokens over E routed
   experts with top-k routing, the expected number of distinct experts touched
   is E * (1 - (1 - k/E)^B): B=1 streams ~k/E of the routed weights (i.e. the
   per-token active expert set), and a full batch eventually touches every
   expert, recovering the dense weight-streaming cost.

   The shared (always-active) weight share is not published in model cards, so
   it is estimated from the catalog's total and active parameter counts:
   active ≈ shared + (k/E) * (total - shared). shared_b comes back in the same
   "B" parameter units as params_b. */
double decode_weight_bytes(const GothamModel* m, double bytes_per_param,
                           int B, double* shared_b) {
  const int E = m->experts;
  const double total = m->params_b;
  double shared = total;
  if (E > 1 && m->active_b > 0 && m->active_b < total) {
    const double k = std::max(1, std::min(m->topk > 0 ? m->topk : 1, E));
    const double kfrac = k / static_cast<double>(E);
    const double est =
        (m->active_b - kfrac * total) / (1.0 - kfrac);
    shared = std::max(0.0, std::min(total, est));
    const double exp_w = total - shared;
    const int b = B > 0 ? B : 1;
    const double fill = 1.0 - std::pow(1.0 - kfrac, static_cast<double>(b));
    shared += exp_w * fill;
  }
  if (shared_b) *shared_b = shared;
  return shared * 1e9 * bytes_per_param;
}

double flops_per_token(const GothamModel* m) {
  return 2.0 * active_params(m);
}

double attention_prefill(const GothamModel* m, int S) {
  return 2.0 * S * S * m->layers * (m->hidden + kv_dim(m));
}

double attention_decode(const GothamModel* m, int S) {
  return 2.0 * S * m->layers * (m->hidden + kv_dim(m));
}

}  // namespace

double gotham_peak_flops(const GothamGpu* gpu, int precision, double scale) {
  const double s = scale > 0 ? scale : 1.0;
  double p;
  switch (precision) {
    case 0: p = gpu->fp32_tflops; break;
    case 2: p = gpu->fp8_tflops > 0 ? gpu->fp8_tflops : gpu->fp16_tflops; break;
    case 3: p = gpu->fp8_tflops > 0 ? gpu->fp8_tflops : gpu->fp16_tflops * 2.0; break;
    case 4: p = gpu->fp16_tflops * 4.0; break;
    default: p = gpu->fp16_tflops; break;
  }
  return p * 1e12 * s;
}

int gotham_simulate(const GothamModel* m, const GothamGpu* g,
                    const GothamWorkload* cfg, GothamResult* out) {
  if (!m || !g || !cfg || !out) return 0;
  *out = GothamResult{};

  const int B = cfg->B > 0 ? cfg->B : 1;
  const int S = cfg->S > 0 ? cfg->S : 1;
  const int gpus = cfg->gpus > 0 ? cfg->gpus : 1;
  const double compute_scale = cfg->compute_scale > 0 ? cfg->compute_scale : 1.0;
  const double bw_scale = cfg->bandwidth_scale > 0 ? cfg->bandwidth_scale : 1.0;
  const double kvb = kv_bytes(cfg->kv_precision);
  const double kd = kv_dim(m);

  out->w_bytes = m->params_b * 1e9 * bytes_per_elem(cfg->precision) * quant_overhead(cfg->precision);
  out->kv_write_per_token = 2.0 * kd * m->layers * kvb;
  out->kv_read_per_token = 2.0 * kd * m->layers * S * kvb;
  out->act_per_token = 16.0 * m->hidden;
  out->kv_dim = kd;
  out->peak = gotham_peak_flops(g, cfg->precision, compute_scale);
  out->bw = g->bandwidth_gbps * 1e9 * bw_scale;
  out->ridge = out->peak / out->bw;
  const double wpp = bytes_per_elem(cfg->precision) * quant_overhead(cfg->precision);
  out->decode_w_bytes = decode_weight_bytes(m, wpp, B, &out->decode_streamed_b);

  const auto run_phase = [&](GothamPhase* ph, double flops, double bytes,
                             double tokens) {
    const double fg = flops / gpus;
    const double bg = bytes / gpus;
    const double I = fg / bg;
    const double achieved = std::min(out->peak, out->bw * I);
    ph->flops = flops;
    ph->bytes = bytes;
    ph->tokens = tokens;
    ph->flops_per_gpu = fg;
    ph->bytes_per_gpu = bg;
    ph->intensity = I;
    ph->achieved = achieved;
    ph->utilization = achieved / out->peak;
    ph->bound = I < out->ridge ? 0 : 1;
    ph->time = fg / achieved;
    ph->throughput = tokens / ph->time;
    ph->latency_per_token = ph->time / tokens;
    ph->valid = 1;
  };

  if (cfg->phase == 0 || cfg->phase == 2) {
    const double tokens = static_cast<double>(B) * S;
    const double flops = tokens * flops_per_token(m) + B * attention_prefill(m, S);
    const double bytes = out->w_bytes + tokens * (out->kv_write_per_token + out->act_per_token);
    run_phase(&out->prefill, flops, bytes, tokens);
  }
  if (cfg->phase == 1 || cfg->phase == 2) {
    const double tokens = B;
    const double flops = B * (flops_per_token(m) + attention_decode(m, S));
    const double bytes = out->decode_w_bytes + B * (out->kv_read_per_token +
                                                    out->kv_write_per_token +
                                                    out->act_per_token);
    run_phase(&out->decode, flops, bytes, tokens);
  }

  out->valid = 1;
  return 1;
}

int gotham_memory(const GothamModel* m, const GothamWorkload* cfg,
                  double out[4]) {
  if (!m || !cfg || !out) return 0;
  const int B = cfg->B > 0 ? cfg->B : 1;
  const int S = cfg->S > 0 ? cfg->S : 1;
  const double kvb = kv_bytes(cfg->kv_precision);
  const double kd = kv_dim(m);
  out[0] = m->params_b * 1e9 * bytes_per_elem(cfg->precision) * quant_overhead(cfg->precision);
  out[1] = 2.0 * kd * m->layers * B * S * kvb;
  out[2] = 16.0 * B * S * m->hidden;
  out[3] = out[0] + out[1] + out[2];
  return 1;
}

const char* gotham_version(void) {
  return kVersion;
}
