(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.Sim = api;
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const BYTES_PER_ELEM = { fp32: 4, fp16: 2, bf16: 2, fp8: 1, int8: 1, int4: 0.5 };
  const QUANT_OVERHEAD = { int4: 1.06, int8: 1.03 };

  function bytesPerElement(precision) {
    return BYTES_PER_ELEM[precision] || 2;
  }

  function weightBytes(model, precision) {
    const overhead = QUANT_OVERHEAD[precision] || 1;
    return model.params * 1e9 * bytesPerElement(precision) * overhead;
  }

  function kvDim(model) {
    const h = model.heads || 1;
    const kh = model.kv_heads || h;
    return (model.hidden * kh) / h;
  }

  function activeParams(model) {
    return (model.active || model.params) * 1e9;
  }

  function flopsPerToken(model) {
    return 2 * activeParams(model);
  }

  function attentionPrefill(model, S) {
    return 2 * S * S * model.layers * (model.hidden + kvDim(model));
  }

  function attentionDecode(model, S) {
    return 2 * S * model.layers * (model.hidden + kvDim(model));
  }

  function peakFlops(gpu, precision, scale) {
    const s = scale || 1;
    let p;
    if (precision === 'fp32') p = gpu.fp32_TFLOPS;
    else if (precision === 'fp8') p = gpu.fp8_TFLOPS || gpu.fp16_TFLOPS;
    else if (precision === 'int8') p = (gpu.fp8_TFLOPS || gpu.fp16_TFLOPS * 2);
    else if (precision === 'int4') p = gpu.fp16_TFLOPS * 4;
    else p = gpu.fp16_TFLOPS;
    return p * 1e12 * s;
  }

  /**
   * Per-phase FLOPs and DRAM traffic for a model under a workload config.
   * cfg: { precision, kvPrecision, phase, B, S }
   */
  function computeWorkload(model, cfg) {
    const kvBytes = bytesPerElement(cfg.kvPrecision || 'fp16');
    const wBytes = weightBytes(model, cfg.precision);
    const kd = kvDim(model);
    const kvWritePerToken = 2 * kd * model.layers * kvBytes;
    const kvReadPerToken = 2 * kd * model.layers * cfg.S * kvBytes;
    const actPerToken = 16 * model.hidden; // rough activation traffic per token
    const out = { wBytes, kvWritePerToken, kvReadPerToken, actPerToken };

    if (cfg.phase === 'prefill' || cfg.phase === 'both') {
      const tokens = cfg.B * cfg.S;
      out.prefill = {
        tokens,
        flops: tokens * flopsPerToken(model) + cfg.B * attentionPrefill(model, cfg.S),
        bytes: wBytes + tokens * (kvWritePerToken + actPerToken),
      };
    }
    if (cfg.phase === 'decode' || cfg.phase === 'both') {
      out.decode = {
        tokensPerStep: cfg.B,
        flops: cfg.B * (flopsPerToken(model) + attentionDecode(model, cfg.S)),
        bytes: wBytes + cfg.B * (kvReadPerToken + kvWritePerToken + actPerToken),
      };
    }
    return out;
  }

  /**
   * Full roofline simulation for one model on one GPU type.
   * cfg: { precision, kvPrecision, phase, B, S, gpus, computeScale, bandwidthScale }
   */
  function simulate(model, gpu, cfg) {
    const work = computeWorkload(model, cfg);
    const peak = peakFlops(gpu, cfg.precision, cfg.computeScale);
    const bw = gpu.bandwidth_GBps * 1e9 * (cfg.bandwidthScale || 1);
    const ridge = peak / bw;
    const g = cfg.gpus || 1;
    const res = { model, gpu, cfg, work, peak, bw, ridge };

    function runPhase(key, tokens) {
      const w = work[key];
      const flopsG = w.flops / g;
      const bytesG = w.bytes / g;
      const intensity = flopsG / bytesG;
      const achieved = Math.min(peak, bw * intensity);
      const bound = intensity < ridge ? 'memory' : 'compute';
      const time = flopsG / achieved;
      res[key] = {
        flops: w.flops,
        bytes: w.bytes,
        tokens,
        flopsPerGpu: flopsG,
        bytesPerGpu: bytesG,
        intensity,
        achieved,
        utilization: achieved / peak,
        bound,
        time,
        throughput: tokens / time,
        latencyPerToken: time / tokens,
      };
    }

    if (work.prefill) runPhase('prefill', work.prefill.tokens);
    if (work.decode) runPhase('decode', work.decode.tokensPerStep);
    return res;
  }

  /**
   * Weights + KV cache + activation buffer estimate, in bytes (total across all GPUs).
   */
  function memoryFootprint(model, cfg) {
    const kd = kvDim(model);
    const kv = 2 * kd * model.layers * cfg.B * cfg.S * bytesPerElement(cfg.kvPrecision || 'fp16');
    const act = 16 * cfg.B * cfg.S * model.hidden;
    const weights = weightBytes(model, cfg.precision);
    return { weights, kv, act, total: weights + kv + act };
  }

  /**
   * Decode trajectory across batch sizes for the roofline sweep.
   */
  function decodeSweep(model, gpu, cfg) {
    const pts = [];
    const c = Object.assign({}, cfg, { phase: 'decode' });
    const steps = 44;
    for (let i = 0; i < steps; i++) {
      c.B = Math.max(1, Math.round(Math.pow(1024, i / (steps - 1))));
      const s = simulate(model, gpu, c);
      pts.push({ B: c.B, x: s.decode.intensity, y: s.decode.achieved, bound: s.decode.bound });
    }
    return pts;
  }

  return {
    bytesPerElement,
    weightBytes,
    kvDim,
    activeParams,
    flopsPerToken,
    attentionPrefill,
    attentionDecode,
    peakFlops,
    computeWorkload,
    simulate,
    memoryFootprint,
    decodeSweep,
  };
});
