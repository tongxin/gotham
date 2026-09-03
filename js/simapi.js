/* Unified simulation API: talks to the local Python server when present, and
   falls back to the in-browser WebAssembly build of the C++ core (for static
   hosting on GitHub Pages, or file:// usage). Both modes return the same JSON
   shapes as /api/simulate and /api/simulate_l2. */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.SimAPI = api;
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const PREC_ENUM = { fp32: 0, fp16: 1, fp8: 2, int8: 3, int4: 4 };
  const KV_ENUM = { fp16: 0, fp8: 1 };
  const PHASE_ENUM = { prefill: 0, decode: 1, both: 2 };
  const BOUND_NAMES = ['compute', 'dram', 'smem', 'l2'];

  /* ---- flat-array layouts (see cpp/wasm_glue.cpp) ---- */
  function modelArr(m) {
    return [
      m.params, m.active || 0, m.layers, m.hidden, m.heads, m.kv_heads || 0,
      m.vocab || 32000, m.ffn || 0, m.experts || 0, m.topk || 0, m.shared || 0,
    ];
  }
  function gpu1Arr(g) {
    return [g.fp16_TFLOPS, g.fp8_TFLOPS || 0, g.fp32_TFLOPS, g.bandwidth_GBps, g.memory_GB];
  }
  function gpu2Arr(g) {
    return [
      g.fp16_TFLOPS, g.fp8_TFLOPS || 0, g.fp32_TFLOPS, g.bandwidth_GBps, g.memory_GB,
      g.sm_count, g.clock_ghz, g.smem_per_sm_kb * 1024, g.regs_per_sm_kb * 1024,
      g.l1_per_sm_kb * 1024, g.l2_mb * 1024 * 1024, g.l2_bw_gbps,
      (g.tmem_per_sm_kb || 0) * 1024, g.smem_bw_b_per_clk,
      2048, 64, 32,
    ];
  }
  function cfg1Arr(c) {
    return [
      PHASE_ENUM[c.phase] != null ? PHASE_ENUM[c.phase] : 2,
      c.B, c.S, c.gpus,
      PREC_ENUM[c.precision], KV_ENUM[c.kvPrecision || 'fp16'],
      c.computeScale != null ? c.computeScale : 1,
      c.bandwidthScale != null ? c.bandwidthScale : 1,
    ];
  }
  function cfg2Arr(c) {
    return [
      ...cfg1Arr(c),
      c.flashAttention ? 1 : 0, c.recompute ? 1 : 0, c.fuseLayer ? 1 : 0,
      c.qTile || 64, c.kTile || 64,
      c.l2UsableFrac != null ? c.l2UsableFrac : 0.8,
      c.occupancyTarget || 4, c.threadsPerBlock || 512, c.regsPerThread || 64,
    ];
  }

  function allocDoubles(mod, arr) {
    const p = mod._malloc(arr.length * 8);
    (mod.HEAPF64 || new Float64Array(mod.wasmMemory.buffer)).set(arr, p / 8);
    return p;
  }
  function readDoubles(mod, p, n) {
    const f64 = mod.HEAPF64 || new Float64Array(mod.wasmMemory.buffer);
    return Array.prototype.slice.call(f64.subarray(p / 8, p / 8 + n));
  }
  function readCString(mod, p, cap) {
    const u8 = mod.HEAPU8 || new Uint8Array(mod.wasmMemory.buffer);
    let s = '';
    for (let i = 0; i < cap; i++) {
      const b = u8[p + i];
      if (!b) break;
      s += String.fromCharCode(b);
    }
    return s;
  }

  function wasmL1(mod, m, gpu, cfg) {
    const pm = allocDoubles(mod, modelArr(m));
    const pg = allocDoubles(mod, gpu1Arr(gpu));
    const pc = allocDoubles(mod, cfg1Arr(cfg));
    const po = mod._malloc(37 * 8);
    const ok = mod._gotham_wasm_simulate(pm, pg, pc, po);
    const out = ok ? readDoubles(mod, po, 37) : null;
    mod._free(pm); mod._free(pg); mod._free(pc); mod._free(po);
    return out;
  }

  function wasmMem(mod, m, cfg) {
    const pm = allocDoubles(mod, modelArr(m));
    const pc = allocDoubles(mod, cfg1Arr(cfg));
    const po = mod._malloc(4 * 8);
    mod._gotham_wasm_memory(pm, pc, po);
    const out = readDoubles(mod, po, 4);
    mod._free(pm); mod._free(pc); mod._free(po);
    return { weights: out[0], kv: out[1], act: out[2], total: out[3] };
  }

  function wasmPeak(mod, gpuArr, precision, scale) {
    const pg = allocDoubles(mod, gpuArr);
    const v = mod._gotham_wasm_peak_flops(pg, PREC_ENUM[precision] != null ? PREC_ENUM[precision] : 1, scale != null ? scale : 1);
    mod._free(pg);
    return v;
  }

  function wasmVersion(mod) {
    const ptr = mod._gotham_wasm_version();
    if (mod.UTF8ToString) return mod.UTF8ToString(ptr);
    return readCString(mod, ptr, 64);
  }

  function wasmL2(mod, m, gpu, cfg) {
    const pm = allocDoubles(mod, modelArr(m));
    const pg = allocDoubles(mod, gpu2Arr(gpu));
    const pc = allocDoubles(mod, cfg2Arr(cfg));
    const po = mod._malloc(313 * 8);
    const pn = mod._malloc(24 * 32);
    const ok = mod._gotham_wasm_l2_simulate(pm, pg, pc, po, pn, 24 * 32);
    const out = ok ? readDoubles(mod, po, 313) : null;
    const names = [];
    for (let i = 0; i < 24; i++) names.push(readCString(mod, pn + i * 32, 32));
    mod._free(pm); mod._free(pg); mod._free(pc); mod._free(po); mod._free(pn);
    if (!out) return null;
    const n = out[288];
    const kernels = [];
    for (let i = 0; i < n; i++) {
      const b = i * 12;
      kernels.push({
        name: names[i],
        flopsPerGpu: out[b],
        dramPerGpu: out[b + 1],
        smemPerGpu: out[b + 2],
        tCompute: out[b + 3],
        tDram: out[b + 4],
        tSmem: out[b + 5],
        tL2: out[b + 6],
        tTotal: out[b + 7],
        achieved: out[b + 8],
        occupancyUtil: out[b + 9],
        bound: BOUND_NAMES[out[b + 10]] || 'compute',
      });
    }
    return {
      kernels,
      layerFlops: out[289],
      layerDramPerGpu: out[290],
      layerSmemPerGpu: out[291],
      layerTime: out[292],
      totalTime: out[293],
      tokens: out[294],
      throughput: out[295],
      latencyPerToken: out[296],
      achieved: out[297],
      utilization: out[298],
      pEff: out[299],
      tComputeSum: out[300],
      tDramSum: out[301],
      tSmemSum: out[302],
      occupancyUtil: out[303],
      l2Hit: out[304],
      memWeights: out[305],
      memKv: out[306],
      memAct: out[307],
      memTotal: out[308],
      ridge: out[309],
      peak: out[310],
      bw: out[311],
    };
  }

  function phaseDict(o, off) {
    if (!o || !o[off + 12]) return null;
    return {
      flops: o[off],
      bytes: o[off + 1],
      tokens: o[off + 2],
      flopsPerGpu: o[off + 3],
      bytesPerGpu: o[off + 4],
      intensity: o[off + 5],
      achieved: o[off + 6],
      utilization: o[off + 7],
      time: o[off + 8],
      throughput: o[off + 9],
      latencyPerToken: o[off + 10],
      bound: o[off + 11] === 0 ? 'memory' : 'compute',
    };
  }

  function buildCeilings(catalog, gpu, cfg, mod, gpuArr) {
    const bw = gpu.bandwidth_GBps * 1e9 * (cfg.bandwidthScale != null ? cfg.bandwidthScale : 1);
    const scale = cfg.computeScale != null ? cfg.computeScale : 1;
    const label = id => {
      const p = catalog.precisions.find(x => x.id === id);
      return p ? p.label : id;
    };
    const ceilings = [{
      kind: 'primary', id: cfg.precision,
      peak: wasmPeak(mod, gpuArr, cfg.precision, scale),
      bandwidth: bw,
      label: gpu.name + ' · ' + label(cfg.precision),
      primary: true,
    }];
    if (cfg.showPrecisionCeilings !== false) {
      catalog.precisions.forEach(p => {
        if (p.id === cfg.precision) return;
        if (p.needsFp8 && !gpu.fp8_TFLOPS) return;
        ceilings.push({
          kind: 'precision', id: p.id,
          peak: wasmPeak(mod, gpuArr, p.id, scale),
          bandwidth: bw, label: label(p.id), primary: false,
        });
      });
    }
    if (cfg.showOtherCeilings !== false) {
      catalog.gpus.forEach(o => {
        if (o.id === gpu.id) return;
        ceilings.push({
          kind: 'gpu', id: o.id,
          peak: o.fp16_TFLOPS * 1e12,
          bandwidth: o.bandwidth_GBps * 1e9,
          label: o.name, primary: false,
        });
      });
    }
    return ceilings;
  }

  function buildSimulateResponse(payload, catalog, mod) {
    const cfg = payload.cfg || {};
    const gpu = catalog.gpus.find(g => g.id === payload.gpu) || catalog.gpus[0];
    const models = catalog.models.filter(m => (payload.models || []).indexOf(m.id) >= 0);
    const gpuArr = gpu1Arr(gpu);
    const first = wasmL1(mod, models[0], gpu, cfg);
    const results = models.map(m => {
      const out = wasmL1(mod, m, gpu, cfg);
      const mem = wasmMem(mod, m, cfg);
      const g = cfg.gpus || 1;
      return {
        model: m,
        prefill: phaseDict(out, 0),
        decode: phaseDict(out, 13),
        decodeWBytes: out[35],
        decodeStreamedB: out[36],
        memory_per_gpu: {
          weights: mem.weights / g,
          kv: mem.kv / g,
          act: mem.act / g,
          total: mem.total / g,
        },
      };
    });
    const sweeps = [];
    if (cfg.sweep && (!cfg.phase || cfg.phase === 'decode' || cfg.phase === 'both')) {
      models.forEach(m => {
        const points = [];
        for (let i = 0; i < 44; i++) {
          const b = Math.max(1, Math.round(Math.pow(1024, i / 43)));
          const cc = Object.assign({}, cfg, { phase: 'decode', B: b });
          const o = wasmL1(mod, m, gpu, cc);
          const d = phaseDict(o, 13);
          if (d) points.push({ B: b, x: d.intensity, y: d.achieved });
        }
        sweeps.push({ model: m.id, points });
      });
    }
    return {
      gpu,
      peak: first[31],
      bw: first[32],
      ridge: first[33],
      decodeWBytes: first[35],
      decodeStreamedB: first[36],
      coreVersion: wasmVersion(mod),
      ceilings: buildCeilings(catalog, gpu, cfg, mod, gpuArr),
      results,
      sweeps,
    };
  }

  function buildL2Response(payload, catalog, mod) {
    const cfg = payload.cfg || {};
    const gpu = catalog.gpus.find(g => g.id === payload.gpu) || catalog.gpus[0];
    const models = catalog.models.filter(m => (payload.models || []).indexOf(m.id) >= 0);
    const phases = (!cfg.phase || cfg.phase === 'both') ? ['prefill', 'decode'] : [cfg.phase];
    const results = models.map(m => {
      const entry = { model: m, phases: {} };
      phases.forEach(ph => {
        const cc = Object.assign({}, cfg, { phase: ph });
        const out = wasmL1(mod, m, gpu, cc);
        entry.phases[ph] = {
          l2: wasmL2(mod, m, gpu, cc),
          l1: phaseDict(out, ph === 'prefill' ? 0 : 13),
        };
      });
      return entry;
    });
    return { gpu, coreVersion: wasmVersion(mod), results, phases };
  }

  function fetchWithTimeout(url, ms) {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), ms);
    return fetch(url, { signal: ctrl.signal }).finally(() => clearTimeout(t));
  }

  function post(path, payload) {
    return fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }).then(r => {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.json();
    });
  }

  function loadWasmModule() {
    return new Promise((resolve, reject) => {
      if (typeof createGothamModule !== 'undefined') { resolve(createGothamModule); return; }
      if (typeof module === 'object' && module.require) {
        try { resolve(module.require('../wasm/gotham.js')); } catch (e) { reject(e); }
        return;
      }
      const s = document.createElement('script');
      s.src = 'wasm/gotham.js';
      s.onload = () => resolve(window.createGothamModule);
      s.onerror = () => reject(new Error('wasm/gotham.js failed to load'));
      document.head.appendChild(s);
    });
  }

  const state = { mode: 'server', catalog: null, mod: null, ready: null };

  function init() {
    if (state.ready) return state.ready;
    state.ready = (async () => {
      try {
        const r = await fetchWithTimeout('api/data', 700);
        if (r.ok) {
          state.catalog = await r.json();
          state.mode = 'server';
          return;
        }
      } catch (e) { /* no server — fall through to WASM */ }
      state.mode = 'wasm';
      const create = await loadWasmModule();
      state.mod = await create({});
      if (typeof GOTHAM_CATALOG !== 'undefined') {
        state.catalog = GOTHAM_CATALOG;
      } else {
        const r = await fetchWithTimeout('data.json', 3000);
        state.catalog = await r.json();
      }
    })();
    return state.ready;
  }

  function getCatalog() {
    return init().then(() => state.catalog);
  }

  function simulate(payload) {
    return init().then(() =>
      state.mode === 'server'
        ? post('api/simulate', payload)
        : buildSimulateResponse(payload, state.catalog, state.mod)
    );
  }

  function simulateL2(payload) {
    return init().then(() =>
      state.mode === 'server'
        ? post('api/simulate_l2', payload)
        : buildL2Response(payload, state.catalog, state.mod)
    );
  }

  return {
    get mode() { return state.mode; },
    getCatalog,
    simulate,
    simulateL2,
    _buildSimulateResponse: buildSimulateResponse,
    _buildL2Response: buildL2Response,
  };
});
