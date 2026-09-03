/* Verify the WebAssembly build against known values from the Python/C++ core.
   Run after building wasm/gotham.js:  node tools/test_wasm.js */
'use strict';

const createModule = require('../wasm/gotham.js');
const SimAPI = require('../js/simapi.js');
const catalog = require('../data.json');

function approx(actual, expected, tol, label) {
  if (Math.abs(actual - expected) / expected > tol) {
    throw new Error(label + ': got ' + actual + ', expected ' + expected);
  }
}

(async () => {
  const mod = await createModule({});
  const cfg = {
    phase: 'both', B: 1, S: 2048, gpus: 1,
    precision: 'fp16', kvPrecision: 'fp16',
    computeScale: 1, bandwidthScale: 1,
    sweep: false, showPrecisionCeilings: true, showOtherCeilings: true,
  };
  const l1 = SimAPI._buildSimulateResponse({ models: ['llama3-8b'], gpu: 'h100', cfg }, catalog, mod);
  const p = l1.results[0].prefill;
  const d = l1.results[0].decode;
  approx(p.intensity, 2081, 0.02, 'prefill intensity');
  approx(d.throughput, 205, 0.06, 'decode tok/s');
  approx(l1.ridge, 295.4, 0.01, 'ridge');
  console.log('L1 wasm ok - prefill I =', p.intensity.toFixed(1),
    '| decode tok/s =', d.throughput.toFixed(0),
    '| ceilings =', l1.ceilings.length);

  const l2 = SimAPI._buildL2Response({
    models: ['llama3-8b'], gpu: 'h100',
    cfg: {
      phase: 'both', B: 1, S: 2048, gpus: 1, precision: 'fp16', kvPrecision: 'fp16',
      computeScale: 1, bandwidthScale: 1, flashAttention: true, recompute: true, fuseLayer: true,
    },
  }, catalog, mod);
  const d2 = l2.results[0].phases.decode.l2;
  const p2 = l2.results[0].phases.prefill.l2;
  approx(d2.throughput, 225.6, 0.1, 'L2 decode tok/s');
  console.log('L2 wasm ok - prefill tok/s =', p2.throughput.toFixed(0),
    '| decode tok/s =', d2.throughput.toFixed(0),
    '| kernels =', d2.kernels.length, '| l2hit =', d2.l2Hit.toFixed(3));

  const moe = SimAPI._buildL2Response({
    models: ['mixtral-8x7b'], gpu: 'h100',
    cfg: {
      phase: 'decode', B: 1, S: 2048, gpus: 1, precision: 'fp16', kvPrecision: 'fp16',
      computeScale: 1, bandwidthScale: 1, flashAttention: true, recompute: true, fuseLayer: true,
    },
  }, catalog, mod);
  const names = moe.results[0].phases.decode.l2.kernels.map(k => k.name);
  if (names.indexOf('mlp_router') < 0 || names.indexOf('mlp_experts') < 0) {
    throw new Error('MoE kernels missing: ' + names.join(','));
  }
  console.log('MoE wasm ok - kernels:', names.join(', '));
  console.log('All WASM checks passed.');
})().catch(e => { console.error(e.message); process.exit(1); });
