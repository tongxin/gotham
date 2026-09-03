/* Shared L1 selection state between index.html and validation.html.

   index.html persists its current selection (models, GPU, precision, KV
   precision, phase, B/S/G, mode, efficiency sliders) into localStorage on every
   refresh. validation.html reads it to filter the measured benchmark records
   and replay them at the exact operating point the L1 page is showing. */
(function (root) {
  'use strict';

  var KEY = 'gotham.l1Config.v1';
  var DEFAULTS = {
    models: ['llama3-8b', 'gpt2-1p5b', 'mixtral-8x7b'],
    gpuId: 'h100',
    precision: 'fp16',
    kvPrecision: 'fp16',
    phase: 'both',
    B: 1,
    S: 2048,
    gpus: 1,
    mode: 'realistic',
    computeScale: 0.70,
    bandwidthScale: 0.75,
  };

  function read() {
    try {
      var raw = root.localStorage && root.localStorage.getItem(KEY);
      if (!raw) return null;
      return Object.assign({}, DEFAULTS, JSON.parse(raw));
    } catch (e) {
      return null;
    }
  }

  function write(cfg) {
    try {
      root.localStorage && root.localStorage.setItem(KEY, JSON.stringify(cfg));
    } catch (e) { /* storage unavailable (file:// privacy) — ignore */ }
  }

  function onChange(fn) {
    if (!root.addEventListener) return;
    root.addEventListener('storage', function (e) {
      if (e.key === KEY && e.newValue) fn(read());
    });
  }

  root.L1State = {
    KEY: KEY,
    DEFAULTS: DEFAULTS,
    read: read,
    write: write,
    onChange: onChange,
  };
})(typeof window !== 'undefined' ? window : this);
