(function () {
  'use strict';

  var PALETTE = ['#7aa2f7', '#9ece6a', '#f7768e', '#e0af68', '#bb9af7', '#7dcfff', '#ff9e64', '#73daca', '#c0caf5'];
  var PREC_COLORS = { fp32: '#565f89', fp8: '#bb9af7', int8: '#ff9e64', int4: '#f7768e' };

  var state = {
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
    showOtherCeilings: true,
    showPrecisionCeilings: true,
    showSweep: false,
  };

  var catalog = null;
  var last = null;
  var requestSeq = 0;
  var timer = null;
  var rooflineChart = null;

  function $(id) { return document.getElementById(id); }
  function gpu() { return catalog && catalog.gpus.find(function (g) { return g.id === state.gpuId; }); }
  function selectedModels() { return catalog ? catalog.models.filter(function (m) { return state.models.indexOf(m.id) >= 0; }) : []; }
  function modelColor(i) { return PALETTE[i % PALETTE.length]; }
  function precisionLabel(id) {
    var p = catalog && catalog.precisions.find(function (x) { return x.id === id; });
    return p ? p.label : id;
  }

  function sig(v, n) {
    if (!isFinite(v)) return '–';
    if (v === 0) return '0';
    n = n || 3;
    var e = Math.floor(Math.log10(Math.abs(v)));
    var f = Math.pow(10, n - 1 - e);
    return String(Math.round(v * f) / f);
  }
  function fmtTFlops(v) { return sig(v / 1e12, 4) + ' TFLOPS'; }
  function fmtPct(v) { return (v * 100).toFixed(0) + '%'; }
  function fmtTime(s) {
    if (s >= 1) return s.toFixed(2) + ' s';
    if (s >= 1e-3) return (s * 1e3).toFixed(1) + ' ms';
    return (s * 1e6).toFixed(0) + ' µs';
  }

  function setStatus(msg, kind) {
    var el = $('appStatus');
    el.textContent = msg;
    el.className = 'app-status' + (kind ? ' ' + kind : '');
  }

  function sliderPos(scale) {
    return Math.max(-15, Math.min(10, Math.round(Math.log2(scale) * 10)));
  }

  function setScales(comp, bw, moveSliders) {
    state.computeScale = comp;
    state.bandwidthScale = bw;
    $('cOut').textContent = Math.round(comp * 100) + '%';
    $('bwOut').textContent = Math.round(bw * 100) + '%';
    if (moveSliders) {
      $('cSlider').value = sliderPos(comp);
      $('bwSlider').value = sliderPos(bw);
    }
  }

  function applyMode(mode) {
    state.mode = mode;
    var seg = $('modeSeg');
    seg.querySelectorAll('button').forEach(function (b) {
      b.classList.toggle('active', b.getAttribute('data-mode') === mode);
    });
    var r = (catalog && catalog.realistic) || { computeEfficiency: 0.70, bandwidthEfficiency: 0.75 };
    if (mode === 'realistic') {
      setScales(r.computeEfficiency, r.bandwidthEfficiency, true);
      $('modeHint').textContent = 'Preset calibrated from validation records: compute ' +
        Math.round(r.computeEfficiency * 100) + '% · DRAM ' +
        Math.round(r.bandwidthEfficiency * 100) + '% of spec. Sliders override.';
    } else {
      setScales(1, 1, true);
      $('modeHint').textContent = 'Classic upper-bound analysis at 100% of the spec-sheet peaks.';
    }
    refresh();
  }

  function buildGpuSelect() {
    var sel = $('gpuSelect');
    sel.innerHTML = catalog.gpus.map(function (g) {
      return '<option value="' + g.id + '">' + g.name + '</option>';
    }).join('');
    sel.value = state.gpuId;
  }

  function buildPrecisionSelects() {
    var g = gpu();
    var sel = $('precisionSelect');
    sel.innerHTML = catalog.precisions.filter(function (p) {
      return !p.needsFp8 || g.fp8_TFLOPS != null;
    }).map(function (p) {
      return '<option value="' + p.id + '">' + p.label + '</option>';
    }).join('');
    var valid = catalog.precisions.some(function (p) {
      return p.id === state.precision && (!p.needsFp8 || g.fp8_TFLOPS != null);
    });
    if (!valid) state.precision = 'fp16';
    sel.value = state.precision;

    var ksel = $('kvPrecisionSelect');
    ksel.innerHTML = catalog.kvPrecisions.map(function (p) {
      return '<option value="' + p.id + '">' + p.label + '</option>';
    }).join('');
    ksel.value = state.kvPrecision;
  }

  function buildModelList() {
    var wrap = $('modelList');
    wrap.innerHTML = '';
    catalog.models.forEach(function (m) {
      var size = m.params >= 1 ? sig(m.params, 3) + 'B' : Math.round(m.params * 1000) + 'M';
      if (m.moe) size += ' · ' + sig(m.active, 3) + 'B act.';
      var item = document.createElement('label');
      item.className = 'model-item' + (state.models.indexOf(m.id) >= 0 ? ' selected' : '');
      item.innerHTML = '<input type="checkbox" value="' + m.id + '"' + (state.models.indexOf(m.id) >= 0 ? ' checked' : '') + '>' +
        '<span class="m-name">' + m.name + '</span><span class="m-size">' + size + '</span>';
      wrap.appendChild(item);
    });
  }

  function wireEvents() {
    $('gpuSelect').addEventListener('change', function () {
      state.gpuId = this.value;
      buildPrecisionSelects();
      refresh();
    });
    $('precisionSelect').addEventListener('change', function () { state.precision = this.value; refresh(); });
    $('kvPrecisionSelect').addEventListener('change', function () { state.kvPrecision = this.value; refresh(); });

    var seg = $('phaseSeg');
    seg.addEventListener('click', function (ev) {
      var btn = ev.target.closest('button');
      if (!btn) return;
      state.phase = btn.getAttribute('data-phase');
      seg.querySelectorAll('button').forEach(function (b) { b.classList.toggle('active', b === btn); });
      refresh();
    });
    $('modeSeg').addEventListener('click', function (ev) {
      var btn = ev.target.closest('button');
      if (!btn) return;
      applyMode(btn.getAttribute('data-mode'));
    });

    function bindSlider(inputId, outId, fn) {
      $(inputId).addEventListener('input', function () {
        fn(+this.value, $(outId));
        refresh();
      });
    }
    bindSlider('bSlider', 'bOut', function (v, out) { state.B = Math.pow(2, v); out.textContent = state.B.toLocaleString('en-US'); });
    bindSlider('sSlider', 'sOut', function (v, out) { state.S = Math.pow(2, v); out.textContent = state.S.toLocaleString('en-US'); });
    bindSlider('gpuSlider', 'gOut', function (v, out) { state.gpus = Math.pow(2, v); out.textContent = state.gpus.toLocaleString('en-US'); });
    bindSlider('cSlider', 'cOut', function (v, out) { state.computeScale = Math.pow(2, v / 10); out.textContent = Math.round(state.computeScale * 100) + '%'; });
    bindSlider('bwSlider', 'bwOut', function (v, out) { state.bandwidthScale = Math.pow(2, v / 10); out.textContent = Math.round(state.bandwidthScale * 100) + '%'; });

    $('showOtherCeilings').addEventListener('change', function () { state.showOtherCeilings = this.checked; refresh(); });
    $('showPrecisionCeilings').addEventListener('change', function () { state.showPrecisionCeilings = this.checked; refresh(); });
    $('showSweep').addEventListener('change', function () { state.showSweep = this.checked; refresh(); });

    $('modelAll').addEventListener('click', function () {
      state.models = catalog.models.map(function (m) { return m.id; });
      buildModelList();
      refresh();
    });
    $('modelNone').addEventListener('click', function () {
      state.models = [];
      buildModelList();
      refresh();
    });
    $('modelList').addEventListener('change', function (ev) {
      var cb = ev.target;
      if (!cb.matches('input[type="checkbox"]')) return;
      var id = cb.value;
      var idx = state.models.indexOf(id);
      if (cb.checked && idx < 0) state.models.push(id);
      if (!cb.checked && idx >= 0) state.models.splice(idx, 1);
      buildModelList();
      refresh();
    });
  }

  function refresh() {
    if (!catalog) return;
    var seq = ++requestSeq;
    clearTimeout(timer);
    timer = setTimeout(function () {
      SimAPI.simulate({
        models: state.models,
        gpu: state.gpuId,
        cfg: {
          phase: state.phase,
          B: state.B,
          S: state.S,
          gpus: state.gpus,
          precision: state.precision,
          kvPrecision: state.kvPrecision,
          computeScale: state.computeScale,
          bandwidthScale: state.bandwidthScale,
          sweep: state.showSweep,
          showPrecisionCeilings: state.showPrecisionCeilings,
          showOtherCeilings: state.showOtherCeilings,
        },
      })
        .then(function (d) {
          if (seq !== requestSeq) return;
          last = d;
          setStatus('C++ core ' + d.coreVersion + ' · ' + d.gpu.name);
          renderAll();
        })
        .catch(function () {
          if (seq !== requestSeq) return;
          setStatus('Server error — check that python3 -m simulator.server is running', 'err');
        });
    }, 90);
  }

  function renderSpecs() {
    var g = gpu();
    $('gpuSpecs').innerHTML =
      '<div><b>HBM</b>' + g.memory_GB + ' GB</div>' +
      '<div><b>Bandwidth</b>' + g.bandwidth_GBps.toLocaleString('en-US') + ' GB/s</div>' +
      '<div><b>FP16</b>' + g.fp16_TFLOPS + ' TFLOPS</div>' +
      '<div><b>FP8</b>' + (g.fp8_TFLOPS ? g.fp8_TFLOPS + ' TFLOPS' : 'n/a') + '</div>' +
      '<div><b>FP32</b>' + g.fp32_TFLOPS + ' TFLOPS</div>' +
      '<div><b>SRAM</b>' + g.sram_MB + ' MB</div>' +
      '<div style="grid-column:1/-1"><b>Note</b>' + g.note + '</div>';
  }

  function renderStats() {
    if (!last) return;
    var g = gpu();
    $('statCards').innerHTML =
      '<div class="card"><div class="c-label">Peak compute</div><div class="c-value">' + fmtTFlops(last.peak) +
      '</div><div class="c-note">' + precisionLabel(state.precision) + (state.computeScale !== 1 ? ' · ' + Math.round(state.computeScale * 100) + '%' : '') + '</div></div>' +
      '<div class="card"><div class="c-label">Memory bandwidth</div><div class="c-value">' + Math.round(last.bw / 1e9).toLocaleString('en-US') +
      ' GB/s</div><div class="c-note">' + (state.bandwidthScale !== 1 ? Math.round(state.bandwidthScale * 100) + '% of spec' : 'spec peak') + '</div></div>' +
      '<div class="card"><div class="c-label">Ridge point</div><div class="c-value">' + sig(last.ridge, 3) +
      ' FLOP/B</div><div class="c-note">left of ridge = memory-bound</div></div>' +
      '<div class="card"><div class="c-label">GPU memory</div><div class="c-value">' + g.memory_GB +
      ' GB</div><div class="c-note">' + state.gpus + ' GPU' + (state.gpus > 1 ? 's (TP sharded)' : '') + '</div></div>';
  }

  function pointFor(model, phase, color, label, shape) {
    return {
      title: model.name + ' · ' + label,
      color: color,
      x: phase.intensity,
      y: phase.achieved,
      shape: shape,
      rows: [
        { k: 'Intensity', v: sig(phase.intensity, 3) + ' FLOP/B' },
        { k: 'Achieved', v: fmtTFlops(phase.achieved) },
        { k: 'Utilization', v: fmtPct(phase.utilization) },
        { k: 'Bound by', v: phase.bound === 'memory' ? 'Memory bandwidth' : 'Compute' },
        { k: 'Throughput', v: sig(phase.throughput, 4) + ' tok/s' },
        { k: 'Time', v: fmtTime(phase.time) + (label === 'Decode' ? ' / step' : '') },
        { k: 'FLOPs / GPU', v: Charts.fmtSI(phase.flopsPerGpu) },
        { k: 'DRAM / GPU', v: Charts.fmtBytes(phase.bytesPerGpu) },
      ],
    };
  }

  function ceilingColor(c) {
    if (c.primary) return '#7aa2f7';
    if (c.kind === 'precision') return PREC_COLORS[c.id] || '#565f89';
    return '#3b445f';
  }

  function renderRoofline() {
    var container = $('roofline');
    var legendEl = $('rooflineLegend');
    var models = selectedModels();
    if (!last || !models.length) {
      container.innerHTML = '<div class="empty">Select at least one model to plot.</div>';
      legendEl.innerHTML = '';
      return;
    }
    var points = [];
    last.results.forEach(function (r, i) {
      var color = modelColor(i);
      if (r.prefill) points.push(pointFor(r.model, r.prefill, color, 'Prefill', 'circle'));
      if (r.decode) points.push(pointFor(r.model, r.decode, color, 'Decode', 'square'));
    });
    var ceilings = last.ceilings.map(function (c) {
      return { peak: c.peak, bandwidth: c.bandwidth, color: ceilingColor(c), primary: !!c.primary };
    });
    var sweeps = (last.sweeps || []).map(function (sw, i) {
      return { color: modelColor(i), points: sw.points.map(function (p) { return { x: p.x, y: p.y }; }) };
    });
    rooflineChart.update({ points: points, ceilings: ceilings, sweeps: sweeps, primaryColor: '#7aa2f7' });
    legendEl.innerHTML = models.map(function (m, i) {
      return '<span class="lg"><span class="sw" style="background:' + modelColor(i) + '"></span>' + m.name + '</span>';
    }).join('') +
      '<span class="lg"><span class="sw sq" style="background:#d7dce8"></span>Prefill</span>' +
      '<span class="lg"><span class="sw" style="background:#d7dce8;border-radius:2px"></span>Decode</span>';
  }

  function renderMemory() {
    if (!last) return;
    var g = gpu();
    var rows = last.results.map(function (r, i) {
      return {
        label: r.model.name + (state.gpus > 1 ? ' (per GPU)' : ''),
        color: modelColor(i),
        weights: r.memory_per_gpu.weights,
        kv: r.memory_per_gpu.kv,
        act: r.memory_per_gpu.act,
        total: r.memory_per_gpu.total,
      };
    });
    $('memoryLegend').innerHTML = Charts.memoryChart($('memoryChart'), rows, g.memory_GB * 1e9);
  }

  function renderThroughput() {
    if (!last) return;
    var rows = last.results.map(function (r, i) {
      var bars = [];
      if (r.prefill) bars.push({ name: 'Prefill', label: 'Prefill', value: r.prefill.throughput, color: modelColor(i) });
      if (r.decode) bars.push({ name: 'Decode', label: 'Decode', value: r.decode.throughput, color: modelColor(i) });
      return { label: r.model.name, color: modelColor(i), bars: bars };
    });
    $('throughputLegend').innerHTML = Charts.throughputChart($('throughputChart'), rows);
  }

  function renderTable() {
    if (!last) return;
    var g = gpu();
    var body = '';
    last.results.forEach(function (r) {
      var perGpuGB = r.memory_per_gpu.total / 1e9;
      var memTag = perGpuGB > g.memory_GB
        ? '<span class="tag oom">' + perGpuGB.toFixed(1) + ' / ' + g.memory_GB + ' GB</span>'
        : '<span class="tag ok">' + perGpuGB.toFixed(1) + ' / ' + g.memory_GB + ' GB</span>';
      var decW = r.decodeWBytes ? r.decodeWBytes / state.gpus : 0;
      if (r.prefill) body += tableRow(r.model.name, 'Prefill', r.prefill, memTag, null);
      if (r.decode) body += tableRow(r.model.name, 'Decode', r.decode, memTag, decW);
    });
    $('resultsTable').innerHTML =
      '<table><thead><tr>' +
      '<th>Model</th><th>Phase</th><th>FLOPs / GPU</th><th>DRAM / GPU</th><th>Intensity</th>' +
      '<th>Achieved</th><th>Util</th><th>Bound</th><th>Tokens/s</th><th>Time</th>' +
      '<th>Decode W stream / GPU</th><th>Mem / GPU</th>' +
      '</tr></thead><tbody>' + body + '</tbody></table>';
  }

  function tableRow(name, phase, ph, memTag, decWPerGpu) {
    return '<tr>' +
      '<td class="lbl">' + name + '</td>' +
      '<td>' + phase + '</td>' +
      '<td>' + Charts.fmtSI(ph.flopsPerGpu) + '</td>' +
      '<td>' + Charts.fmtBytes(ph.bytesPerGpu) + '</td>' +
      '<td>' + sig(ph.intensity, 3) + ' FLOP/B</td>' +
      '<td>' + sig(ph.achieved / 1e12, 4) + ' T</td>' +
      '<td>' + fmtPct(ph.utilization) + '</td>' +
      '<td><span class="tag ' + ph.bound + '">' + ph.bound + '</span></td>' +
      '<td>' + sig(ph.throughput, 4) + '</td>' +
      '<td>' + fmtTime(ph.time) + '</td>' +
      '<td>' + (phase === 'Decode' && decWPerGpu ? Charts.fmtBytes(decWPerGpu) : '–') + '</td>' +
      '<td>' + memTag + '</td></tr>';
  }

  function renderAll() {
    renderSpecs();
    renderStats();
    renderRoofline();
    renderMemory();
    renderThroughput();
    renderTable();
  }

  function init() {
    rooflineChart = new Charts.RooflineChart($('roofline'), $('rooflineTip'));
    setStatus('Connecting to simulation core…');
    SimAPI.getCatalog()
      .then(function (d) {
        catalog = d;
        if (state.mode === 'realistic') applyMode('realistic');
        buildGpuSelect();
        buildPrecisionSelects();
        buildModelList();
        wireEvents();
        refresh();
      })
      .catch(function () {
        setStatus('Simulation core unavailable — start the server or fix the WASM build', 'err');
        ['roofline', 'memoryChart', 'throughputChart'].forEach(function (id) {
          $(id).innerHTML = '<div class="empty">Waiting for the simulation core…</div>';
        });
        $('resultsTable').innerHTML = '<div class="empty">Waiting for the simulation core…</div>';
      });
    window.addEventListener('resize', function () { if (last) renderAll(); });
  }

  init();
})();
