(function () {
  'use strict';

  var DATA = window.GPU_SIM_DATA;
  var Sim = window.Sim;
  var Charts = window.Charts;

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
    computeScale: 1,
    bandwidthScale: 1,
    showOtherCeilings: true,
    showPrecisionCeilings: true,
    showSweep: false,
  };

  function $(id) { return document.getElementById(id); }
  function gpu() { return DATA.gpus.find(function (g) { return g.id === state.gpuId; }); }
  function selectedModels() { return DATA.models.filter(function (m) { return state.models.indexOf(m.id) >= 0; }); }
  function modelColor(i) { return PALETTE[i % PALETTE.length]; }
  function precisionLabel(id) {
    var p = DATA.precisions.find(function (x) { return x.id === id; });
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

  function buildGpuSelect() {
    var sel = $('gpuSelect');
    sel.innerHTML = DATA.gpus.map(function (g) {
      return '<option value="' + g.id + '">' + g.name + '</option>';
    }).join('');
    sel.value = state.gpuId;
  }

  function buildPrecisionSelects() {
    var g = gpu();
    var sel = $('precisionSelect');
    sel.innerHTML = DATA.precisions.filter(function (p) {
      return !p.needsFp8 || g.fp8_TFLOPS != null;
    }).map(function (p) {
      return '<option value="' + p.id + '">' + p.label + '</option>';
    }).join('');
    if (DATA.precisions.filter(function (p) { return p.id === state.precision; }).length === 0 ||
        !DATA.precisions.find(function (p) { return p.id === state.precision && (!p.needsFp8 || g.fp8_TFLOPS != null); })) {
      state.precision = 'fp16';
    }
    sel.value = state.precision;

    var ksel = $('kvPrecisionSelect');
    ksel.innerHTML = DATA.kvPrecisions.map(function (p) {
      return '<option value="' + p.id + '">' + p.label + '</option>';
    }).join('');
    ksel.value = state.kvPrecision;
  }

  function buildModelList() {
    var wrap = $('modelList');
    wrap.innerHTML = '';
    DATA.models.forEach(function (m, i) {
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
      renderAll();
    });
    $('precisionSelect').addEventListener('change', function () { state.precision = this.value; renderAll(); });
    $('kvPrecisionSelect').addEventListener('change', function () { state.kvPrecision = this.value; renderAll(); });

    var seg = $('phaseSeg');
    seg.addEventListener('click', function (ev) {
      var btn = ev.target.closest('button');
      if (!btn) return;
      state.phase = btn.getAttribute('data-phase');
      seg.querySelectorAll('button').forEach(function (b) { b.classList.toggle('active', b === btn); });
      renderAll();
    });

    function bindSlider(inputId, outId, fn) {
      $(inputId).addEventListener('input', function () {
        fn(+this.value, $(outId));
        renderAll();
      });
    }
    bindSlider('bSlider', 'bOut', function (v, out) { state.B = Math.pow(2, v); out.textContent = state.B.toLocaleString('en-US'); });
    bindSlider('sSlider', 'sOut', function (v, out) { state.S = Math.pow(2, v); out.textContent = state.S.toLocaleString('en-US'); });
    bindSlider('gpuSlider', 'gOut', function (v, out) { state.gpus = Math.pow(2, v); out.textContent = state.gpus.toLocaleString('en-US'); });
    bindSlider('cSlider', 'cOut', function (v, out) { state.computeScale = Math.pow(2, v / 10); out.textContent = Math.round(state.computeScale * 100) + '%'; });
    bindSlider('bwSlider', 'bwOut', function (v, out) { state.bandwidthScale = Math.pow(2, v / 10); out.textContent = Math.round(state.bandwidthScale * 100) + '%'; });

    $('showOtherCeilings').addEventListener('change', function () { state.showOtherCeilings = this.checked; renderAll(); });
    $('showPrecisionCeilings').addEventListener('change', function () { state.showPrecisionCeilings = this.checked; renderAll(); });
    $('showSweep').addEventListener('change', function () { state.showSweep = this.checked; renderAll(); });

    $('modelAll').addEventListener('click', function () {
      state.models = DATA.models.map(function (m) { return m.id; });
      buildModelList();
      renderAll();
    });
    $('modelNone').addEventListener('click', function () {
      state.models = [];
      buildModelList();
      renderAll();
    });
    $('modelList').addEventListener('change', function (ev) {
      var cb = ev.target;
      if (!cb.matches('input[type="checkbox"]')) return;
      var id = cb.value;
      var idx = state.models.indexOf(id);
      if (cb.checked && idx < 0) state.models.push(id);
      if (!cb.checked && idx >= 0) state.models.splice(idx, 1);
      buildModelList();
      renderAll();
    });
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
    var g = gpu();
    var peak = Sim.peakFlops(g, state.precision, state.computeScale);
    var bw = g.bandwidth_GBps * 1e9 * state.bandwidthScale;
    $('statCards').innerHTML =
      '<div class="card"><div class="c-label">Peak compute</div><div class="c-value">' + fmtTFlops(peak) +
      '</div><div class="c-note">' + precisionLabel(state.precision) + (state.computeScale !== 1 ? ' · ' + Math.round(state.computeScale * 100) + '%' : '') + '</div></div>' +
      '<div class="card"><div class="c-label">Memory bandwidth</div><div class="c-value">' + Math.round(bw / 1e9).toLocaleString('en-US') +
      ' GB/s</div><div class="c-note">' + (state.bandwidthScale !== 1 ? Math.round(state.bandwidthScale * 100) + '% of spec' : 'spec peak') + '</div></div>' +
      '<div class="card"><div class="c-label">Ridge point</div><div class="c-value">' + sig(peak / bw, 3) +
      ' FLOP/B</div><div class="c-note">left of ridge = memory-bound</div></div>' +
      '<div class="card"><div class="c-label">GPU memory</div><div class="c-value">' + g.memory_GB +
      ' GB</div><div class="c-note">' + state.gpus + ' GPU' + (state.gpus > 1 ? 's (TP sharded)' : '') + '</div></div>';
  }

  function pointFor(modelInfo, phase, label, shape) {
    return {
      title: modelInfo.model.name + ' · ' + label,
      color: modelInfo.color,
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

  function renderRoofline() {
    var container = $('roofline');
    var legendEl = $('rooflineLegend');
    var g = gpu();
    var models = selectedModels();
    if (!models.length) {
      container.innerHTML = '<div class="empty">Select at least one model to plot.</div>';
      legendEl.innerHTML = '';
      return;
    }

    var sims = models.map(function (m, i) {
      return { model: m, sim: Sim.simulate(m, g, state), color: modelColor(i) };
    });
    var points = [];
    sims.forEach(function (s) {
      if (s.sim.prefill) points.push(pointFor(s, s.sim.prefill, 'Prefill', 'circle'));
      if (s.sim.decode) points.push(pointFor(s, s.sim.decode, 'Decode', 'square'));
    });

    var peak = sims[0].sim.peak;
    var bw = sims[0].sim.bw;
    var ceilings = [{
      peak: peak,
      bandwidth: bw,
      color: '#7aa2f7',
      label: g.name + ' · ' + precisionLabel(state.precision),
      primary: true,
    }];

    if (state.showPrecisionCeilings) {
      DATA.precisions.forEach(function (p) {
        if (p.id === state.precision) return;
        if (p.needsFp8 && g.fp8_TFLOPS == null) return;
        ceilings.push({
          peak: Sim.peakFlops(g, p.id, state.computeScale),
          bandwidth: bw,
          color: PREC_COLORS[p.id] || '#565f89',
          label: precisionLabel(p.id),
        });
      });
    }
    if (state.showOtherCeilings) {
      DATA.gpus.forEach(function (other) {
        if (other.id === g.id) return;
        ceilings.push({
          peak: other.fp16_TFLOPS * 1e12,
          bandwidth: other.bandwidth_GBps * 1e9,
          color: '#3b445f',
          label: other.name,
        });
      });
    }

    var sweeps = [];
    if (state.showSweep && (state.phase === 'decode' || state.phase === 'both')) {
      sims.forEach(function (s) {
        sweeps.push({
          color: s.color,
          points: Sim.decodeSweep(s.model, g, state).map(function (p) { return { x: p.x, y: p.y }; }),
        });
      });
    }

    rooflineChart.update({ points: points, ceilings: ceilings, sweeps: sweeps, primaryColor: '#7aa2f7' });
    legendEl.innerHTML = models.map(function (m, i) {
      return '<span class="lg"><span class="sw" style="background:' + modelColor(i) + '"></span>' + m.name + '</span>';
    }).join('') +
      '<span class="lg"><span class="sw sq" style="background:#d7dce8"></span>Prefill</span>' +
      '<span class="lg"><span class="sw" style="background:#d7dce8;border-radius:2px"></span>Decode</span>';
  }

  function renderMemory() {
    var g = gpu();
    var models = selectedModels();
    var rows = models.map(function (m, i) {
      var mem = Sim.memoryFootprint(m, state);
      return {
        label: m.name + (state.gpus > 1 ? ' (per GPU)' : ''),
        color: modelColor(i),
        weights: mem.weights / state.gpus,
        kv: mem.kv / state.gpus,
        act: mem.act / state.gpus,
        total: mem.total / state.gpus,
      };
    });
    $('memoryLegend').innerHTML = Charts.memoryChart($('memoryChart'), rows, g.memory_GB * 1e9);
  }

  function renderThroughput() {
    var g = gpu();
    var models = selectedModels();
    var rows = models.map(function (m, i) {
      var s = Sim.simulate(m, g, state);
      var bars = [];
      if (s.prefill) bars.push({ name: 'Prefill', label: 'Prefill', value: s.prefill.throughput, color: modelColor(i) });
      if (s.decode) bars.push({ name: 'Decode', label: 'Decode', value: s.decode.throughput, color: modelColor(i) });
      return { label: m.name, color: modelColor(i), bars: bars };
    });
    $('throughputLegend').innerHTML = Charts.throughputChart($('throughputChart'), rows);
  }

  function renderTable() {
    var g = gpu();
    var models = selectedModels();
    if (!models.length) {
      $('resultsTable').innerHTML = '<div class="empty">Select at least one model.</div>';
      return;
    }
    var body = '';
    models.forEach(function (m, i) {
      var s = Sim.simulate(m, g, state);
      var mem = Sim.memoryFootprint(m, state);
      var perGpuGB = mem.total / state.gpus / 1e9;
      var memTag = perGpuGB > g.memory_GB
        ? '<span class="tag oom">' + perGpuGB.toFixed(1) + ' / ' + g.memory_GB + ' GB</span>'
        : '<span class="tag ok">' + perGpuGB.toFixed(1) + ' / ' + g.memory_GB + ' GB</span>';
      if (s.prefill) body += tableRow(m.name, 'Prefill', s.prefill, memTag);
      if (s.decode) body += tableRow(m.name, 'Decode', s.decode, memTag);
    });
    $('resultsTable').innerHTML =
      '<table><thead><tr>' +
      '<th>Model</th><th>Phase</th><th>FLOPs / GPU</th><th>DRAM / GPU</th><th>Intensity</th>' +
      '<th>Achieved</th><th>Util</th><th>Bound</th><th>Tokens/s</th><th>Time</th><th>Mem / GPU</th>' +
      '</tr></thead><tbody>' + body + '</tbody></table>';
  }

  function tableRow(name, phase, ph, memTag) {
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

  var rooflineChart = new Charts.RooflineChart($('roofline'), $('rooflineTip'));
  init();
  window.addEventListener('resize', function () { renderAll(); });
})();
