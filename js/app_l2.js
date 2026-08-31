(function () {
  'use strict';

  var PALETTE = ['#7aa2f7', '#9ece6a', '#f7768e', '#e0af68', '#bb9af7', '#7dcfff', '#ff9e64', '#73daca', '#c0caf5'];

  var state = {
    models: ['llama3-8b', 'glm-5.2'],
    gpuId: 'h100',
    precision: 'fp16',
    kvPrecision: 'fp16',
    phase: 'both',
    B: 1,
    S: 2048,
    gpus: 1,
    computeScale: 1,
    bandwidthScale: 1,
    flashAttention: true,
    recompute: true,
    fuseLayer: true,
    l2UsableFrac: 0.8,
    occupancyTarget: 4,
  };

  var catalog = null;
  var last = null;
  var requestSeq = 0;
  var timer = null;

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

    function bindSlider(inputId, outId, fn) {
      $(inputId).addEventListener('input', function () {
        fn(+this.value, $(outId));
        refresh();
      });
    }
    bindSlider('bSlider', 'bOut', function (v, out) { state.B = Math.pow(2, v); out.textContent = state.B.toLocaleString('en-US'); });
    bindSlider('sSlider', 'sOut', function (v, out) { state.S = Math.pow(2, v); out.textContent = state.S.toLocaleString('en-US'); });
    bindSlider('gpuSlider', 'gOut', function (v, out) { state.gpus = Math.pow(2, v); out.textContent = state.gpus.toLocaleString('en-US'); });
    bindSlider('l2Slider', 'l2Out', function (v, out) { state.l2UsableFrac = v / 10; out.textContent = Math.round(v * 10) + '%'; });
    bindSlider('occSlider', 'occOut', function (v, out) { state.occupancyTarget = v; out.textContent = v; });

    $('flashAttention').addEventListener('change', function () { state.flashAttention = this.checked; refresh(); });
    $('recompute').addEventListener('change', function () { state.recompute = this.checked; refresh(); });
    $('fuseLayer').addEventListener('change', function () { state.fuseLayer = this.checked; refresh(); });

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
      SimAPI.simulateL2({
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
          flashAttention: state.flashAttention,
          recompute: state.recompute,
          fuseLayer: state.fuseLayer,
          l2UsableFrac: state.l2UsableFrac,
          occupancyTarget: state.occupancyTarget,
        },
      })
        .then(function (d) {
          if (seq !== requestSeq) return;
          last = d;
          setStatus('L2 core ' + d.coreVersion + ' · ' + d.gpu.name);
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
      '<div><b>SMs</b>' + g.sm_count + '</div>' +
      '<div><b>L2</b>' + g.l2_mb + ' MB</div>' +
      '<div><b>SMEM/SM</b>' + g.smem_per_sm_kb + ' KB</div>' +
      '<div><b>TMEM/SM</b>' + (g.tmem_per_sm_kb ? g.tmem_per_sm_kb + ' KB' : 'n/a') + '</div>' +
      '<div style="grid-column:1/-1"><b>Note</b>' + g.note + '</div>';
  }

  function renderStats() {
    if (!last || !last.results.length) return;
    var r = last.results[0];
    var g = gpu();
    var pf = r.phases.prefill, dc = r.phases.decode;
    var occVals = [];
    if (pf) occVals.push(pf.l2.occupancyUtil);
    if (dc) occVals.push(dc.l2.occupancyUtil);
    var occ = occVals.length ? occVals.reduce(function (a, b) { return a + b; }, 0) / occVals.length : 0;
    var hit = (pf ? pf.l2.l2Hit : (dc ? dc.l2.l2Hit : 0));
    $('statCards').innerHTML =
      '<div class="card"><div class="c-label">L2 prefill</div><div class="c-value">' + (pf ? sig(pf.l2.throughput, 4) + ' tok/s' : '–') +
      '</div><div class="c-note">L1: ' + (pf ? sig(pf.l1.throughput, 4) : '–') + ' tok/s</div></div>' +
      '<div class="card"><div class="c-label">L2 decode</div><div class="c-value">' + (dc ? sig(dc.l2.throughput, 4) + ' tok/s' : '–') +
      '</div><div class="c-note">L1: ' + (dc ? sig(dc.l1.throughput, 4) : '–') + ' tok/s</div></div>' +
      '<div class="card"><div class="c-label">Occupancy</div><div class="c-value">' + fmtPct(occ) +
      '</div><div class="c-note">CTAs/SM vs target ' + state.occupancyTarget + '</div></div>' +
      '<div class="card"><div class="c-label">L2 hit</div><div class="c-value">' + fmtPct(hit) +
      '</div><div class="c-note">of layer traffic from cache</div></div>' +
      '<div class="card"><div class="c-label">Mem / GPU</div><div class="c-value">' + (r.phases.prefill ? (r.phases.prefill.l2.memTotal / 1e9).toFixed(1) : (r.phases.decode.l2.memTotal / 1e9).toFixed(1)) +
      ' GB</div><div class="c-note">' + g.memory_GB + ' GB HBM</div></div>';
  }

  function renderKernels() {
    if (!last || !last.results.length) return;
    var r = last.results[0];
    var legend = '';
    if (r.phases.prefill) {
      $('prefillKernelTitle').style.display = '';
      $('kernelChartPrefill').style.display = '';
      legend = Charts.kernelBars($('kernelChartPrefill'), r.phases.prefill.l2.kernels);
    } else {
      $('prefillKernelTitle').style.display = 'none';
      $('kernelChartPrefill').style.display = 'none';
    }
    if (r.phases.decode) {
      $('decodeKernelTitle').style.display = '';
      $('kernelChartDecode').style.display = '';
      if (!legend) legend = Charts.kernelBars($('kernelChartDecode'), r.phases.decode.l2.kernels);
      else Charts.kernelBars($('kernelChartDecode'), r.phases.decode.l2.kernels);
    } else {
      $('decodeKernelTitle').style.display = 'none';
      $('kernelChartDecode').style.display = 'none';
    }
    $('kernelLegend').innerHTML = legend;
  }

  function renderTable() {
    if (!last) return;
    var g = gpu();
    var body = '';
    last.results.forEach(function (r) {
      ['prefill', 'decode'].forEach(function (ph) {
        if (!r.phases[ph]) return;
        var l1 = r.phases[ph].l1, l2 = r.phases[ph].l2;
        var ratio = l2.throughput > 0 ? l1.throughput / l2.throughput : 0;
        var memTag = l2.memTotal / 1e9 > g.memory_GB
          ? '<span class="tag oom">' + (l2.memTotal / 1e9).toFixed(1) + ' / ' + g.memory_GB + ' GB</span>'
          : '<span class="tag ok">' + (l2.memTotal / 1e9).toFixed(1) + ' / ' + g.memory_GB + ' GB</span>';
        body += '<tr>' +
          '<td class="lbl">' + r.model.name + '</td>' +
          '<td>' + ph + '</td>' +
          '<td>' + sig(l1.throughput, 4) + '</td>' +
          '<td>' + sig(l2.throughput, 4) + '</td>' +
          '<td>' + (ratio >= 1 ? '×' + sig(ratio, 3) + ' slower' : '×' + sig(ratio, 3) + ' faster') + '</td>' +
          '<td>' + fmtTime(l1.time) + '</td>' +
          '<td>' + fmtTime(l2.totalTime) + '</td>' +
          '<td>' + fmtPct(l2.utilization) + '</td>' +
          '<td>' + fmtPct(l2.occupancyUtil) + '</td>' +
          '<td>' + fmtPct(l2.l2Hit) + '</td>' +
          '<td>' + memTag + '</td></tr>';
      });
    });
    $('resultsTable').innerHTML =
      '<table><thead><tr>' +
      '<th>Model</th><th>Phase</th><th>L1 tok/s</th><th>L2 tok/s</th><th>L1/L2</th>' +
      '<th>L1 time</th><th>L2 time</th><th>Util</th><th>Occupancy</th><th>L2 hit</th><th>Mem / GPU</th>' +
      '</tr></thead><tbody>' + body + '</tbody></table>';
  }

  function renderMemory() {
    if (!last) return;
    var g = gpu();
    var rows = last.results.map(function (r, i) {
      var ph = r.phases.prefill || r.phases.decode;
      return {
        label: r.model.name + (state.gpus > 1 ? ' (per GPU)' : ''),
        color: modelColor(i),
        weights: ph.l2.memWeights,
        kv: ph.l2.memKv,
        act: ph.l2.memAct,
        total: ph.l2.memTotal,
      };
    });
    Charts.memoryChart($('memoryChart'), rows, g.memory_GB * 1e9);
  }

  function renderAll() {
    renderSpecs();
    renderStats();
    renderKernels();
    renderTable();
    renderMemory();
  }

  function init() {
    setStatus('Connecting to simulation core…');
    SimAPI.getCatalog()
      .then(function (d) {
        catalog = d;
        buildGpuSelect();
        buildPrecisionSelects();
        buildModelList();
        wireEvents();
        refresh();
      })
      .catch(function () {
        setStatus('Simulation core unavailable — start the server or fix the WASM build', 'err');
      });
    window.addEventListener('resize', function () { if (last) renderAll(); });
  }

  init();
})();
