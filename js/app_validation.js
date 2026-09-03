(function () {
  'use strict';

  function $(id) { return document.getElementById(id); }

  var catalog = null;
  var coreVersion = '—';

  function escapeHtml(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }
  function num(v, n) {
    n = n == null ? 3 : n;
    if (!isFinite(v)) return '–';
    if (v === 0) return '0';
    var e = Math.floor(Math.log10(Math.abs(v)));
    var f = Math.pow(10, n - 1 - e);
    return String(Math.round(v * f) / f);
  }
  function errPct(e) {
    if (e == null || !isFinite(e)) return '–';
    var cls = Math.abs(e) > 50 ? 'oom' : (Math.abs(e) > 15 ? 'warn' : 'ok');
    return '<span class="tag ' + cls + '">' + (e >= 0 ? '+' : '−') + ' ' +
      Math.abs(e).toFixed(1) + '%</span>';
  }
  function setStatus(msg, kind) {
    var el = $('appStatus');
    el.textContent = msg;
    el.className = 'app-status' + (kind ? ' ' + kind : '');
  }

  function baseCfg(rec, phase, mode) {
    var r = (catalog && catalog.realistic) || {};
    var ce = mode === 'realistic' ? (r.computeEfficiency != null ? r.computeEfficiency : 0.70) : 1;
    var be = mode === 'realistic' ? (r.bandwidthEfficiency != null ? r.bandwidthEfficiency : 0.75) : 1;
    return {
      phase: phase,
      B: rec.B,
      S: rec.S,
      gpus: rec.gpus,
      precision: rec.precision,
      kvPrecision: rec.kvPrecision,
      computeScale: ce,
      bandwidthScale: be,
    };
  }

  function call(rec, cfg) {
    return SimAPI.simulate({ models: [rec.model], gpu: rec.gpu, cfg: cfg });
  }

  async function modeResult(rec, mode) {
    if (rec.comparable === 'decode') {
      var d = await call(rec, baseCfg(rec, 'decode', mode));
      var ph = d.results[0].decode;
      var predicted = ph.time * 1e3;             /* ms per decode step (batch B) */
      var bytes = ph.bytes;                       /* system-wide DRAM bytes/step */
      return {
        predicted: predicted,
        signedErrorPct: ((predicted - rec.measured) / rec.measured) * 100,
        throughputTokPerSec: ph.throughput,
        bytesPerStep: bytes,
        bound: ph.bound,
        coreVersion: d.coreVersion,
      };
    }
    if (rec.comparable === 'composite_serial') {
      var cfgP = baseCfg(rec, 'prefill', mode);
      var cfgD = baseCfg(rec, 'decode', mode);
      var [pre, dec] = await Promise.all([call(rec, cfgP), call(rec, cfgD)]);
      var p = pre.results[0].prefill;
      var dph = dec.results[0].decode;
      var waves = Math.ceil(rec.N / rec.B);
      var total = waves * (p.time + rec.O * dph.time);
      return {
        predicted: total,
        signedErrorPct: ((total - rec.measured) / rec.measured) * 100,
        totalDurationS: total,
        prefillTimeS: p.time,
        decodeStepMs: dph.time * 1e3,
        waves: waves,
        coreVersion: pre.coreVersion,
      };
    }
    return null;
  }

  async function evalRow(rec) {
    var gpu = catalog.gpus.find(function (g) { return g.id === rec.gpu; });
    var model = catalog.models.find(function (m) { return m.id === rec.model; });
    var row = {
      rec: rec,
      model: model,
      gpu: gpu,
      modes: { spec: null, realistic: null },
    };
    for (var mode of ['spec', 'realistic']) {
      row.modes[mode] = await modeResult(rec, mode);
      if (row.modes[mode] && row.modes[mode].coreVersion) coreVersion = row.modes[mode].coreVersion;
    }
    if (rec.comparable === 'decode' && row.modes.spec) {
      var bw = gpu.bandwidth_GBps * 1e9 * rec.gpus;
      row.impliedBwEff = row.modes.spec.bytesPerStep / (rec.measured / 1e3) / bw;
    }
    return row;
  }

  function sourceCell(row) {
    var rec = row.rec;
    return '<a href="' + escapeHtml(rec.url) + '" target="_blank" rel="noopener" class="ref-link">' +
      escapeHtml(rec.source.split(' (')[0]) + '</a>' +
      '<div class="sub">' + escapeHtml(rec.published || '') + ' · ' +
      escapeHtml((rec.engine || '') + (rec.engineVersion ? ' ' + rec.engineVersion : '')) + '</div>';
  }

  function modelCell(row) {
    return '<b>' + escapeHtml(row.model.name) + '</b><div class="sub">' +
      escapeHtml(row.gpu.name) + (row.rec.gpus > 1 ? ' ×' + row.rec.gpus : '') +
      ' · B=' + row.rec.B + ' S=' + num(row.rec.S, 3) + '</div>';
  }

  function precCell(rec) {
    return escapeHtml(rec.precision.toUpperCase()) +
      '<div class="sub">KV ' + escapeHtml(rec.kvPrecision.toUpperCase()) + '</div>';
  }

  function modeCells(m, metricScale) {
    if (!m) return '<td>–</td><td>–</td>';
    return '<td>' + num(m.predicted * metricScale, 4) + '</td>' +
      '<td>' + errPct(m.signedErrorPct) + '</td>';
  }

  function renderDecode(rows) {
    var body = rows.filter(function (r) { return r.rec.comparable === 'decode'; })
      .map(function (r) {
        var m = r.rec.measured;
        return '<tr>' +
          '<td>' + sourceCell(r) + '</td>' +
          '<td>' + modelCell(r) + '</td>' +
          '<td>' + precCell(r.rec) + '</td>' +
          '<td>' + num(m, 4) + ' ms<div class="sub">' +
            (r.rec.measuredThroughputTokPerSec ? num(r.rec.measuredThroughputTokPerSec, 4) + ' tok/s' : '') +
          '</div></td>' +
          modeCells(r.modes.spec, 1) +
          modeCells(r.modes.realistic, 1) +
          '<td>' + (r.impliedBwEff != null ? Math.round(r.impliedBwEff * 100) + '%' : '–') + '</td>' +
          '<td title="' + escapeHtml(r.rec.notes || '') + '" class="sub">' +
            escapeHtml((r.rec.notes || '').slice(0, 90)) + '…</td>' +
          '</tr>';
      }).join('');
    $('decodeTable').innerHTML =
      '<table><thead><tr>' +
      '<th>Source</th><th>Model · GPU</th><th>Precision</th><th>Measured ITL</th>' +
      '<th>Spec pred</th><th>Spec err</th><th>Realistic pred</th><th>Realistic err</th>' +
      '<th>Implied HBM</th><th>Notes</th>' +
      '</tr></thead><tbody>' + body + '</tbody></table>';
  }

  function renderComposite(rows) {
    var body = rows.filter(function (r) { return r.rec.comparable === 'composite_serial'; })
      .map(function (r) {
        var m = r.rec;
        var s = r.modes.spec, z = r.modes.realistic;
        return '<tr>' +
          '<td>' + sourceCell(r) + '</td>' +
          '<td>' + modelCell(r) + '</td>' +
          '<td>' + num(m.measured, 4) + ' s<div class="sub">' +
            (m.measuredOutputTokPerSec ? num(m.measuredOutputTokPerSec, 4) + ' out tok/s' : '') + '</div></td>' +
          '<td>' + num(s ? s.predicted : null, 4) + ' s<div class="sub">' +
            (s ? s.waves + ' waves · prefill ' + num(s.prefillTimeS, 3) + ' s · step ' + num(s.decodeStepMs, 3) + ' ms' : '') +
          '</div></td><td>' + (s ? errPct(s.signedErrorPct) : '–') + '</td>' +
          '<td>' + num(z ? z.predicted : null, 4) + ' s</td><td>' + (z ? errPct(z.signedErrorPct) : '–') + '</td>' +
          '<td title="' + escapeHtml(m.notes || '') + '" class="sub">' + escapeHtml((m.notes || '').slice(0, 110)) + '…</td>' +
          '</tr>';
      }).join('');
    $('compositeTable').innerHTML =
      '<table><thead><tr>' +
      '<th>Source</th><th>Model · GPU</th><th>Measured duration</th>' +
      '<th>Spec pred</th><th>Spec err</th><th>Realistic pred</th><th>Realistic err</th><th>Notes</th>' +
      '</tr></thead><tbody>' + body + '</tbody></table>';
  }

  function renderReference(rows) {
    var body = rows.filter(function (r) { return r.rec.comparable === 'reference'; })
      .map(function (r) {
        var m = r.rec;
        return '<tr>' +
          '<td>' + sourceCell(r) + '</td>' +
          '<td>' + modelCell(r) + '</td>' +
          '<td>' + precCell(m) + '</td>' +
          '<td>' + num(m.measured, 4) + '<div class="sub">' + escapeHtml(m.metric) + '</div></td>' +
          '<td title="' + escapeHtml(m.notes || '') + '" class="sub">' + escapeHtml((m.notes || '').slice(0, 130)) + '…</td>' +
          '</tr>';
      }).join('');
    $('referenceTable').innerHTML =
      '<table><thead><tr><th>Source</th><th>Model · GPU</th><th>Precision</th>' +
      '<th>Measured</th><th>Why it needs L2/L3</th></tr></thead><tbody>' + body + '</tbody></table>';
  }

  function renderSummary(rows, realistic) {
    var dec = rows.filter(function (r) { return r.rec.comparable === 'decode'; });
    function mape(mode) {
      var es = dec.map(function (r) { return r.modes[mode] && r.modes[mode].signedErrorPct; })
        .filter(function (e) { return e != null && isFinite(e); });
      if (!es.length) return null;
      return es.reduce(function (a, b) { return a + Math.abs(b); }, 0) / es.length;
    }
    var comp = rows.find(function (r) { return r.rec.comparable === 'composite_serial'; });
    var sMape = mape('spec');
    var rMape = mape('realistic');
    $('summaryCards').innerHTML =
      '<div class="card"><div class="c-label">Comparable decode rows</div><div class="c-value">' + dec.length +
      '</div><div class="c-note">single-stream ITL records</div></div>' +
      '<div class="card"><div class="c-label">MAPE · spec peaks</div><div class="c-value">' +
      (sMape == null ? '–' : sMape.toFixed(1) + '%') +
      '</div><div class="c-note">upper-bound model, 100% of datasheet</div></div>' +
      '<div class="card"><div class="c-label">MAPE · realistic</div><div class="c-value">' +
      (rMape == null ? '–' : rMape.toFixed(1) + '%') +
      '</div><div class="c-note">compute ' + Math.round(realistic.computeEfficiency * 100) +
      '% · DRAM ' + Math.round(realistic.bandwidthEfficiency * 100) + '%</div></div>' +
      '<div class="card"><div class="c-label">End-to-end error</div><div class="c-value">' +
      (comp && comp.modes.realistic ? errPct(comp.modes.realistic.signedErrorPct) : '–') +
      '</div><div class="c-note">realistic · serial-wave approximation</div></div>';
  }

  async function init() {
    setStatus('Loading catalog…');
    try {
      catalog = await SimAPI.getCatalog();
      var realistic = catalog.realistic || { computeEfficiency: 0.70, bandwidthEfficiency: 0.75 };
      $('effCompute').textContent = Math.round(realistic.computeEfficiency * 100);
      $('effBandwidth').textContent = Math.round(realistic.bandwidthEfficiency * 100);
      var records = catalog.validationBenchmarks || [];
      var rows = [];
      for (var i = 0; i < records.length; i++) {
        var rec = records[i];
        if (rec.comparable === 'reference') continue;
        setStatus('Running row ' + (i + 1) + '/' + records.length + ' (' + rec.id + ')…');
        rows.push(await evalRow(rec));
      }
      renderDecode(rows);
      renderComposite(rows);
      var refs = records.filter(function (r) { return r.comparable === 'reference'; })
        .map(function (rec) {
          return {
            rec: rec,
            model: catalog.models.find(function (m) { return m.id === rec.model; }),
            gpu: catalog.gpus.find(function (g) { return g.id === rec.gpu; }),
          };
        });
      renderReference(refs);
      renderSummary(rows, realistic);
      $('coreTag').textContent = 'C++ core ' + coreVersion + ' · ' + SimAPI.mode;
      $('coreTag').className = 'tag ok';
      setStatus('L1 validation · ' + SimAPI.mode + (SimAPI.mode === 'wasm' ? ' (WebAssembly core)' : ' (local C++ core)'));
    } catch (err) {
      setStatus('Validation failed: ' + err.message, 'err');
      ['decodeTable', 'compositeTable', 'referenceTable'].forEach(function (id) {
        $(id).innerHTML = '<div class="empty">' + escapeHtml(String(err.message || err)) + '</div>';
      });
    }
  }

  init();
})();
