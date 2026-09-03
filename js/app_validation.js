(function () {
  'use strict';

  function $(id) { return document.getElementById(id); }

  var catalog = null;
  var coreVersion = '—';
  var l1 = null;              /* selection persisted by index.html */

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
  function fmtPct(v) { return Math.round(v * 100) + '%'; }
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
  function modelById(id) {
    return catalog.models.find(function (m) { return m.id === id; });
  }
  function gpuById(id) {
    return catalog.gpus.find(function (g) { return g.id === id; });
  }
  function precisionLabel(id) {
    var p = catalog.precisions.find(function (x) { return x.id === id; });
    return p ? p.label : id;
  }

  /* ---- operating-point pairs ------------------------------------------- */
  function presetPair() {
    var r = (catalog && catalog.realistic) || {};
    return [
      r.computeEfficiency != null ? r.computeEfficiency : 0.70,
      r.bandwidthEfficiency != null ? r.bandwidthEfficiency : 0.75,
    ];
  }
  function selPair() {
    if (l1 && l1.computeScale != null && l1.bandwidthScale != null) {
      return [l1.computeScale, l1.bandwidthScale];
    }
    return presetPair();
  }
  function pairLabel(p) {
    if (p[0] === 1 && p[1] === 1) return 'spec (100%)';
    return 'compute ' + fmtPct(p[0]) + ' · DRAM ' + fmtPct(p[1]);
  }

  function baseCfg(rec, phase, pair) {
    return {
      phase: phase,
      B: rec.B,
      S: rec.S,
      gpus: rec.gpus,
      precision: rec.precision,
      kvPrecision: rec.kvPrecision,
      computeScale: pair[0],
      bandwidthScale: pair[1],
    };
  }
  function call(rec, cfg) {
    return SimAPI.simulate({ models: [rec.model], gpu: rec.gpu, cfg: cfg });
  }

  async function modeResult(rec, pair) {
    if (rec.comparable === 'decode') {
      var d = await call(rec, baseCfg(rec, 'decode', pair));
      var ph = d.results[0].decode;
      var predicted = ph.time * 1e3;               /* ms per decode step */
      return {
        predicted: predicted,
        signedErrorPct: ((predicted - rec.measured) / rec.measured) * 100,
        throughputTokPerSec: ph.throughput,
        bytesPerStep: ph.bytes,
        bound: ph.bound,
        coreVersion: d.coreVersion,
      };
    }
    if (rec.comparable === 'composite_serial') {
      var [pre, dec] = await Promise.all([
        call(rec, baseCfg(rec, 'prefill', pair)),
        call(rec, baseCfg(rec, 'decode', pair)),
      ]);
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
    var row = {
      rec: rec,
      model: modelById(rec.model),
      gpu: gpuById(rec.gpu),
      modes: { spec: null, sel: null },
    };
    var spec = await modeResult(rec, [1, 1]);
    var sel = selPair();
    var same = Math.abs(sel[0] - 1) < 1e-9 && Math.abs(sel[1] - 1) < 1e-9;
    row.modes.spec = spec;
    if (!same) row.modes.sel = await modeResult(rec, sel);
    if (spec && spec.coreVersion) coreVersion = spec.coreVersion;
    if (rec.comparable === 'decode' && spec) {
      var bw = row.gpu.bandwidth_GBps * 1e9 * rec.gpus;
      row.impliedBwEff = spec.bytesPerStep / (rec.measured / 1e3) / bw;
    }
    return row;
  }

  /* ---- coverage / missing-data analysis -------------------------------- */
  function coveredRecords() {
    if (!l1 || !l1.models.length) return [];
    return (catalog.validationBenchmarks || []).filter(function (r) {
      return l1.models.indexOf(r.model) >= 0 && r.gpu === l1.gpuId;
    });
  }
  function buildCoverage() {
    var el = $('coverageNotes');
    if (!l1) {
      el.innerHTML = '<p class="side-note">No L1 selection saved yet. Open the ' +
        '<a href="index.html" class="ref-link">L1 page</a>, toggle models / GPU / precision / ' +
        'operating point, and come back — below shows the full measured set meanwhile.</p>';
      return;
    }
    var notes = [];
    var all = catalog.validationBenchmarks || [];
    var covered = [];
    var gpu = gpuById(l1.gpuId);
    var gpuName = gpu ? gpu.name : l1.gpuId;
    l1.models.forEach(function (mid) {
      var m = modelById(mid);
      var name = m ? m.name : mid;
      var hits = all.filter(function (r) { return r.model === mid && r.gpu === l1.gpuId; });
      if (!hits.length) {
        notes.push({ kind: 'warn', text: 'No measured validation data for <b>' + name +
          '</b> on <b>' + gpuName + '</b>. Realistic assumptions for this combination cannot be ' +
          'checked yet — add a benchmark record in simulator/validation_data.py.' });
        return;
      }
      covered.push(name + ' · ' + gpuName);
      var precHits = hits.filter(function (r) { return r.precision === l1.precision; });
      if (!precHits.length) {
        var precs = hits.map(function (r) { return r.precision.toUpperCase(); })
          .filter(function (v, i, a) { return a.indexOf(v) === i; }).join(' / ');
        notes.push({ kind: 'warn', text: 'L1 shows <b>' + precisionLabel(l1.precision) +
          '</b> weights, but measured records for <b>' + name + '</b> on <b>' + gpuName +
          '</b> used <b>' + precs + '</b>. Rows are shown for context with the dtype mismatch ' +
          'flagged; exact realistic comparison at your precision is not available.' });
      }
    });
    if (l1.phase === 'prefill') {
      notes.push({ kind: 'info', text: 'Measured records cover decode-only and end-to-end runs, ' +
        'not a prefill-only phase; prefill contributes inside the composite row.' });
    }
    if (l1.models.length && covered.length) {
      notes.unshift({ kind: 'ok', text: 'Measured coverage matches the selection: ' +
        covered.join('; ') + '.' });
    }
    if (!l1.models.length) {
      notes.push({ kind: 'warn', text: 'No models are selected on the L1 page. Select at least ' +
        'one model there to filter the validation rows.' });
    }
    el.innerHTML = notes.map(function (n) {
      return '<p class="side-note"><span class="tag ' + (n.kind === 'warn' ? 'oom' : n.kind === 'ok' ? 'ok' : 'ref') +
        '">' + (n.kind === 'warn' ? 'missing' : n.kind === 'ok' ? 'coverage' : 'note') +
        '</span> ' + n.text + '</p>';
    }).join('');
  }

  function renderSyncChips() {
    var el = $('syncChips');
    if (!l1) {
      el.innerHTML = 'No selection stored — this page shows every measured record until you ' +
        'toggle a configuration on the <a href="index.html" class="ref-link">L1 page</a>.';
      return;
    }
    var gpu = gpuById(l1.gpuId);
    var models = l1.models.map(function (mid) {
      var m = modelById(mid);
      return m ? m.name : mid;
    });
    var shown = models.length > 4 ? models.slice(0, 4).join(', ') + ' +' + (models.length - 4) : models.join(', ');
    var pair = selPair();
    el.innerHTML =
      'Models: <b>' + escapeHtml(shown) + '</b> &nbsp;·&nbsp; GPU: <b>' +
      escapeHtml(gpu ? gpu.name : l1.gpuId) + '</b> &nbsp;·&nbsp; weights: <b>' +
      escapeHtml(precisionLabel(l1.precision).split(' (')[0]) + '</b> &nbsp;·&nbsp; KV: <b>' +
      l1.kvPrecision.toUpperCase() + '</b> &nbsp;·&nbsp; phase: <b>' + l1.phase +
      '</b> &nbsp;·&nbsp; B=' + l1.B + ' S=' + num(l1.S, 3) + ' ×' + l1.gpus +
      ' &nbsp;·&nbsp; operating point: <b>' + l1.mode + ' (' + pairLabel(pair) + ')</b>.' +
      ' <span class="tag ref">prediction column uses the B/S/G of each measurement</span>';
  }

  /* ---- rendering -------------------------------------------------------- */
  function sourceCell(row) {
    var rec = row.rec;
    return '<a href="' + escapeHtml(rec.url) + '" target="_blank" rel="noopener" class="ref-link">' +
      escapeHtml(rec.source.split(' (')[0]) + '</a>' +
      '<div class="sub">' + escapeHtml(rec.published || '') + ' · ' +
      escapeHtml((rec.engine || '') + (rec.engineVersion ? ' ' + rec.engineVersion : '')) + '</div>';
  }
  function modelCell(row) {
    var cfgParts = [];
    if (row.rec.B) cfgParts.push('B=' + row.rec.B);
    if (row.rec.S) cfgParts.push('S=' + num(row.rec.S, 3));
    return '<b>' + escapeHtml(row.model.name) + '</b><div class="sub">' +
      escapeHtml(row.gpu.name) + (row.rec.gpus > 1 ? ' ×' + row.rec.gpus : '') +
      (cfgParts.length ? ' · ' + cfgParts.join(' ') : '') + '</div>';
  }
  function mismatchTags(rec) {
    if (!l1) return '–';
    if (rec.comparable === 'reference') {
      return '<span class="tag ref">scheduler-level · concurrency unpublished</span>';
    }
    var tags = [];
    if (rec.precision !== l1.precision) {
      tags.push('measured ' + rec.precision.toUpperCase() + ' weights · L1 ' + l1.precision.toUpperCase());
    }
    if (rec.kvPrecision !== l1.kvPrecision) {
      tags.push('measured KV ' + rec.kvPrecision.toUpperCase() + ' · L1 ' + l1.kvPrecision.toUpperCase());
    }
    if (rec.gpus && l1.gpus && rec.gpus !== l1.gpus) {
      tags.push('measured ×' + rec.gpus + ' · L1 ×' + l1.gpus);
    }
    return tags.length ? tags.map(escapeHtml).join('<br>') : '<span class="tag ok">config match</span>';
  }
  function modeCells(m, scale) {
    if (!m) return '<td>–</td><td>–</td>';
    return '<td>' + num(m.predicted * scale, 4) + '</td><td>' + errPct(m.signedErrorPct) + '</td>';
  }
  function pairColumns() {
    var p = selPair();
    var same = Math.abs(p[0] - 1) < 1e-9 && Math.abs(p[1] - 1) < 1e-9;
    var label = (l1 && l1.models.length) ? 'L1 sel. pred' : 'Realistic preset pred';
    return {
      headers: '<th>Spec pred</th><th>Spec err</th>' +
        (same ? '' : '<th>' + label + '</th><th>err</th>'),
      same: same,
    };
  }

  function renderDecode(rows, cols) {
    var body = rows.map(function (r) {
      var rec = r.rec;
      return '<tr>' +
        '<td>' + sourceCell(r) + '</td>' +
        '<td>' + modelCell(r) + '</td>' +
        '<td>' + rec.precision.toUpperCase() + '<div class="sub">KV ' + rec.kvPrecision.toUpperCase() + '</div></td>' +
        '<td>' + mismatchTags(rec) + '</td>' +
        '<td>' + num(rec.measured, 4) + ' ms<div class="sub">' +
          (rec.measuredThroughputTokPerSec ? num(rec.measuredThroughputTokPerSec, 4) + ' tok/s' : '') +
        '</div></td>' +
        modeCells(r.modes.spec, 1) +
        (!cols.same ? modeCells(r.modes.sel, 1) : '') +
        '<td>' + (r.impliedBwEff != null ? fmtPct(r.impliedBwEff) : '–') + '</td>' +
        '<td title="' + escapeHtml(rec.notes || '') + '" class="sub">' +
          escapeHtml((rec.notes || '').slice(0, 95)) + '…</td>' +
        '</tr>';
    }).join('');
    $('decodeTable').innerHTML =
      '<table><thead><tr>' +
      '<th>Source</th><th>Model · GPU</th><th>Measured dtype</th><th>vs L1 selection</th>' +
      '<th>Measured ITL</th>' + cols.headers +
      '<th>Implied HBM</th><th>Notes</th>' +
      '</tr></thead><tbody>' + (body || '<tr><td colspan="11" class="sub">No matching decode records.</td></tr>') +
      '</tbody></table>';
  }

  function renderComposite(rows, cols) {
    var body = rows.map(function (r) {
      var rec = r.rec;
      var s = r.modes.spec, z = r.modes.sel;
      return '<tr>' +
        '<td>' + sourceCell(r) + '</td>' +
        '<td>' + modelCell(r) + '</td>' +
        '<td>' + mismatchTags(rec) + '</td>' +
        '<td>' + num(rec.measured, 4) + ' s<div class="sub">' +
          (rec.measuredOutputTokPerSec ? num(rec.measuredOutputTokPerSec, 4) + ' out tok/s' : '') + '</div></td>' +
        '<td>' + num(s ? s.predicted : null, 4) + ' s</td><td>' + (s ? errPct(s.signedErrorPct) : '–') + '</td>' +
        (!cols.same
          ? '<td>' + num(z ? z.predicted : null, 4) + ' s</td><td>' + (z ? errPct(z.signedErrorPct) : '–') + '</td>'
          : '') +
        '<td title="' + escapeHtml(rec.notes || '') + '" class="sub">' +
          escapeHtml((rec.notes || '').slice(0, 110)) + '…</td>' +
        '</tr>';
    }).join('');
    $('compositeTable').innerHTML =
      '<table><thead><tr>' +
      '<th>Source</th><th>Model · GPU</th><th>vs L1 selection</th><th>Measured duration</th>' +
      cols.headers +
      '<th>Notes</th></tr></thead><tbody>' +
      (body || '<tr><td colspan="8" class="sub">No matching composite records.</td></tr>') +
      '</tbody></table>';
  }

  function renderReference(rows) {
    var body = rows.map(function (r) {
      var rec = r.rec;
      return '<tr>' +
        '<td>' + sourceCell(r) + '</td>' +
        '<td>' + modelCell(r) + '</td>' +
        '<td>' + mismatchTags(rec) + '</td>' +
        '<td>' + num(rec.measured, 4) + '<div class="sub">' + escapeHtml(rec.metric) + '</div></td>' +
        '<td title="' + escapeHtml(rec.notes || '') + '" class="sub">' +
          escapeHtml((rec.notes || '').slice(0, 130)) + '…</td>' +
        '</tr>';
    }).join('');
    $('referenceTable').innerHTML =
      '<table><thead><tr><th>Source</th><th>Model · GPU</th><th>vs L1 selection</th>' +
      '<th>Measured</th><th>Why it needs L2/L3</th></tr></thead><tbody>' +
      (body || '<tr><td colspan="5" class="sub">No matching reference records.</td></tr>') +
      '</tbody></table>';
  }

  function renderSummary(rows, cols) {
    var dec = rows.filter(function (r) { return r.rec.comparable === 'decode'; });
    function mape(mode) {
      var es = dec.map(function (r) { return r.modes[mode] && r.modes[mode].signedErrorPct; })
        .filter(function (e) { return e != null && isFinite(e); });
      if (!es.length) return null;
      return es.reduce(function (a, b) { return a + Math.abs(b); }, 0) / es.length;
    }
    var comp = rows.find(function (r) { return r.rec.comparable === 'composite_serial'; });
    var sMape = mape('spec');
    var selMape = mape('sel');
    var p = selPair();
    var hasSel = !!(l1 && l1.models.length);
    var scopeNote = hasSel ? 'filtered to the L1 selection' : 'full measured set';
    $('summaryCards').innerHTML =
      '<div class="card"><div class="c-label">Comparable decode rows</div><div class="c-value">' + dec.length +
      '</div><div class="c-note">' + scopeNote + '</div></div>' +
      '<div class="card"><div class="c-label">MAPE · spec</div><div class="c-value">' +
      (sMape == null ? '–' : sMape.toFixed(1) + '%') +
      '</div><div class="c-note">100% of datasheet peaks</div></div>' +
      '<div class="card"><div class="c-label">MAPE · ' + (hasSel ? 'L1 selection' : 'realistic preset') +
      '</div><div class="c-value">' +
      (selMape == null ? (cols.same ? '= spec' : '–') : selMape.toFixed(1) + '%') +
      '</div><div class="c-note">' + pairLabel(p) + '</div></div>' +
      '<div class="card"><div class="c-label">End-to-end error</div><div class="c-value">' +
      (comp && comp.modes.sel ? errPct(comp.modes.sel.signedErrorPct) :
        (comp && comp.modes.spec ? errPct(comp.modes.spec.signedErrorPct) : '–')) +
      '</div><div class="c-note">' + (hasSel ? 'L1 selection' : 'realistic preset') +
      ' · serial-wave approx</div></div>';
  }

  async function init() {
    setStatus('Loading catalog…');
    l1 = window.L1State ? L1State.read() : null;
    if (window.L1State) L1State.onChange(function () { location.reload(); });
    try {
      catalog = await SimAPI.getCatalog();
      var realistic = catalog.realistic || { computeEfficiency: 0.70, bandwidthEfficiency: 0.75 };
      var p = selPair();
      $('effCompute').textContent = fmtPct(p[0]);
      $('effBandwidth').textContent = fmtPct(p[1]);
      renderSyncChips();
      buildCoverage();

      var records = catalog.validationBenchmarks || [];
      var matched = coveredRecords();
      var pool = (l1 && l1.models.length) ? matched : records;
      var comparable = pool.filter(function (r) { return r.comparable !== 'reference'; });
      var cols = pairColumns();
      var rows = [];
      for (var i = 0; i < comparable.length; i++) {
        setStatus('Running row ' + (i + 1) + '/' + comparable.length + ' (' + comparable[i].id + ')…');
        rows.push(await evalRow(comparable[i]));
      }
      renderDecode(rows.filter(function (r) { return r.rec.comparable === 'decode'; }), cols);
      renderComposite(rows.filter(function (r) { return r.rec.comparable === 'composite_serial'; }), cols);
      var refs = pool.filter(function (r) { return r.comparable === 'reference'; })
        .map(function (rec) {
          return { rec: rec, model: modelById(rec.model), gpu: gpuById(rec.gpu) };
        });
      renderReference(refs);
      renderSummary(rows, cols);
      $('coreTag').textContent = 'C++ core ' + coreVersion + ' · ' + SimAPI.mode;
      $('coreTag').className = 'tag ok';
      setStatus('L1 validation · ' + SimAPI.mode +
        (l1 ? ' · synced to L1 selection' : ' · no L1 selection stored'));
    } catch (err) {
      setStatus('Validation failed: ' + err.message, 'err');
      ['decodeTable', 'compositeTable', 'referenceTable'].forEach(function (id) {
        $(id).innerHTML = '<div class="empty">' + escapeHtml(String(err.message || err)) + '</div>';
      });
    }
  }

  init();
})();
