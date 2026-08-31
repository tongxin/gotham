(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.Charts = api;
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const NS = 'http://www.w3.org/2000/svg';

  function el(tag, attrs, parent) {
    const n = document.createElementNS(NS, tag);
    if (attrs) {
      for (const k in attrs) {
        if (attrs[k] != null) n.setAttribute(k, attrs[k]);
      }
    }
    if (parent) parent.appendChild(n);
    return n;
  }

  function fmtSI(v) {
    if (!isFinite(v) || v <= 0) return '0';
    const units = [['P', 1e15], ['T', 1e12], ['G', 1e9], ['M', 1e6], ['k', 1e3]];
    for (const [u, m] of units) {
      if (v >= m) {
        const x = v / m;
        return (x >= 100 ? x.toFixed(0) : x.toFixed(1)) + u;
      }
    }
    return v.toFixed(1);
  }

  function fmtBytes(v) {
    if (!isFinite(v)) return '–';
    const units = [['TB', 1e12], ['GB', 1e9], ['MB', 1e6], ['KB', 1e3]];
    for (const [u, m] of units) {
      if (v >= m) return (v / m).toFixed(1) + ' ' + u;
    }
    return v.toFixed(0) + ' B';
  }

  function logTicks(min, max) {
    const out = [];
    let k = Math.floor(Math.log10(min));
    while (Math.pow(10, k) < max * 1.0001) {
      const base = Math.pow(10, k);
      for (const m of [1, 2, 5]) {
        const v = m * base;
        if (v >= min * 0.9999 && v <= max * 1.0001) out.push(v);
      }
      k += 1;
    }
    return out;
  }

  class RooflineChart {
    constructor(container, tipEl) {
      this.container = container;
      this.tipEl = tipEl;
    }

    _showTip(ev, data) {
      const rect = this.container.getBoundingClientRect();
      const rows = (data.rows || []).map(r =>
        '<div class="tip-row"><span>' + r.k + '</span><b>' + r.v + '</b></div>'
      ).join('');
      this.tipEl.innerHTML =
        '<div class="tip-title" style="color:' + data.color + '">' + data.title + '</div>' + rows;
      this.tipEl.classList.remove('hidden');
      const left = ev.clientX - rect.left + 16;
      const top = ev.clientY - rect.top - 10;
      this.tipEl.style.left = (left + 240 > rect.width ? left - 260 : left) + 'px';
      this.tipEl.style.top = top + 'px';
    }

    _hideTip() {
      this.tipEl.classList.add('hidden');
    }

    update(opts) {
      this.container.innerHTML = '';
      const W = Math.max(this.container.clientWidth || 960, 520);
      const H = Math.max(this.container.clientHeight || 520, 440);
      const m = { l: 74, r: 28, t: 30, b: 54 };
      const pw = W - m.l - m.r;
      const ph = H - m.t - m.b;
      const svg = el('svg', { viewBox: '0 0 ' + W + ' ' + H, width: '100%', height: '100%' }, this.container);

      const defs = el('defs', {}, svg);
      const grad = el('linearGradient', { id: 'rlFill', x1: 0, y1: 0, x2: 0, y2: 1 }, defs);
      el('stop', { offset: '0%', 'stop-color': opts.primaryColor || '#7aa2f7', 'stop-opacity': '0.25' }, grad);
      el('stop', { offset: '100%', 'stop-color': opts.primaryColor || '#7aa2f7', 'stop-opacity': '0.02' }, grad);

      const xs = [];
      for (const p of opts.points) xs.push(p.x);
      for (const c of opts.ceilings || []) xs.push(c.peak / c.bandwidth);
      const sweeps = opts.sweeps || [];
      for (const sw of sweeps) for (const p of sw.points) xs.push(p.x);
      const xMin = Math.max(1e-3, Math.pow(10, Math.floor(Math.log10(Math.min.apply(null, xs) * 0.6))));
      const xMax = Math.pow(10, Math.ceil(Math.log10(Math.max.apply(null, xs) * 1.5)));

      const ys = [];
      for (const c of opts.ceilings || []) ys.push(c.peak);
      for (const p of opts.points) ys.push(p.y);
      for (const sw of sweeps) for (const p of sw.points) ys.push(p.y);
      const yMin = Math.pow(10, Math.floor(Math.log10(Math.max(1e9, Math.min.apply(null, ys) * 0.35))));
      const yMax = Math.pow(10, Math.ceil(Math.log10(Math.max.apply(null, ys) * 2.2)));

      const logX = v => Math.log10(v / xMin) / Math.log10(xMax / xMin);
      const logY = v => Math.log10(v / yMin) / Math.log10(yMax / yMin);
      const X = v => m.l + logX(v) * pw;
      const Y = v => m.t + ph - logY(v) * ph;

      for (const t of logTicks(xMin, xMax)) {
        el('line', { x1: X(t), x2: X(t), y1: m.t, y2: m.t + ph, stroke: '#212838', 'stroke-width': 1 }, svg);
      }
      for (const t of logTicks(yMin, yMax)) {
        el('line', { x1: m.l, x2: m.l + pw, y1: Y(t), y2: Y(t), stroke: '#212838', 'stroke-width': 1 }, svg);
      }

      for (const t of logTicks(xMin, xMax)) {
        const tx = el('text', { x: X(t), y: m.t + ph + 18, 'text-anchor': 'middle', class: 'axis' }, svg);
        tx.textContent = fmtSI(t);
      }
      for (const t of logTicks(yMin, yMax)) {
        const tx = el('text', { x: m.l - 8, y: Y(t) + 4, 'text-anchor': 'end', class: 'axis' }, svg);
        tx.textContent = fmtSI(t);
      }
      const xl = el('text', { x: m.l + pw / 2, y: H - 10, 'text-anchor': 'middle', class: 'axis-title' }, svg);
      xl.textContent = 'Operational intensity (FLOP/byte)';
      const yl = el('text', {
        x: 18, y: m.t + ph / 2, 'text-anchor': 'middle', class: 'axis-title',
        transform: 'rotate(-90 18 ' + (m.t + ph / 2) + ')',
      }, svg);
      yl.textContent = 'Performance (FLOP/s)';

      const primary = (opts.ceilings || []).find(c => c.primary) || (opts.ceilings || [])[0];
      for (const c of opts.ceilings || []) {
        const ridge = c.peak / c.bandwidth;
        const pts = [];
        pts.push(X(xMin) + ',' + Y(Math.max(c.bandwidth * xMin, yMin)));
        if (ridge <= xMax) {
          pts.push(X(ridge) + ',' + Y(c.peak));
          pts.push(X(xMax) + ',' + Y(c.peak));
        } else {
          pts.push(X(xMax) + ',' + Y(c.bandwidth * xMax));
        }
        el('polyline', {
          points: pts.join(' '), fill: 'none', stroke: c.color,
          'stroke-width': c.primary ? 2.5 : 1.2,
          'stroke-dasharray': c.primary ? null : '5 5',
          opacity: c.primary ? 1 : 0.5,
        }, svg);
        if (c.primary) {
          const fillPts = m.l + ',' + (m.t + ph) + ' ' + pts.join(' ') + ' ' + (m.l + pw) + ',' + (m.t + ph);
          el('polygon', { points: fillPts, fill: 'url(#rlFill)', stroke: 'none' }, svg);
          const rx = X(Math.min(Math.max(ridge, xMin), xMax));
          el('line', {
            x1: rx, x2: rx, y1: m.t, y2: m.t + ph,
            stroke: c.color, 'stroke-dasharray': '4 4', 'stroke-width': 1, opacity: 0.7,
          }, svg);
          const lb = el('text', { x: rx, y: m.t + 14, 'text-anchor': 'middle', class: 'axis', fill: c.color }, svg);
          lb.textContent = 'ridge ' + fmtSI(ridge) + ' FLOP/B';
        }
      }

      for (const sw of sweeps) {
        const line = sw.points.map(p => X(p.x) + ',' + Y(p.y)).join(' ');
        el('polyline', {
          points: line, fill: 'none', stroke: sw.color, 'stroke-width': 2,
          'stroke-dasharray': '6 3', opacity: 0.9,
        }, svg);
        sw.points.forEach((p, i) => {
          if (i % 8 === 0 || i === sw.points.length - 1) {
            el('circle', { cx: X(p.x), cy: Y(p.y), r: 2.5, fill: sw.color, opacity: 0.8 }, svg);
          }
        });
      }

      for (const p of opts.points) {
        const g = el('g', { class: 'pt', cursor: 'pointer' }, svg);
        const cx = X(p.x);
        const cy = Y(p.y);
        if (p.shape === 'square') {
          el('rect', {
            x: cx - 6, y: cy - 6, width: 12, height: 12, rx: 2.5,
            fill: p.color, stroke: '#0f1117', 'stroke-width': 2,
          }, g);
        } else {
          el('circle', { cx: cx, cy: cy, r: 7, fill: p.color, stroke: '#0f1117', 'stroke-width': 2 }, g);
        }
        g.addEventListener('mousemove', ev => this._showTip(ev, p));
        g.addEventListener('mouseleave', () => this._hideTip());
      }
    }
  }

  function memoryChart(container, rows, capacity) {
    container.innerHTML = '';
    const W = Math.max(container.clientWidth || 900, 520);
    const H = rows.length * 46 + 62;
    const svg = el('svg', { viewBox: '0 0 ' + W + ' ' + H, width: '100%', height: '100%' }, container);
    const labelW = 220;
    const x0 = labelW + 16;
    const x1 = W - 30;
    const maxVal = Math.max(capacity, Math.max.apply(null, rows.map(r => r.total))) * 1.08;
    const S = v => x0 + (v / maxVal) * (x1 - x0);

    const segs = [
      ['Weights', '#7aa2f7'],
      ['KV cache', '#7dcfff'],
      ['Activations', '#e0af68'],
      ['Capacity', '#f7768e'],
    ];
    const legendItems = segs.map(s => '<span class="lg"><span class="sw" style="background:' + s[1] + '"></span>' + s[0] + '</span>').join('');
    svg.setAttribute('data-legend', legendItems);

    rows.forEach((r, i) => {
      const y = 46 + i * 46;
      const tx = el('text', { x: labelW - 10, y: y + 15, 'text-anchor': 'end', fill: '#d7dce8', 'font-size': 12.5, 'font-weight': 600 }, svg);
      tx.textContent = r.label;
      let x = x0;
      const draw = (val, color) => {
        if (val <= 0) return;
        const w = Math.max(S(x + val) - x, 1);
        el('rect', { x: x, y: y + 2, width: w, height: 24, rx: 3, fill: color, opacity: 0.92 }, svg);
        x += val;
      };
      draw(r.weights, '#7aa2f7');
      draw(r.kv, '#7dcfff');
      draw(r.act, '#e0af68');
      const capX = S(capacity);
      el('line', { x1: capX, x2: capX, y1: y - 4, y2: y + 32, stroke: '#f7768e', 'stroke-width': 1.5, 'stroke-dasharray': '4 3' }, svg);
      const total = el('text', { x: x1 - 8, y: y + 20, 'text-anchor': 'end', 'font-family': 'SFMono-Regular, Menlo, monospace', 'font-size': 11.5 }, svg);
      total.textContent = fmtBytes(r.total) + ' / ' + fmtBytes(capacity);
      total.setAttribute('fill', r.total > capacity ? '#f7768e' : '#8b94a8');
      if (r.total > capacity) {
        const tag = el('text', { x: capX + 8, y: y - 8, fill: '#f7768e', 'font-size': 10.5, 'font-weight': 700 }, svg);
        tag.textContent = 'EXCEEDS CAPACITY';
      }
    });
    return legendItems;
  }

  function throughputChart(container, rows) {
    container.innerHTML = '';
    if (!rows.length) return '';
    const W = Math.max(container.clientWidth || 900, 520);
    const barsPerRow = Math.max.apply(null, rows.map(r => r.bars.length));
    const H = rows.length * (26 * barsPerRow + 14) + 44;
    const svg = el('svg', { viewBox: '0 0 ' + W + ' ' + H, width: '100%', height: '100%' }, container);
    const labelW = 220;
    const x0 = labelW + 16;
    const x1 = W - 34;
    const maxV = Math.max.apply(null, rows.flatMap(r => r.bars.map(b => b.value)));

    rows.forEach((r, i) => {
      const yTop = 40 + i * (26 * barsPerRow + 14);
      const tx = el('text', { x: labelW - 10, y: yTop + 15, 'text-anchor': 'end', fill: '#d7dce8', 'font-size': 12.5, 'font-weight': 600 }, svg);
      tx.textContent = r.label;
      r.bars.forEach((b, j) => {
        const y = yTop + j * 26;
        el('rect', { x: x0, y: y, width: x1 - x0, height: 16, rx: 3, fill: '#1b2130' }, svg);
        const w = (b.value / maxV) * (x1 - x0);
        el('rect', { x: x0, y: y, width: Math.max(w, 1), height: 16, rx: 3, fill: b.color, opacity: 0.9 }, svg);
        const lb = el('text', { x: x0 + w + 8, y: y + 12, fill: '#d7dce8', 'font-family': 'SFMono-Regular, Menlo, monospace', 'font-size': 11 }, svg);
        lb.textContent = b.label + ': ' + (b.value >= 1e4 ? fmtSI(b.value) : b.value.toFixed(1)) + ' tok/s';
      });
    });
    const legendItems = rows.map(r =>
      '<span class="lg"><span class="sw" style="background:' + r.color + '"></span>' + r.label + '</span>'
    ).join('');
    return legendItems;
  }

  return {
    RooflineChart,
    memoryChart,
    throughputChart,
    fmtSI,
    fmtBytes,
  };
});
