#!/usr/bin/env python3
"""Live web demo for real-camera calibration guidance with ghost-board overlays.

This app:
1. Opens a real camera.
2. Lets the user collect 5 seed images with a visible ChArUco or AprilGrid board.
3. Estimates seed intrinsics from those captures.
4. Generates the next-k best target board appearances from a discrete target bank.
5. Overlays a ghost board texture on the live feed so the user can align the real board.

The app uses only the Python standard library plus OpenCV/numpy already used elsewhere
in this repo. Open http://127.0.0.1:<port> in a browser after starting it.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import pathlib
import sys
import threading
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from calibration_board_utils import ensure_aprilgrid_texture, ensure_charuco_texture, detect_kalibr_aprilgrid
from detect_checkerboard_uv import detect_charuco
from OASIS import FIM as fim


PLOTLY_JS = get_plotlyjs()


HTML_PAGE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Live Calibration Demo</title>
  <script>__PLOTLY_JS__</script>
  <style>
    :root { --plot-height: 260px; --thumb-width: 240px; --thumb-height: 150px; --pose3d-height: 360px; }
    body { font-family: sans-serif; margin: 0; background: #111; color: #f2f2f2; }
    .wrap { display: grid; grid-template-columns: minmax(0, 1.15fr) minmax(0, 0.85fr); gap: 20px; padding: 20px; max-width: 100vw; box-sizing: border-box; }
    .panel { background: #1b1b1b; border-radius: 14px; padding: 16px; min-width: 0; overflow: hidden; max-height: calc(100vh - 40px); overflow-y: auto; }
    img { width: 100%; border-radius: 12px; background: #000; }
    .app-btn { margin: 6px 8px 6px 0; padding: 10px 14px; border: 0; border-radius: 10px; cursor: pointer; }
    .primary { background: #2e7d32; color: white; }
    .secondary { background: #1565c0; color: white; }
    .warn { background: #b71c1c; color: white; }
    pre { white-space: pre-wrap; word-break: break-word; background: #0f0f0f; padding: 12px; border-radius: 10px; }
    .strip-widget { background: #0f0f0f; border-radius: 10px; padding: 10px; min-width: 0; overflow: hidden; }
    .strip-controls { display: flex; justify-content: flex-end; gap: 8px; margin-bottom: 8px; }
    .strip-controls .app-btn { padding: 6px 10px; border-radius: 8px; margin: 0; }
    .thumb-strip { display: flex; gap: 10px; overflow-x: auto; overflow-y: hidden; padding-bottom: 8px; min-height: calc(var(--thumb-height) + 34px); width: 100%; box-sizing: border-box; }
    .thumb-card { min-width: var(--thumb-width); max-width: var(--thumb-width); }
    .thumb-card img { height: var(--thumb-height); object-fit: cover; }
    .thumb-card .label { font-size: 0.8rem; color: #ccc; margin-top: 4px; }
    .muted { color: #b5b5b5; }
    .metric { font-size: 1.1rem; margin: 6px 0; }
    .plots { display: grid; grid-template-columns: 1fr; gap: 14px; margin-top: 12px; }
    .plot-box { background: #0f0f0f; border-radius: 10px; padding: 10px; }
    .metric-plot { width: 100%; height: var(--plot-height); display: block; }
    .legend { font-size: 0.8rem; color: #cfcfcf; margin-top: 6px; display: flex; flex-wrap: wrap; gap: 8px 12px; align-items: center; }
    .filter-row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-top: 8px; }
    .filter-row select { background: #1f1f1f; color: #f2f2f2; border: 1px solid #555; border-radius: 8px; padding: 6px 8px; }
    .toggle-row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-top: 8px; }
    .toggle-chip { background: #1f1f1f; color: #ddd; border: 1px solid #555; border-radius: 999px; padding: 6px 10px; cursor: pointer; font-size: 0.82rem; }
    .toggle-chip.active { background: #1565c0; color: #fff; border-color: #1565c0; }
    .controls { display: flex; gap: 18px; align-items: center; flex-wrap: wrap; margin: 10px 0 14px 0; }
    .controls label { font-size: 0.9rem; color: #ddd; }
    .subsection { margin-top: 16px; }
    .swatch { display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 6px; vertical-align: middle; }
    .status-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; margin: 10px 0 12px 0; }
    .status-card { background: #121212; border: 1px solid #2a2a2a; border-radius: 10px; padding: 10px 12px; }
    .status-label { font-size: 0.76rem; color: #a8a8a8; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 6px; }
    .status-value { font-size: 1.02rem; color: #f2f2f2; font-weight: 600; }
    .status-wide { grid-column: 1 / -1; }
    .matrix-card { background: #101010; border: 1px solid #2a2a2a; border-radius: 10px; padding: 12px; margin-top: 10px; }
    .matrix-title { font-size: 0.82rem; color: #bdbdbd; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.04em; }
    .matrix-grid { display: grid; gap: 6px; font-family: monospace; }
    .matrix-row { display: grid; grid-template-columns: 18px 1fr 1fr 1fr 18px; gap: 10px; align-items: center; }
    .matrix-bracket { color: #90caf9; font-size: 1.3rem; line-height: 1; }
    .matrix-cell { background: #181818; border-radius: 8px; padding: 8px 10px; text-align: right; color: #f5f5f5; }
    .message-card { margin-top: 10px; }
    .clickable { cursor: zoom-in; }
    .modal { position: fixed; inset: 0; background: rgba(0,0,0,0.82); display: none; align-items: center; justify-content: center; z-index: 20; }
    .modal.open { display: flex; }
    .modal-inner { width: 96vw; height: 96vh; display: flex; flex-direction: column; align-items: center; justify-content: center; }
    .modal-toolbar { display: flex; gap: 12px; align-items: center; margin-bottom: 10px; color: #ddd; }
    .modal-toolbar a { color: #90caf9; text-decoration: none; }
    .modal-close { background: #b71c1c; color: white; }
    .modal-image-wrap { flex: 1; width: 100%; overflow: auto; display: flex; align-items: center; justify-content: center; }
    .modal-inner img { width: auto; max-width: none; max-height: none; object-fit: contain; border-radius: 12px; transform-origin: center center; }
    .modal-caption { margin-top: 8px; text-align: center; color: #ddd; }
    #poses3d-live { width: 100%; height: var(--pose3d-height); border-radius: 12px; overflow: hidden; }
    #poses3d-live .modebar { transform: scale(0.85); transform-origin: top right; }
    #poses3d-live .modebar-btn { margin: 0 !important; padding: 0 !important; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="panel">
      <h2>Live Camera</h2>
      <div class="controls">
        <label>Left gallery size
          <input id="thumb-size" type="range" min="180" max="360" step="10" value="240" oninput="setThumbSize(this.value)">
        </label>
      </div>
      <img id="live" src="/frame.jpg" alt="live camera">
      <div style="margin-top: 12px;">
        <button class="app-btn primary" onclick="post('/api/capture')">Capture Current Pose</button>
        <button class="app-btn secondary" onclick="post('/api/calibrate')">Calibrate</button>
        <button class="app-btn secondary" onclick="post('/api/next_target')">Skip To Next Target</button>
        <button class="app-btn warn" onclick="post('/api/reset')">Reset Session</button>
      </div>
      <p class="muted">During the seed stage, collect five diverse board poses. After that, match the real board to the ghost overlay and capture each target.</p>
      <div class="subsection">
        <h3>Recent Captures</h3>
        <div class="strip-widget">
          <div class="strip-controls">
            <button class="app-btn secondary" onclick="scrollStrip('thumbs', -280)">◀</button>
            <button class="app-btn secondary" onclick="scrollStrip('thumbs', 280)">▶</button>
          </div>
          <div class="thumb-strip" id="thumbs"></div>
        </div>
      </div>
      <div class="subsection">
        <h3>Final Reprojection Inspection</h3>
        <div class="strip-widget">
          <div class="strip-controls">
            <button class="app-btn secondary" onclick="scrollStrip('reproj-thumbs', -280)">◀</button>
            <button class="app-btn secondary" onclick="scrollStrip('reproj-thumbs', 280)">▶</button>
          </div>
          <div class="thumb-strip" id="reproj-thumbs"></div>
        </div>
      </div>
    </div>
    <div class="panel">
      <h2>Session Status</h2>
      <div class="controls">
        <label>3D view height
          <input id="pose3d-size" type="range" min="260" max="620" step="10" value="360" oninput="setPose3dSize(this.value)">
        </label>
        <label>Plot size
          <input id="plot-size" type="range" min="180" max="420" step="10" value="260" oninput="setPlotSize(this.value)">
        </label>
        <label>Plot scale
          <select id="plot-scale" onchange="refreshStatus()">
            <option value="log" selected>log</option>
            <option value="linear">linear</option>
          </select>
        </label>
      </div>
      <div class="plot-box" style="margin-bottom: 14px;">
        <h3 style="margin: 0 0 8px 0;">Chosen Poses 3D View</h3>
        <div id="poses3d-live"></div>
        <div class="legend">Camera is fixed at the origin. Seed, guided, live, and next-target poses are shown in different colors.</div>
      </div>
      <div id="status"></div>
      <div class="plots">
        <div class="plot-box">
          <h3 style="margin: 0 0 8px 0;">Image-Plane Point Coverage</h3>
          <img id="coverage-plot" alt="image-plane coverage plot">
          <div class="legend">Accumulated detected board points from accepted captures in image coordinates.</div>
        </div>
        <div class="plot-box">
          <h3 style="margin: 0 0 8px 0;">Final Residual Heatmap</h3>
          <img id="residual-heatmap" alt="final residual heatmap">
          <div class="legend">Available after final calibration. Residual magnitude is accumulated over the image plane.</div>
        </div>
        <div class="plot-box">
          <h3 style="margin: 0 0 8px 0;">Eigenvalue Spectrum</h3>
          <svg class="metric-plot" id="eig-plot" viewBox="0 0 620 180"></svg>
          <div class="legend">Y-axis: log10(eigenvalue), X-axis: accepted capture index</div>
          <div class="legend" id="eig-spectrum-legend"></div>
          <div class="toggle-row" id="eig-line-toggles"></div>
        </div>
        <div class="plot-box">
          <h3 style="margin: 0 0 8px 0;">Eigenvalue Extrema</h3>
          <svg class="metric-plot" id="eig-extrema-plot" viewBox="0 0 620 180"></svg>
          <div class="legend">Blue: max eigenvalue, Red: min eigenvalue</div>
          <div class="legend" id="eig-extrema-legend"></div>
          <div class="toggle-row" id="eig-extrema-toggles"></div>
        </div>
        <div class="plot-box">
          <h3 style="margin: 0 0 8px 0;">Parameter Uncertainty</h3>
          <svg class="metric-plot" id="uncert-plot" viewBox="0 0 620 180"></svg>
          <div class="legend">Approx. std-dev from the inverse calibration information matrix</div>
          <div class="legend" id="uncert-legend"></div>
          <div class="toggle-row" id="uncert-toggles"></div>
        </div>
        <div class="plot-box">
          <h3 style="margin: 0 0 8px 0;">Final Reprojection RMS By Image</h3>
          <svg class="metric-plot" id="reproj-plot" viewBox="0 0 620 180"></svg>
          <div class="legend">Y-axis: RMS reprojection error in pixels, X-axis: capture index</div>
          <div class="legend" id="reproj-legend"></div>
        </div>
      </div>
    </div>
  </div>
  <div class="modal" id="image-modal" onclick="closeModal()">
    <div class="modal-inner" onclick="event.stopPropagation()">
      <div class="modal-toolbar">
        <label>Zoom
          <input id="modal-zoom" type="range" min="1" max="4" step="0.1" value="1" oninput="setModalZoom(this.value)">
        </label>
        <a id="modal-open-link" href="#" target="_blank" rel="noopener noreferrer">Open original</a>
        <button class="app-btn modal-close" onclick="closeModal()">Back</button>
      </div>
      <div class="modal-image-wrap">
        <img id="modal-img" alt="expanded preview">
      </div>
      <div class="modal-caption" id="modal-caption"></div>
    </div>
  </div>
  <script>
    function setPlotSize(value) {
      document.documentElement.style.setProperty('--plot-height', value + 'px');
    }
    function setThumbSize(value) {
      document.documentElement.style.setProperty('--thumb-width', value + 'px');
      document.documentElement.style.setProperty('--thumb-height', Math.round(value * 0.625) + 'px');
    }
    function setPose3dSize(value) {
      document.documentElement.style.setProperty('--pose3d-height', value + 'px');
      refreshPosePlot();
    }
    function formatMatrixCell(value) {
      return Number(value).toFixed(4);
    }
    function renderCameraMatrix(matrix) {
      if (!matrix || !Array.isArray(matrix) || !matrix.length) return 'n/a';
      const rows = matrix.map((row, idx) => {
        const left = idx === 0 ? '⎡' : (idx === matrix.length - 1 ? '⎣' : '⎢');
        const right = idx === 0 ? '⎤' : (idx === matrix.length - 1 ? '⎦' : '⎥');
        const cells = row.map(value => `<div class="matrix-cell">${formatMatrixCell(value)}</div>`).join('');
        return `<div class="matrix-row"><div class="matrix-bracket">${left}</div>${cells}<div class="matrix-bracket">${right}</div></div>`;
      }).join('');
      return `
        <div class="matrix-card">
          <div class="matrix-title">Final Camera Calibration Matrix</div>
          <div class="matrix-grid">${rows}</div>
        </div>
      `;
    }
    function getPlotScale() {
      return document.getElementById('plot-scale').value || 'log';
    }
    function scaleSeries(values) {
      const mode = getPlotScale();
      if (mode === 'linear') return values.slice();
      return values.map(v => Math.log10(Math.max(v, 1e-12)));
    }
    function formatScaleValue(value) {
      const mode = getPlotScale();
      return mode === 'linear' ? Number(value).toExponential(2) : value.toFixed(1);
    }
    function scrollStrip(id, delta) {
      document.getElementById(id).scrollBy({ left: delta, behavior: 'smooth' });
    }
    let spectrumFilter = 'all';
    let extremaFilter = 'all';
    let uncertaintyFilter = 'all';
    function getSpectrumFilter() { return spectrumFilter; }
    function getExtremaFilter() { return extremaFilter; }
    function getUncertaintyFilter() { return uncertaintyFilter; }
    function renderToggleChips(containerId, options, activeValue, setterName) {
      const container = document.getElementById(containerId);
      container.innerHTML = '';
      for (const opt of options) {
        const btn = document.createElement('button');
        btn.className = 'toggle-chip' + (opt.value === activeValue ? ' active' : '');
        btn.textContent = opt.label;
        btn.onclick = () => { window[setterName](opt.value); };
        container.appendChild(btn);
      }
    }
    function setSpectrumFilter(value) { spectrumFilter = value; refreshStatus(); }
    function setExtremaFilter(value) { extremaFilter = value; refreshStatus(); }
    function setUncertaintyFilter(value) { uncertaintyFilter = value; refreshStatus(); }
    function syncSpectrumFilter(numEig) {
      const options = [{ value: 'all', label: 'All lines' }];
      for (let i = 0; i < numEig; i++) {
        options.push({
          value: String(i),
          label: i === 0 ? `eig ${i + 1} (min)` : (i === numEig - 1 ? `eig ${i + 1} (max)` : `eig ${i + 1}`),
        });
      }
      if (!options.some(opt => opt.value === spectrumFilter)) spectrumFilter = 'all';
      renderToggleChips('eig-line-toggles', options, spectrumFilter, 'setSpectrumFilter');
    }
    function syncExtremaFilter() {
      renderToggleChips(
        'eig-extrema-toggles',
        [
          { value: 'all', label: 'Both' },
          { value: 'max', label: 'Max only' },
          { value: 'min', label: 'Min only' },
        ],
        extremaFilter,
        'setExtremaFilter'
      );
    }
    function syncUncertaintyFilter(names) {
      const options = [{ value: 'all', label: 'All params' }].concat(names.map((name, idx) => ({ value: String(idx), label: name })));
      if (!options.some(opt => opt.value === uncertaintyFilter)) uncertaintyFilter = 'all';
      renderToggleChips('uncert-toggles', options, uncertaintyFilter, 'setUncertaintyFilter');
    }
    function makeSeriesPath(values, width, height, pad, minValOverride, maxValOverride) {
      if (!values || values.length === 0) return '';
      const minVal = (minValOverride === undefined || minValOverride === null) ? Math.min(...values) : minValOverride;
      const maxVal = (maxValOverride === undefined || maxValOverride === null) ? Math.max(...values) : maxValOverride;
      const span = Math.max(maxVal - minVal, 1e-9);
      return values.map((v, i) => {
        const x = pad + (values.length === 1 ? 0 : i * (width - 2 * pad) / (values.length - 1));
        const y = height - pad - ((v - minVal) / span) * (height - 2 * pad);
        return (i === 0 ? 'M' : 'L') + x.toFixed(2) + ' ' + y.toFixed(2);
      }).join(' ');
    }
    function resetSvg(svg, width, height) {
      svg.innerHTML = '';
      const bg = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      bg.setAttribute('x', '0'); bg.setAttribute('y', '0'); bg.setAttribute('width', width); bg.setAttribute('height', height);
      bg.setAttribute('fill', '#0f0f0f');
      svg.appendChild(bg);
    }
    function drawAxes(svg, width, height, pad, xLabel, yLabel, yTicks, xTicks) {
      const ns = 'http://www.w3.org/2000/svg';
      const xAxis = document.createElementNS(ns, 'line');
      xAxis.setAttribute('x1', pad);
      xAxis.setAttribute('y1', height - pad);
      xAxis.setAttribute('x2', width - pad);
      xAxis.setAttribute('y2', height - pad);
      xAxis.setAttribute('stroke', '#7a7a7a');
      svg.appendChild(xAxis);
      const yAxis = document.createElementNS(ns, 'line');
      yAxis.setAttribute('x1', pad);
      yAxis.setAttribute('y1', pad);
      yAxis.setAttribute('x2', pad);
      yAxis.setAttribute('y2', height - pad);
      yAxis.setAttribute('stroke', '#7a7a7a');
      svg.appendChild(yAxis);
      for (const tick of (yTicks || [])) {
        const tickLine = document.createElementNS(ns, 'line');
        tickLine.setAttribute('x1', pad - 4);
        tickLine.setAttribute('y1', tick.y);
        tickLine.setAttribute('x2', pad);
        tickLine.setAttribute('y2', tick.y);
        tickLine.setAttribute('stroke', '#9a9a9a');
        svg.appendChild(tickLine);
        const label = document.createElementNS(ns, 'text');
        label.setAttribute('x', pad - 8);
        label.setAttribute('y', tick.y + 4);
        label.setAttribute('text-anchor', 'end');
        label.setAttribute('fill', '#cfcfcf');
        label.setAttribute('font-size', '11');
        label.textContent = tick.label;
        svg.appendChild(label);
      }
      for (const tick of (xTicks || [])) {
        const tickLine = document.createElementNS(ns, 'line');
        tickLine.setAttribute('x1', tick.x);
        tickLine.setAttribute('y1', height - pad);
        tickLine.setAttribute('x2', tick.x);
        tickLine.setAttribute('y2', height - pad + 4);
        tickLine.setAttribute('stroke', '#9a9a9a');
        svg.appendChild(tickLine);
        const label = document.createElementNS(ns, 'text');
        label.setAttribute('x', tick.x);
        label.setAttribute('y', height - pad + 18);
        label.setAttribute('text-anchor', 'middle');
        label.setAttribute('fill', '#cfcfcf');
        label.setAttribute('font-size', '11');
        label.textContent = tick.label;
        svg.appendChild(label);
      }
      const xText = document.createElementNS(ns, 'text');
      xText.setAttribute('x', width / 2);
      xText.setAttribute('y', height - 10);
      xText.setAttribute('text-anchor', 'middle');
      xText.setAttribute('fill', '#cfcfcf');
      xText.setAttribute('font-size', '12');
      xText.textContent = xLabel;
      svg.appendChild(xText);
      const yText = document.createElementNS(ns, 'text');
      yText.setAttribute('x', 18);
      yText.setAttribute('y', height / 2);
      yText.setAttribute('text-anchor', 'middle');
      yText.setAttribute('transform', `rotate(-90 18 ${height / 2})`);
      yText.setAttribute('fill', '#cfcfcf');
      yText.setAttribute('font-size', '12');
      yText.textContent = yLabel;
      svg.appendChild(yText);
    }
    function makeTicks(values, height, pad, formatter) {
      if (!values || values.length === 0) return [];
      const minVal = Math.min(...values);
      const maxVal = Math.max(...values);
      const span = Math.max(maxVal - minVal, 1e-9);
      return [minVal, (minVal + maxVal) / 2, maxVal].map(v => ({
        y: height - pad - ((v - minVal) / span) * (height - 2 * pad),
        label: formatter(v)
      }));
    }
    function makeXTicks(count, width, pad) {
      if (!count || count <= 0) return [];
      const candidates = Array.from(new Set([0, Math.floor((count - 1) / 2), count - 1])).sort((a, b) => a - b);
      return candidates.map(idx => ({
        x: pad + (count === 1 ? 0 : idx * (width - 2 * pad) / (count - 1)),
        label: String(idx + 1),
      }));
    }
    function renderEigenPlot(history) {
      const svg = document.getElementById('eig-plot');
      const width = 620, height = 180, pad = 40;
      resetSvg(svg, width, height);
      if (!history || history.length === 0) return;
      const colors = ['#ffb300', '#29b6f6', '#66bb6a', '#ef5350', '#ab47bc', '#ffa726', '#26c6da', '#9ccc65', '#ec407a'];
      const numEig = history[0].eigvals.length;
      syncSpectrumFilter(numEig);
      const filterValue = getSpectrumFilter();
      const allValues = [];
      const legend = document.getElementById('eig-spectrum-legend');
      legend.innerHTML = '';
      const seriesList = [];
      const visibleValues = [];
      for (let eigIdx = 0; eigIdx < numEig; eigIdx++) {
        const series = scaleSeries(history.map(item => Math.max(item.eigvals[eigIdx], 1e-12)));
        seriesList.push(series);
        allValues.push(...series);
      }
      let visibleMin = Math.min(...allValues);
      let visibleMax = Math.max(...allValues);
      for (let eigIdx = 0; eigIdx < numEig; eigIdx++) {
        if (filterValue !== 'all' && filterValue !== String(eigIdx)) continue;
        visibleValues.push(...seriesList[eigIdx]);
      }
      if (visibleValues.length > 0) {
        visibleMin = Math.min(...visibleValues);
        visibleMax = Math.max(...visibleValues);
      }
      for (let eigIdx = 0; eigIdx < numEig; eigIdx++) {
        if (filterValue !== 'all' && filterValue !== String(eigIdx)) continue;
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.setAttribute('d', makeSeriesPath(seriesList[eigIdx], width, height, pad, visibleMin, visibleMax));
        path.setAttribute('fill', 'none');
        path.setAttribute('stroke', colors[eigIdx % colors.length]);
        path.setAttribute('stroke-width', eigIdx === numEig - 1 ? '3' : '1.6');
        svg.appendChild(path);
        const name = eigIdx === 0 ? `eig ${eigIdx + 1} (min)` : (eigIdx === numEig - 1 ? `eig ${eigIdx + 1} (max)` : `eig ${eigIdx + 1}`);
        legend.innerHTML += `<span style="margin-right:12px;"><span class="swatch" style="background:${colors[eigIdx % colors.length]}"></span>${name}</span>`;
      }
      drawAxes(svg, width, height, pad, 'capture index', getPlotScale() === 'linear' ? 'eig' : 'log10 eig', makeTicks(visibleValues.length > 0 ? visibleValues : allValues, height, pad, formatScaleValue), makeXTicks(history.length, width, pad));
    }
    function renderEigenExtremaPlot(history) {
      const svg = document.getElementById('eig-extrema-plot');
      const width = 620, height = 180, pad = 40;
      resetSvg(svg, width, height);
      if (!history || history.length === 0) return;
      const maxSeries = scaleSeries(history.map(item => Math.max(item.max_eig, 1e-12)));
      const minSeries = scaleSeries(history.map(item => Math.max(item.min_eig, 1e-12)));
      const extremaFilter = getExtremaFilter();
      syncExtremaFilter();
      document.getElementById('eig-extrema-legend').innerHTML =
        '<span style="margin-right:12px;"><span class="swatch" style="background:#42a5f5"></span>max eig</span>' +
        '<span style="margin-right:12px;"><span class="swatch" style="background:#ef5350"></span>min eig</span>';
      const visibleExtremaValues = [];
      if (extremaFilter === 'all' || extremaFilter === 'max') visibleExtremaValues.push(...maxSeries);
      if (extremaFilter === 'all' || extremaFilter === 'min') visibleExtremaValues.push(...minSeries);
      const sharedMin = Math.min(...visibleExtremaValues);
      const sharedMax = Math.max(...visibleExtremaValues);
      const makePath = (series, color, widthPx) => {
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.setAttribute('d', makeSeriesPath(series, width, height, pad, sharedMin, sharedMax));
        path.setAttribute('fill', 'none');
        path.setAttribute('stroke', color);
        path.setAttribute('stroke-width', widthPx);
        svg.appendChild(path);
      };
      if (extremaFilter === 'all' || extremaFilter === 'max') makePath(maxSeries, '#42a5f5', '3');
      if (extremaFilter === 'all' || extremaFilter === 'min') makePath(minSeries, '#ef5350', '2');
      drawAxes(svg, width, height, pad, 'capture index', getPlotScale() === 'linear' ? 'eig' : 'log10 eig', makeTicks(visibleExtremaValues, height, pad, formatScaleValue), makeXTicks(history.length, width, pad));
    }
    function renderUncertaintyPlot(history) {
      const svg = document.getElementById('uncert-plot');
      const width = 620, height = 180, pad = 40;
      resetSvg(svg, width, height);
      if (!history || history.length === 0 || !history[0].param_std) return;
      const names = history[0].param_std.map(item => item.name);
      syncUncertaintyFilter(names);
      const filterValue = getUncertaintyFilter();
      const colors = ['#ffb300', '#29b6f6', '#66bb6a', '#ef5350', '#ab47bc', '#ffa726', '#26c6da', '#9ccc65', '#ec407a'];
      const legend = document.getElementById('uncert-legend');
      legend.innerHTML = '';
      const seriesList = [];
      const allValues = [];
      const visibleValues = [];
      for (let i = 0; i < names.length; i++) {
        const series = scaleSeries(history.map(item => Math.max(item.param_std[i].value, 1e-12)));
        seriesList.push(series);
        allValues.push(...series);
      }
      let visibleMin = Math.min(...allValues);
      let visibleMax = Math.max(...allValues);
      for (let i = 0; i < names.length; i++) {
        if (filterValue !== 'all' && filterValue !== String(i)) continue;
        visibleValues.push(...seriesList[i]);
      }
      if (visibleValues.length > 0) {
        visibleMin = Math.min(...visibleValues);
        visibleMax = Math.max(...visibleValues);
      }
      for (let i = 0; i < names.length; i++) {
        if (filterValue !== 'all' && filterValue !== String(i)) continue;
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.setAttribute('d', makeSeriesPath(seriesList[i], width, height, pad, visibleMin, visibleMax));
        path.setAttribute('fill', 'none');
        path.setAttribute('stroke', colors[i % colors.length]);
        path.setAttribute('stroke-width', i < 4 ? '2.5' : '1.6');
        svg.appendChild(path);
        legend.innerHTML += `<span style="margin-right:12px;"><span class="swatch" style="background:${colors[i % colors.length]}"></span>${names[i]}</span>`;
      }
      drawAxes(svg, width, height, pad, 'capture index', getPlotScale() === 'linear' ? 'std' : 'log10 std', makeTicks(visibleValues.length > 0 ? visibleValues : allValues, height, pad, formatScaleValue), makeXTicks(history.length, width, pad));
    }
    function renderReprojectionPlot(items) {
      const svg = document.getElementById('reproj-plot');
      const width = 620, height = 180, pad = 40;
      resetSvg(svg, width, height);
      if (!items || items.length === 0) return;
      document.getElementById('reproj-legend').innerHTML =
        '<span style="margin-right:12px;"><span class="swatch" style="background:#66bb6a"></span>RMS reprojection error</span>';
      const rmsSeries = items.map(item => Math.max(item.rms_error_px, 1e-9));
      const sharedMin = Math.min(...rmsSeries);
      const sharedMax = Math.max(...rmsSeries);
      const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      path.setAttribute('d', makeSeriesPath(rmsSeries, width, height, pad, sharedMin, sharedMax));
      path.setAttribute('fill', 'none');
      path.setAttribute('stroke', '#66bb6a');
      path.setAttribute('stroke-width', '3');
      svg.appendChild(path);
      const maxVal = Math.max(...rmsSeries);
      const minVal = Math.min(...rmsSeries);
      const span = Math.max(maxVal - minVal, 1e-9);
      items.forEach((item, idx) => {
        const x = pad + (items.length === 1 ? 0 : idx * (width - 2 * pad) / (items.length - 1));
        const y = height - pad - ((item.rms_error_px - minVal) / span) * (height - 2 * pad);
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', x.toFixed(2));
        circle.setAttribute('cy', y.toFixed(2));
        circle.setAttribute('r', '3.5');
        circle.setAttribute('fill', '#a5d6a7');
        svg.appendChild(circle);
      });
      drawAxes(svg, width, height, pad, 'capture index', 'RMS px', makeTicks(rmsSeries, height, pad, v => v.toFixed(2)), makeXTicks(items.length, width, pad));
    }
    function setModalZoom(value) {
      document.getElementById('modal-img').style.transform = `scale(${value})`;
    }
    function openModal(src, caption) {
      document.getElementById('modal-img').src = src;
      document.getElementById('modal-open-link').href = src;
      document.getElementById('modal-zoom').value = 1;
      setModalZoom(1);
      document.getElementById('modal-caption').textContent = caption || '';
      document.getElementById('image-modal').classList.add('open');
    }
    function closeModal() {
      document.getElementById('image-modal').classList.remove('open');
    }
    async function refreshStatus() {
      const resp = await fetch('/api/status');
      const data = await resp.json();
      const status = document.getElementById('status');
      status.innerHTML = `
        <div class="status-grid">
          <div class="status-card"><div class="status-label">Stage</div><div class="status-value">${data.stage}</div></div>
          <div class="status-card"><div class="status-label">Captured</div><div class="status-value">${data.num_captures} / ${data.target_total}</div></div>
          <div class="status-card"><div class="status-label">Input Dims</div><div class="status-value">${data.camera_input_dims[0]} x ${data.camera_input_dims[1]}</div></div>
          <div class="status-card"><div class="status-label">Detection</div><div class="status-value">${data.detection_status}</div></div>
          <div class="status-card"><div class="status-label">Alignment Error</div><div class="status-value">${data.alignment_error_px === null ? 'n/a' : data.alignment_error_px.toFixed(2) + ' px'}</div></div>
          <div class="status-card"><div class="status-label">Seed RMS</div><div class="status-value">${data.seed_rms === null ? 'n/a' : data.seed_rms.toFixed(4) + ' px'}</div></div>
          <div class="status-card"><div class="status-label">Final RMS</div><div class="status-value">${data.final_rms === null ? 'n/a' : data.final_rms.toFixed(4) + ' px'}</div></div>
          <div class="status-card"><div class="status-label">Current Min Eig</div><div class="status-value">${data.current_min_eig === null ? 'n/a' : Number(data.current_min_eig).toExponential(4)}</div></div>
          <div class="status-card"><div class="status-label">Current Max Eig</div><div class="status-value">${data.current_max_eig === null ? 'n/a' : Number(data.current_max_eig).toExponential(4)}</div></div>
          <div class="status-card status-wide message-card"><div class="status-label">Session Message</div><div class="status-value" style="font-weight:500;">${data.message}</div></div>
        </div>
        ${renderCameraMatrix(data.final_camera_matrix)}
      `;
      renderEigenPlot(data.eig_history || []);
      renderEigenExtremaPlot(data.eig_history || []);
      renderUncertaintyPlot(data.eig_history || []);
      const coverage = document.getElementById('coverage-plot');
      if (data.image_plane_coverage_jpeg_base64) {
        coverage.src = 'data:image/jpeg;base64,' + data.image_plane_coverage_jpeg_base64;
      }
      const residualHeatmap = document.getElementById('residual-heatmap');
      if (data.final_residual_heatmap_jpeg_base64) {
        residualHeatmap.src = 'data:image/jpeg;base64,' + data.final_residual_heatmap_jpeg_base64;
      } else {
        residualHeatmap.src = '';
      }
      const thumbs = document.getElementById('thumbs');
      thumbs.innerHTML = '';
      for (const item of data.recent_captures) {
        const card = document.createElement('div');
        card.className = 'thumb-card';
        const img = document.createElement('img');
        img.className = 'clickable';
        img.src = 'data:image/jpeg;base64,' + item.jpeg_base64;
        img.title = item.label;
        img.onclick = () => openModal(item.full_image_url || img.src, item.label);
        const label = document.createElement('div');
        label.className = 'label';
        label.textContent = item.label;
        card.appendChild(img);
        card.appendChild(label);
        thumbs.appendChild(card);
      }
      const reproj = document.getElementById('reproj-thumbs');
      reproj.innerHTML = '';
      for (const item of data.final_projection_overlays) {
        const card = document.createElement('div');
        card.className = 'thumb-card';
        const img = document.createElement('img');
        img.className = 'clickable';
        img.src = 'data:image/jpeg;base64,' + item.jpeg_base64;
        img.title = item.label;
        img.onclick = () => openModal(item.full_overlay_url || img.src, item.label + ' | rms=' + item.rms_error_px.toFixed(3) + ' px');
        const label = document.createElement('div');
        label.className = 'label';
        label.textContent = item.label + ' | rms=' + item.rms_error_px.toFixed(3) + ' px';
        card.appendChild(img);
        card.appendChild(label);
        reproj.appendChild(card);
      }
      renderReprojectionPlot(data.final_projection_overlays || []);
    }
    async function post(path) {
      await fetch(path, { method: 'POST' });
      await refreshStatus();
    }
    let lastPoseRevision = null;
    let lastPoseLayoutCamera = null;
    let posePlotInitialized = false;
    async function refreshPosePlot() {
      const resp = await fetch('/api/poses3d_json');
      const payload = await resp.json();
      if (payload.revision !== undefined && payload.revision === lastPoseRevision && posePlotInitialized) {
        return;
      }
      const layout = payload.layout || {};
      layout.paper_bgcolor = '#0f0f0f';
      layout.plot_bgcolor = '#0f0f0f';
      layout.font = { color: '#f2f2f2' };
      if (lastPoseLayoutCamera) {
        layout.scene = layout.scene || {};
        layout.scene.camera = lastPoseLayoutCamera;
      }
      Plotly.react('poses3d-live', payload.data || [], layout, {
        displayModeBar: 'hover',
        responsive: true,
        scrollZoom: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['lasso2d', 'select2d', 'toImage', 'autoScale2d', 'resetScale2d']
      });
      const poseDiv = document.getElementById('poses3d-live');
      if (!posePlotInitialized) {
        poseDiv.on('plotly_relayout', (eventData) => {
          if (eventData && eventData['scene.camera']) {
            lastPoseLayoutCamera = eventData['scene.camera'];
          }
        });
        posePlotInitialized = true;
      }
      if (poseDiv.layout && poseDiv.layout.scene && poseDiv.layout.scene.camera) {
        lastPoseLayoutCamera = poseDiv.layout.scene.camera;
      }
      lastPoseRevision = payload.revision;
    }
    function refreshFrame() {
      document.getElementById('live').src = '/frame.jpg?ts=' + Date.now();
    }
    setInterval(refreshFrame, 180);
    setInterval(refreshStatus, 700);
    setInterval(refreshPosePlot, 1500);
    refreshStatus();
    refreshPosePlot();
  </script>
</body>
</html>
""".replace("__PLOTLY_JS__", PLOTLY_JS)


@dataclass
class CaptureRecord:
    stage: str
    image_path: str
    num_points: int
    descriptor: list[float]
    jpeg_base64: str


@dataclass
class DetectionResult:
    ok: bool
    image_points: np.ndarray
    object_points: np.ndarray
    board_quad: np.ndarray | None
    num_points: int
    label: str
    point_ids: np.ndarray | None = None


def available_video_devices() -> list[str]:
    return sorted(str(path) for path in pathlib.Path("/dev").glob("video*"))


def open_camera(camera_index: int, width: int, height: int, fps: float) -> cv2.VideoCapture:
    attempts = [
        ("default", lambda: cv2.VideoCapture(int(camera_index))),
        ("v4l2", lambda: cv2.VideoCapture(int(camera_index), cv2.CAP_V4L2)),
        ("device-path", lambda: cv2.VideoCapture(f"/dev/video{int(camera_index)}", cv2.CAP_V4L2)),
    ]

    errors: list[str] = []
    for name, factory in attempts:
        cap = factory()
        if cap is None:
            errors.append(f"{name}: constructor returned None")
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        cap.set(cv2.CAP_PROP_FPS, float(fps))
        ok, frame = cap.read()
        if cap.isOpened() and ok and frame is not None:
            return cap
        errors.append(f"{name}: opened={cap.isOpened()} frame_ok={ok}")
        cap.release()

    devices = ", ".join(available_video_devices()) or "none"
    raise RuntimeError(
        f"Could not open camera index {camera_index}. "
        f"Tried default/GStreamer and V4L2 paths. "
        f"Available devices: {devices}. "
        f"Attempt details: {'; '.join(errors)}"
    )


def order_quad(quad: np.ndarray) -> np.ndarray:
    pts = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float64)
    ordered[0] = pts[np.argmin(sums)]
    ordered[2] = pts[np.argmax(sums)]
    ordered[1] = pts[np.argmin(diffs)]
    ordered[3] = pts[np.argmax(diffs)]
    return ordered


def quad_descriptor(quad: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    ordered = order_quad(quad)
    width, height = image_size
    center = ordered.mean(axis=0)
    area = cv2.contourArea(ordered.astype(np.float32))
    vec_top = ordered[1] - ordered[0]
    angle = math.atan2(float(vec_top[1]), float(vec_top[0]))
    skew_x = float(((ordered[1][1] - ordered[0][1]) + (ordered[2][1] - ordered[3][1])) / 2.0)
    skew_y = float(((ordered[3][0] - ordered[0][0]) + (ordered[2][0] - ordered[1][0])) / 2.0)
    return np.array(
        [
            center[0] / max(width, 1),
            center[1] / max(height, 1),
            area / max(width * height, 1),
            angle / math.pi,
            skew_x / max(height, 1),
            skew_y / max(width, 1),
        ],
        dtype=np.float64,
    )


def camera_from_intrinsics(intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    intrinsics = np.asarray(intrinsics, dtype=np.float64).reshape(-1)
    camera_matrix = np.array(
        [[intrinsics[0], 0.0, intrinsics[2]], [0.0, intrinsics[1], intrinsics[3]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist = np.zeros((5, 1), dtype=np.float64)
    if intrinsics.size > 4:
        tail = intrinsics[4 : min(9, intrinsics.size)]
        dist[: tail.size, 0] = tail
    return camera_matrix, dist


def jpeg_bytes(image_bgr: np.ndarray, quality: int = 85) -> bytes:
    ok, encoded = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("Failed to encode JPEG image.")
    return encoded.tobytes()


def under_root(path: pathlib.Path, root: pathlib.Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def board_outer_corners(width: float, height: float) -> np.ndarray:
    return np.array([[0.0, 0.0, 0.0], [width, 0.0, 0.0], [width, height, 0.0], [0.0, height, 0.0]], dtype=np.float64)


def render_texture_overlay(frame_bgr: np.ndarray, texture_gray: np.ndarray, target_quad: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    overlay = frame_bgr.copy()
    tex_bgr = cv2.cvtColor(texture_gray, cv2.COLOR_GRAY2BGR)
    tex_h, tex_w = tex_bgr.shape[:2]
    src = np.array([[0, 0], [tex_w - 1, 0], [tex_w - 1, tex_h - 1], [0, tex_h - 1]], dtype=np.float32)
    dst = order_quad(target_quad).astype(np.float32)
    homography = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(tex_bgr, homography, (frame_bgr.shape[1], frame_bgr.shape[0]))
    warped_mask = cv2.warpPerspective(np.full((tex_h, tex_w), 255, dtype=np.uint8), homography, (frame_bgr.shape[1], frame_bgr.shape[0]))
    mask = warped_mask > 0
    overlay[mask] = cv2.addWeighted(frame_bgr[mask], 1.0 - alpha, warped[mask], alpha, 0.0)
    cv2.polylines(overlay, [dst.astype(np.int32)], True, (255, 255, 0), 2, cv2.LINE_AA)
    return overlay


class LiveCalibrationSession:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.output_dir = pathlib.Path(args.output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.captures_dir = self.output_dir / "captures"
        self.captures_dir.mkdir(parents=True, exist_ok=True)
        self.texture_dir = self.output_dir / "textures"
        self.texture_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.cap = open_camera(int(args.camera_index), int(args.width), int(args.height), float(args.fps))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or args.width)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or args.height)
        self.image_size = (self.width, self.height)
        self.board_width = float(args.board_cols) * float(args.board_square_size)
        self.board_height = float(args.board_rows) * float(args.board_square_size)
        self.object_points_full = self._build_board_object_points()
        self.texture_gray = self._build_texture()
        self.seed_required = int(args.seed_count)
        # total_captures controls how many frames to collect in total.
        # next_required is derived so guided phase runs until total is reached.
        _total = int(getattr(args, 'total_captures', None) or (int(args.seed_count) + int(args.next_k)))
        self.total_captures_target: int = max(_total, int(args.seed_count) + 1)
        self.next_required = max(int(args.next_k), self.total_captures_target - int(args.seed_count))
        self.auto_capture_enabled: bool = bool(getattr(args, 'auto_capture', True))
        self.auto_capture_threshold_px: float = float(getattr(args, 'auto_threshold_px', 60.0))
        self.auto_capture_cooldown_s: float = float(getattr(args, 'auto_cooldown', 2.5))
        self.last_auto_capture_time: float = 0.0
        self.last_capture_flash_time: float = 0.0
        self.stage = "seed"
        self.message = "Collect five diverse seed poses — hold board steady and move it around."
        self.captures: list[CaptureRecord] = []
        self.detected_descriptors: list[np.ndarray] = []
        self.seed_intrinsics: np.ndarray | None = None
        self.final_intrinsics: np.ndarray | None = None
        self.seed_rms: float | None = None
        self.final_rms: float | None = None
        self.final_camera_matrix: list[list[float]] | None = None
        self.eig_history: list[dict[str, Any]] = []
        self.final_projection_overlay_previews: list[dict[str, Any]] = []
        self.target_bank = self._build_target_bank()
        self.remaining_target_indices = list(range(len(self.target_bank)))
        self.current_target_idx: int | None = None
        self.latest_frame_bgr: np.ndarray | None = None
        self.latest_detection: DetectionResult | None = None
        self.latest_alignment_error: float | None = None
        self.latest_annotated_jpeg: bytes | None = None
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _build_board_object_points(self) -> np.ndarray:
        if self.args.board_type == "charuco":
            cols_inner = self.args.board_cols - 1
            rows_inner = self.args.board_rows - 1
            xs = np.arange(cols_inner, dtype=np.float64) + 1.0
            ys = np.arange(rows_inner, dtype=np.float64) + 1.0
            gx, gy = np.meshgrid(xs, ys)
            points = np.stack(
                [gx.reshape(-1) * self.args.board_square_size, gy.reshape(-1) * self.args.board_square_size, np.zeros(gx.size)],
                axis=1,
            )
            return points.astype(np.float64)

        tag_size = float(self.args.board_marker_size)
        pitch = float(self.args.board_square_size)
        points = []
        for grid_row in range(2 * self.args.board_rows):
            for grid_col in range(2 * self.args.board_cols):
                x = (grid_col // 2) * pitch + (grid_col % 2) * tag_size
                y = (grid_row // 2) * pitch + (grid_row % 2) * tag_size
                points.append((x, y, 0.0))
        return np.asarray(points, dtype=np.float64)

    def _build_texture(self) -> np.ndarray:
        if self.args.board_type == "charuco":
            texture_path = ensure_charuco_texture(
                self.texture_dir / "charuco_texture.png",
                rows=int(self.args.board_rows),
                cols=int(self.args.board_cols),
                square_size=float(self.args.board_square_size),
                marker_size=float(self.args.board_marker_size),
                aruco_dict_name=str(self.args.aruco_dict),
            )
        else:
            tag_spacing = max(float(self.args.board_square_size) / float(self.args.board_marker_size) - 1.0, 0.0)
            texture_path = ensure_aprilgrid_texture(
                self.texture_dir / "aprilgrid_texture.png",
                rows=int(self.args.board_rows),
                cols=int(self.args.board_cols),
                tag_size=float(self.args.board_marker_size),
                tag_spacing=float(tag_spacing),
                aruco_dict_name="DICT_APRILTAG_36h11",
            )
        texture = cv2.imread(str(texture_path), cv2.IMREAD_GRAYSCALE)
        if texture is None:
            raise RuntimeError(f"Failed to read generated board texture: {texture_path}")
        return texture

    def _build_target_bank(self) -> list[np.ndarray]:
        aspect = self.board_height / max(self.board_width, 1.0e-9)
        centers_x = [0.22, 0.5, 0.78]
        centers_y = [0.24, 0.5, 0.76]
        # Three scales give genuine depth variation: far / mid / close
        scales = [0.20, 0.30, 0.42]
        # Perspective tilt factors: (tilt_x, tilt_y)
        # tilt_x > 0 → right side closer (appears larger); tilt_y > 0 → bottom closer.
        # Values ≈0.28 give ~30° effective yaw/pitch; ≈0.38 give ~45°+.
        tilts = [
            ( 0.00,  0.00),                      # frontal
            ( 0.28,  0.00), (-0.28,  0.00),      # moderate yaw right / left
            ( 0.00,  0.28), ( 0.00, -0.28),      # moderate pitch down / up
            ( 0.38,  0.00), (-0.38,  0.00),      # strong yaw
            ( 0.00,  0.38), ( 0.00, -0.38),      # strong pitch
            ( 0.22,  0.22), (-0.22,  0.22),      # diagonal tilts
            ( 0.22, -0.22), (-0.22, -0.22),
        ]
        rotations = [-15.0, 0.0, 15.0]
        bank = []
        for cx_n in centers_x:
            for cy_n in centers_y:
                for scale in scales:
                    for tilt_x, tilt_y in tilts:
                        for rot_deg in rotations:
                            bank.append(self._target_quad(cx_n, cy_n, scale, rot_deg, tilt_x, tilt_y, aspect))
        return bank

    def _target_quad(self, cx_n: float, cy_n: float, scale: float, rot_deg: float, tilt_x: float, tilt_y: float, aspect: float) -> np.ndarray:
        board_w = scale * self.width
        board_h = board_w * aspect
        hw = board_w / 2.0
        hh = board_h / 2.0

        # Per-corner depth model: each corner gets an independent depth factor that
        # encodes how far that corner is from the camera relative to board centre.
        # tilt_x > 0 → right side closer; tilt_y > 0 → bottom side closer.
        # d = 1 + sign(left)*px + sign(top)*py, projected coord = base_coord / d.
        px = float(np.clip(tilt_x, -0.45, 0.45))
        py = float(np.clip(tilt_y, -0.45, 0.45))
        d_tl = max(1.0 + px + py, 0.05)   # left & top: farther when px>0, py>0
        d_tr = max(1.0 - px + py, 0.05)   # right & top
        d_br = max(1.0 - px - py, 0.05)   # right & bottom: closest for px>0, py>0
        d_bl = max(1.0 + px - py, 0.05)   # left & bottom

        # Each corner projected: (x_3d, y_3d) / depth_factor
        base = np.array(
            [
                [-hw / d_tl, -hh / d_tl],  # TL
                [ hw / d_tr, -hh / d_tr],  # TR
                [ hw / d_br,  hh / d_br],  # BR
                [-hw / d_bl,  hh / d_bl],  # BL
            ],
            dtype=np.float64,
        )

        # In-plane rotation applied after perspective (rotation around camera Z-axis)
        angle = math.radians(rot_deg)
        rot = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]], dtype=np.float64)
        quad = base @ rot.T

        quad[:, 0] += cx_n * self.width
        quad[:, 1] += cy_n * self.height
        return order_quad(quad)

    def _detect_board(self, frame_bgr: np.ndarray) -> DetectionResult:
        if self.args.board_type == "charuco":
            found, corners, ids = detect_charuco(
                frame_bgr,
                rows=int(self.args.board_rows),
                cols=int(self.args.board_cols),
                square_size=float(self.args.board_square_size),
                marker_size=float(self.args.board_marker_size),
                aruco_dict_name=str(self.args.aruco_dict),
            )
            if not found or corners is None or ids is None:
                return DetectionResult(False, np.empty((0, 2)), np.empty((0, 3)), None, 0, "board not detected", None)
            image_points = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
            point_ids = np.asarray(ids, dtype=np.int32).reshape(-1)
            object_points = self.object_points_full[point_ids]
        else:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            result = detect_kalibr_aprilgrid(
                gray,
                rows=int(self.args.board_rows),
                cols=int(self.args.board_cols),
                min_tags_for_valid_obs=1,
                max_subpix_displacement2=float(self.args.aprilgrid_max_subpix_displacement2),
                subpix_window_half_width=int(self.args.aprilgrid_subpix_window),
                subpix_max_iters=int(self.args.aprilgrid_subpix_max_iters),
                subpix_epsilon=float(self.args.aprilgrid_subpix_epsilon),
            )
            observed = np.asarray(result["observed"], dtype=bool)
            image_points = np.asarray(result["image_points"], dtype=np.float64)[observed]
            object_points = self.object_points_full[observed]
            point_ids = np.flatnonzero(observed).astype(np.int32)
            if image_points.shape[0] < 4:
                return DetectionResult(False, np.empty((0, 2)), np.empty((0, 3)), None, 0, "board not detected", None)

        if image_points.shape[0] < 4 or object_points.shape[0] < 4:
            return DetectionResult(
                False,
                image_points,
                object_points,
                None,
                int(image_points.shape[0]),
                f"too few points ({int(image_points.shape[0])})",
                point_ids if 'point_ids' in locals() else None,
            )

        homography_result = cv2.findHomography(object_points[:, :2].astype(np.float32), image_points.astype(np.float32), 0)
        if isinstance(homography_result, tuple):
            homography = homography_result[0]
        else:
            homography = homography_result
        board_quad = None
        if homography is not None:
            corners_local = np.array(
                [[0.0, 0.0], [self.board_width, 0.0], [self.board_width, self.board_height], [0.0, self.board_height]],
                dtype=np.float64,
            ).reshape(-1, 1, 2)
            quad = cv2.perspectiveTransform(corners_local.astype(np.float32), homography).reshape(-1, 2)
            board_quad = order_quad(quad)
        return DetectionResult(
            True,
            image_points,
            object_points,
            board_quad,
            int(image_points.shape[0]),
            f"{image_points.shape[0]} points",
            point_ids if 'point_ids' in locals() else None,
        )

    def _calibrate_from_captures(self, captures: list[CaptureRecord]) -> tuple[np.ndarray, float]:
        object_points = []
        image_points = []
        for record in captures:
            meta_path = pathlib.Path(record.image_path).with_suffix(".json")
            meta = json.loads(meta_path.read_text(encoding="ascii"))
            object_points.append(np.asarray(meta["object_points"], dtype=np.float32).reshape(-1, 1, 3))
            image_points.append(np.asarray(meta["image_points"], dtype=np.float32).reshape(-1, 1, 2))
        width, height = self.image_size
        camera_matrix_init = np.array([[width, 0.0, width / 2.0], [0.0, width, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        dist_init = np.zeros((5, 1), dtype=np.float64)
        retval, camera_matrix, dist_coeffs, *_ = cv2.calibrateCamera(
            object_points,
            image_points,
            self.image_size,
            camera_matrix_init,
            dist_init,
            flags=cv2.CALIB_USE_INTRINSIC_GUESS,
        )
        dist = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
        if dist.size < 5:
            dist = np.pad(dist, (0, 5 - dist.size))
        intrinsics = np.array([camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2], *dist[:5]], dtype=np.float64)
        self.final_camera_matrix = camera_matrix.astype(float).tolist()
        return intrinsics, float(retval)

    def _estimate_pose_wc(self, object_points: np.ndarray, image_points: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        if object_points.shape[0] < 4 or image_points.shape[0] < 4:
            return None
        camera_matrix, dist_coeffs = camera_from_intrinsics(intrinsics)
        pnp_flag = cv2.SOLVEPNP_ITERATIVE if object_points.shape[0] >= 6 else cv2.SOLVEPNP_EPNP
        try:
            success, rvec, tvec = cv2.solvePnP(
                object_points.astype(np.float64),
                image_points.astype(np.float64),
                camera_matrix,
                dist_coeffs,
                flags=pnp_flag,
            )
        except cv2.error:
            return None
        if not success:
            return None
        rot_cw, _ = cv2.Rodrigues(rvec)
        t_cw = np.asarray(tvec, dtype=np.float64).reshape(3)
        rot_wc = rot_cw.T
        trans_wc = -(rot_wc @ t_cw)
        return rot_wc, trans_wc

    def _estimate_board_origin_camera(self, object_points: np.ndarray, image_points: np.ndarray, intrinsics: np.ndarray) -> np.ndarray | None:
        if object_points.shape[0] < 4 or image_points.shape[0] < 4:
            return None
        camera_matrix, dist_coeffs = camera_from_intrinsics(intrinsics)
        pnp_flag = cv2.SOLVEPNP_ITERATIVE if object_points.shape[0] >= 6 else cv2.SOLVEPNP_EPNP
        try:
            success, _rvec, tvec = cv2.solvePnP(
                object_points.astype(np.float64),
                image_points.astype(np.float64),
                camera_matrix,
                dist_coeffs,
                flags=pnp_flag,
            )
        except cv2.error:
            return None
        if not success:
            return None
        tvec = np.asarray(tvec, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(tvec)):
            return None
        return tvec

    def _estimate_target_origin_camera(self, target_quad: np.ndarray, intrinsics: np.ndarray) -> np.ndarray | None:
        camera_matrix, dist_coeffs = camera_from_intrinsics(intrinsics)
        image_points = order_quad(target_quad).astype(np.float64)
        object_points = board_outer_corners(self.board_width, self.board_height).astype(np.float64)
        try:
            success, _rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_EPNP,
            )
        except cv2.error:
            return None
        if not success:
            return None
        tvec = np.asarray(tvec, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(tvec)):
            return None
        return tvec

    def _compute_eigvals_for_captures(self, captures: list[CaptureRecord], intrinsics: np.ndarray) -> np.ndarray | None:
        info = self._compute_information_summary(captures, intrinsics)
        if info is None:
            return None
        return np.asarray(info["eigvals"], dtype=float)

    def _compute_information_summary(self, captures: list[CaptureRecord], intrinsics: np.ndarray) -> dict[str, Any] | None:
        if len(captures) < 2:
            return None
        intrinsics = np.asarray(intrinsics, dtype=np.float64)
        if not np.all(np.isfinite(intrinsics)):
            return None
        h_cal = np.diag(1.0 / np.square(fim.default_intrinsics_prior_sigma(intrinsics.size)))
        pose_prior = np.diag(1.0 / np.square(np.asarray((2.0, 2.0, 2.0, 0.5, 0.5, 0.5), dtype=np.float64)))
        valid_contributions = 0
        for record in captures:
            meta_path = pathlib.Path(record.image_path).with_suffix(".json")
            meta = json.loads(meta_path.read_text(encoding="ascii"))
            object_points = np.asarray(meta["object_points"], dtype=np.float64)
            image_points = np.asarray(meta["image_points"], dtype=np.float64)
            if object_points.shape[0] < 4 or image_points.shape[0] < 4:
                continue
            pose = self._estimate_pose_wc(object_points, image_points, intrinsics)
            if pose is None:
                continue
            rot_wc, trans_wc = pose
            valid_mask = np.ones(object_points.shape[0], dtype=bool)
            try:
                jac_intr = fim.numerical_jacobian_intrinsics(object_points, rot_wc, trans_wc, intrinsics, valid_mask)
                jac_pose = fim.numerical_jacobian_pose(object_points, rot_wc, trans_wc, intrinsics, valid_mask)
            except Exception:
                continue
            if not np.all(np.isfinite(jac_intr)) or not np.all(np.isfinite(jac_pose)):
                continue
            sigma = fim.effective_pixel_noise_sigma(1.0)
            weight = 1.0 / (sigma ** 2)
            h_tt = weight * (jac_intr.T @ jac_intr)
            h_tp = weight * (jac_intr.T @ jac_pose)
            h_pp = pose_prior + weight * (jac_pose.T @ jac_pose)
            if not np.all(np.isfinite(h_tt)) or not np.all(np.isfinite(h_tp)) or not np.all(np.isfinite(h_pp)):
                continue
            try:
                h_pp_inv = np.linalg.pinv(h_pp + 1.0e-6 * np.eye(6), rcond=1.0e-10)
            except np.linalg.LinAlgError:
                try:
                    h_pp_inv = np.linalg.pinv(h_pp + 1.0e-3 * np.eye(6), rcond=1.0e-8)
                except np.linalg.LinAlgError:
                    continue
            contrib = h_tt - h_tp @ h_pp_inv @ h_tp.T
            if not np.all(np.isfinite(contrib)):
                continue
            h_cal += contrib
            valid_contributions += 1
        if valid_contributions == 0:
            return None
        h_cal = 0.5 * (h_cal + h_cal.T)
        if not np.all(np.isfinite(h_cal)):
            return None
        try:
            eigvals = np.linalg.eigvalsh(h_cal)
            cov = np.linalg.pinv(h_cal + 1.0e-9 * np.eye(h_cal.shape[0]), rcond=1.0e-10)
        except np.linalg.LinAlgError:
            return None
        std = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        names = ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3"][: std.shape[0]]
        return {
            "eigvals": eigvals,
            "param_std": [{"name": names[i], "value": float(std[i])} for i in range(std.shape[0])],
        }

    def _update_information_history(self) -> None:
        captures = list(self.captures)
        intrinsics = self.final_intrinsics if self.final_intrinsics is not None else self.seed_intrinsics
        if intrinsics is None:
            if len(captures) >= 3:
                try:
                    intrinsics, _ = self._calibrate_from_captures(captures)
                except Exception:
                    return
            else:
                return
        try:
            info = self._compute_information_summary(captures, intrinsics)
        except Exception:
            return
        if info is None:
            return
        eigvals = np.asarray(info["eigvals"], dtype=float)
        self.eig_history.append(
            {
                "capture_count": len(captures),
                "stage": self.stage,
                "eigvals": np.asarray(eigvals, dtype=float).tolist(),
                "max_eig": float(np.max(eigvals)),
                "min_eig": float(np.min(eigvals)),
                "param_std": info["param_std"],
            }
        )

    def _render_final_projection_overlay(self, image_path: pathlib.Path, intrinsics: np.ndarray) -> dict[str, Any] | None:
        meta_path = image_path.with_suffix(".json")
        if not meta_path.exists():
            return None
        meta = json.loads(meta_path.read_text(encoding="ascii"))
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            return None
        object_points = np.asarray(meta["object_points"], dtype=np.float64)
        image_points = np.asarray(meta["image_points"], dtype=np.float64)
        pose = self._estimate_pose_wc(object_points, image_points, intrinsics)
        if pose is None:
            return None
        rot_wc, trans_wc = pose
        projected = fim.project_points(object_points, rot_wc, trans_wc, intrinsics)
        if projected.shape != image_points.shape:
            return None
        finite_mask = np.isfinite(projected).all(axis=1) & np.isfinite(image_points).all(axis=1)
        if not np.any(finite_mask):
            return None
        projected = projected[finite_mask]
        image_points = image_points[finite_mask]
        errors = np.linalg.norm(projected - image_points, axis=1)
        if errors.size == 0 or not np.all(np.isfinite(errors)):
            return None
        overlay = image_bgr.copy()
        max_error = float(np.max(errors)) if errors.size else 1.0
        for obs, proj, err in zip(image_points, projected, errors):
            obs_pt = tuple(int(round(v)) for v in obs)
            proj_pt = tuple(int(round(v)) for v in proj)
            color = (0, int(max(0, 255 - 180 * (float(err) / max(max_error, 1e-9)))), min(255, int(255 * float(err) / max(max_error, 1e-9))))
            cv2.line(overlay, obs_pt, proj_pt, color, 1, cv2.LINE_AA)
            cv2.circle(overlay, obs_pt, 3, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(overlay, proj_pt, 3, (0, 255, 0), 1, cv2.LINE_AA)
        mean_error = float(np.mean(errors))
        rms_error = float(math.sqrt(np.mean(errors * errors)))
        cv2.putText(
            overlay,
            f"mean={mean_error:.3f}px rms={rms_error:.3f}px max={float(np.max(errors)):.3f}px",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
        out_dir = self.output_dir / "final_projection_overlays"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / image_path.name
        cv2.imwrite(str(out_path), overlay)
        return {
            "image_path": str(image_path),
            "overlay_path": str(out_path),
            "mean_error_px": mean_error,
            "rms_error_px": rms_error,
            "max_error_px": float(np.max(errors)),
            "jpeg_base64": base64.b64encode(jpeg_bytes(cv2.resize(overlay, (240, 135), interpolation=cv2.INTER_AREA), quality=75)).decode("ascii"),
            "full_overlay_url": f"/api/file?path={out_path.as_posix()}",
            "label": image_path.name,
        }

    def _overlay_live_reprojection(self, image_bgr: np.ndarray, detection: DetectionResult, intrinsics: np.ndarray) -> np.ndarray:
        if not detection.ok or detection.image_points.shape[0] < 4:
            return image_bgr
        pose = self._estimate_pose_wc(detection.object_points, detection.image_points, intrinsics)
        if pose is None:
            return image_bgr
        rot_wc, trans_wc = pose
        projected = fim.project_points(detection.object_points, rot_wc, trans_wc, intrinsics)
        if projected.shape != detection.image_points.shape:
            return image_bgr
        finite_mask = np.isfinite(projected).all(axis=1) & np.isfinite(detection.image_points).all(axis=1)
        if not np.any(finite_mask):
            return image_bgr
        projected = projected[finite_mask]
        image_points = detection.image_points[finite_mask]
        if not np.all(np.isfinite(projected)):
            return image_bgr
        overlay = image_bgr.copy()
        errors = np.linalg.norm(projected - image_points, axis=1)
        if errors.size == 0 or not np.all(np.isfinite(errors)):
            return image_bgr
        max_error = float(np.max(errors)) if errors.size else 1.0
        for obs, proj, err in zip(image_points, projected, errors):
            obs_pt = tuple(int(round(v)) for v in obs)
            proj_pt = tuple(int(round(v)) for v in proj)
            color = (0, int(max(0, 255 - 180 * (float(err) / max(max_error, 1e-9)))), min(255, int(255 * float(err) / max(max_error, 1e-9))))
            cv2.line(overlay, obs_pt, proj_pt, color, 1, cv2.LINE_AA)
            cv2.circle(overlay, obs_pt, 3, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(overlay, proj_pt, 3, (0, 255, 0), 1, cv2.LINE_AA)
        rms_error = float(math.sqrt(np.mean(errors * errors))) if errors.size else 0.0
        cv2.putText(
            overlay,
            f"final reprojection rms={rms_error:.3f}px",
            (20, max(40, self.height - 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return overlay

    def _render_image_plane_coverage_preview(self) -> str | None:
        captures = list(self.captures)
        if not captures:
            return None
        width, height = self.image_size
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:] = (10, 10, 10)
        grid_rows = 12
        grid_cols = 16
        occupancy = np.zeros((grid_rows, grid_cols), dtype=np.float64)
        all_points: list[np.ndarray] = []
        for record in captures:
            meta_path = pathlib.Path(record.image_path).with_suffix(".json")
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text(encoding="ascii"))
            image_points = np.asarray(meta.get("image_points", []), dtype=np.float64).reshape(-1, 2)
            if image_points.size == 0:
                continue
            finite_mask = np.isfinite(image_points).all(axis=1)
            image_points = image_points[finite_mask]
            if image_points.size == 0:
                continue
            image_points[:, 0] = np.clip(image_points[:, 0], 0.0, max(width - 1, 0))
            image_points[:, 1] = np.clip(image_points[:, 1], 0.0, max(height - 1, 0))
            all_points.append(image_points)
            for pt in image_points:
                col = min(grid_cols - 1, max(0, int((pt[0] / max(width, 1)) * grid_cols)))
                row = min(grid_rows - 1, max(0, int((pt[1] / max(height, 1)) * grid_rows)))
                occupancy[row, col] += 1.0
        if not all_points:
            return None

        occ_max = float(np.max(occupancy)) if np.any(occupancy > 0.0) else 1.0
        cell_w = width / grid_cols
        cell_h = height / grid_rows
        for row in range(grid_rows):
            for col in range(grid_cols):
                count = occupancy[row, col]
                if count <= 0.0:
                    continue
                intensity = count / max(occ_max, 1.0e-9)
                color = (
                    int(20 + 40 * intensity),
                    int(35 + 120 * intensity),
                    int(40 + 200 * intensity),
                )
                x0 = int(round(col * cell_w))
                y0 = int(round(row * cell_h))
                x1 = int(round((col + 1) * cell_w))
                y1 = int(round((row + 1) * cell_h))
                cv2.rectangle(canvas, (x0, y0), (x1, y1), color, thickness=-1)

        overlay = canvas.copy()
        for col in range(grid_cols + 1):
            x = int(round(col * cell_w))
            cv2.line(overlay, (x, 0), (x, height - 1), (55, 55, 55), 1, cv2.LINE_AA)
        for row in range(grid_rows + 1):
            y = int(round(row * cell_h))
            cv2.line(overlay, (0, y), (width - 1, y), (55, 55, 55), 1, cv2.LINE_AA)

        stage_colors = {
            "seed": (255, 180, 70),
            "guided": (80, 220, 120),
            "done": (80, 220, 120),
        }
        for record in captures:
            meta_path = pathlib.Path(record.image_path).with_suffix(".json")
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text(encoding="ascii"))
            image_points = np.asarray(meta.get("image_points", []), dtype=np.float64).reshape(-1, 2)
            if image_points.size == 0:
                continue
            finite_mask = np.isfinite(image_points).all(axis=1)
            image_points = image_points[finite_mask]
            if image_points.size == 0:
                continue
            color = stage_colors.get(record.stage, (180, 180, 180))
            for pt in image_points:
                pt_i = (int(round(float(pt[0]))), int(round(float(pt[1]))))
                cv2.circle(overlay, pt_i, 3, color, -1, cv2.LINE_AA)

        coverage_fraction = float(np.count_nonzero(occupancy > 0.0)) / float(occupancy.size)
        cv2.rectangle(overlay, (0, 0), (width - 1, height - 1), (220, 220, 220), 2, cv2.LINE_AA)
        cv2.putText(
            overlay,
            f"captures={len(captures)} covered_cells={np.count_nonzero(occupancy > 0.0)}/{occupancy.size} ({coverage_fraction * 100.0:.1f}%)",
            (18, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            "seed points = orange, guided points = green",
            (18, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (235, 235, 235),
            2,
            cv2.LINE_AA,
        )

        preview = cv2.resize(overlay, (min(960, width), int(round(min(960, width) * height / max(width, 1)))), interpolation=cv2.INTER_AREA)
        return base64.b64encode(jpeg_bytes(preview, quality=82)).decode("ascii")

    def _render_final_residual_heatmap_preview(self) -> str | None:
        if self.final_intrinsics is None:
            return None
        width, height = self.image_size
        heat = np.zeros((height, width), dtype=np.float32)
        max_err_seen = 0.0
        total_points = 0
        for record in self.captures:
            meta_path = pathlib.Path(record.image_path).with_suffix(".json")
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text(encoding="ascii"))
            object_points = np.asarray(meta.get("object_points", []), dtype=np.float64).reshape(-1, 3)
            image_points = np.asarray(meta.get("image_points", []), dtype=np.float64).reshape(-1, 2)
            if object_points.shape[0] < 4 or image_points.shape[0] < 4:
                continue
            pose = self._estimate_pose_wc(object_points, image_points, self.final_intrinsics)
            if pose is None:
                continue
            rot_wc, trans_wc = pose
            projected = fim.project_points(object_points, rot_wc, trans_wc, self.final_intrinsics)
            if projected.shape != image_points.shape:
                continue
            finite_mask = np.isfinite(projected).all(axis=1) & np.isfinite(image_points).all(axis=1)
            if not np.any(finite_mask):
                continue
            projected = projected[finite_mask]
            image_points = image_points[finite_mask]
            errors = np.linalg.norm(projected - image_points, axis=1)
            if errors.size == 0 or not np.all(np.isfinite(errors)):
                continue
            total_points += int(errors.size)
            max_err_seen = max(max_err_seen, float(np.max(errors)))
            for pt, err in zip(image_points, errors):
                x = int(np.clip(round(float(pt[0])), 0, width - 1))
                y = int(np.clip(round(float(pt[1])), 0, height - 1))
                cv2.circle(heat, (x, y), 18, float(err), thickness=-1, lineType=cv2.LINE_AA)
        if total_points == 0:
            return None
        heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=14.0, sigmaY=14.0)
        heat_norm = heat / max(float(np.max(heat)), 1.0e-9)
        heat_u8 = np.clip(255.0 * heat_norm, 0, 255).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_TURBO)
        background = np.zeros((height, width, 3), dtype=np.uint8)
        background[:] = (16, 16, 16)
        overlay = cv2.addWeighted(background, 0.35, heat_color, 0.95, 0.0)
        grid_rows = 12
        grid_cols = 16
        for col in range(grid_cols + 1):
            x = int(round(col * width / grid_cols))
            cv2.line(overlay, (x, 0), (x, height - 1), (50, 50, 50), 1, cv2.LINE_AA)
        for row in range(grid_rows + 1):
            y = int(round(row * height / grid_rows))
            cv2.line(overlay, (0, y), (width - 1, y), (50, 50, 50), 1, cv2.LINE_AA)
        cv2.rectangle(overlay, (0, 0), (width - 1, height - 1), (220, 220, 220), 2, cv2.LINE_AA)
        cv2.putText(
            overlay,
            f"final residual heatmap | points={total_points} | max residual={max_err_seen:.3f}px",
            (18, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            "hotter colors indicate larger accumulated reprojection residual",
            (18, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )
        preview = cv2.resize(overlay, (min(960, width), int(round(min(960, width) * height / max(width, 1)))), interpolation=cv2.INTER_AREA)
        return base64.b64encode(jpeg_bytes(preview, quality=82)).decode("ascii")

    def _draw_detection_overlay(self, image_bgr: np.ndarray, detection: DetectionResult) -> np.ndarray:
        if not detection.ok or detection.image_points.size == 0:
            return image_bgr
        overlay = image_bgr
        point_ids = None if detection.point_ids is None else np.asarray(detection.point_ids).reshape(-1)
        for idx, obs in enumerate(np.asarray(detection.image_points, dtype=np.float64)):
            obs_pt = tuple(int(round(v)) for v in obs)
            cv2.circle(overlay, obs_pt, 3, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(overlay, obs_pt, 6, (0, 120, 255), 1, cv2.LINE_AA)
            if point_ids is not None and idx < point_ids.size:
                cv2.putText(
                    overlay,
                    str(int(point_ids[idx])),
                    (obs_pt[0] + 5, obs_pt[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
        return overlay

    def _choose_next_target(self) -> int | None:
        if not self.remaining_target_indices:
            return None
        if not self.detected_descriptors:
            return self.remaining_target_indices[0]
        captured = np.vstack(self.detected_descriptors)
        best_idx = None
        best_score = float("-inf")
        for idx in self.remaining_target_indices:
            target_desc = quad_descriptor(self.target_bank[idx], self.image_size)
            score = float(np.min(np.linalg.norm(captured - target_desc[None, :], axis=1)))
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx

    def _write_capture(self, frame_bgr: np.ndarray, detection: DetectionResult) -> CaptureRecord:
        label = f"{self.stage}_{len(self.captures):03d}"
        image_path = self.captures_dir / f"{label}.jpg"
        cv2.imwrite(str(image_path), frame_bgr)
        thumb = cv2.resize(frame_bgr, (240, 135), interpolation=cv2.INTER_AREA)
        meta = {
            "stage": self.stage,
            "image_points": detection.image_points.tolist(),
            "object_points": detection.object_points.tolist(),
            "num_points": detection.num_points,
            "descriptor": quad_descriptor(detection.board_quad, self.image_size).tolist() if detection.board_quad is not None else None,
        }
        image_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="ascii")
        record = CaptureRecord(
            stage=self.stage,
            image_path=str(image_path),
            num_points=detection.num_points,
            descriptor=meta["descriptor"] or [],
            jpeg_base64=base64.b64encode(jpeg_bytes(thumb, quality=75)).decode("ascii"),
        )
        self.captures.append(record)
        if detection.board_quad is not None:
            self.detected_descriptors.append(quad_descriptor(detection.board_quad, self.image_size))
        return record

    def capture_current(self) -> None:
        with self.lock:
            if self.latest_frame_bgr is None or self.latest_detection is None or not self.latest_detection.ok:
                self.message = "Board detection required before capture."
                return
            self._write_capture(self.latest_frame_bgr, self.latest_detection)
            self._update_information_history()
            if self.stage == "seed" and len(self.captures) >= self.seed_required:
                seed_captures = [c for c in self.captures if c.stage == "seed"]
                self.seed_intrinsics, self.seed_rms = self._calibrate_from_captures(seed_captures)
                self.final_camera_matrix = camera_from_intrinsics(self.seed_intrinsics)[0].astype(float).tolist()
                self.stage = "guided"
                self.current_target_idx = self._choose_next_target()
                self.message = "Seed intrinsics estimated. Match the ghost board and capture the next target."
            elif self.stage == "guided":
                if self.current_target_idx in self.remaining_target_indices:
                    self.remaining_target_indices.remove(self.current_target_idx)
                if len(self.captures) >= self.total_captures_target:
                    self.final_intrinsics, self.final_rms = self._calibrate_from_captures(self.captures)
                    self.final_camera_matrix = camera_from_intrinsics(self.final_intrinsics)[0].astype(float).tolist()
                    self.stage = "done"
                    self.current_target_idx = None
                    self.message = f"Done! {len(self.captures)} captures collected. Final calibration complete."
                    self._write_session_summary()
                else:
                    self.current_target_idx = self._choose_next_target()
                    remaining = self.total_captures_target - len(self.captures)
                    self.message = f"Captured! {remaining} more to go — align with the ghost board."
            else:
                self.message = "Calibration session already complete."

    def calibrate_now(self) -> None:
        with self.lock:
            seed_captures = [c for c in self.captures if c.stage == "seed"]
            guided_captures = [c for c in self.captures if c.stage == "guided"]
            if self.stage == "seed":
                if len(seed_captures) < self.seed_required:
                    self.message = (
                        f"Need at least {self.seed_required} seed captures before calibration. "
                        f"Currently have {len(seed_captures)}."
                    )
                    return
                self.seed_intrinsics, self.seed_rms = self._calibrate_from_captures(seed_captures)
                self.final_camera_matrix = camera_from_intrinsics(self.seed_intrinsics)[0].astype(float).tolist()
                self.stage = "guided"
                self.current_target_idx = self._choose_next_target()
                self._update_information_history()
                self.message = "Seed calibration complete. Guided capture stage is ready."
                return

            if self.stage == "guided":
                if len(self.captures) < max(self.seed_required + 1, 6):
                    self.message = (
                        "Not enough total views to run final calibration yet. "
                        f"Currently have {len(self.captures)} captures."
                    )
                    return
                self.final_intrinsics, self.final_rms = self._calibrate_from_captures(self.captures)
                self.final_camera_matrix = camera_from_intrinsics(self.final_intrinsics)[0].astype(float).tolist()
                self.stage = "done"
                self.current_target_idx = None
                self._update_information_history()
                self.message = (
                    f"Final calibration complete from {len(seed_captures)} seed and {len(guided_captures)} guided captures."
                )
                self._write_session_summary()
                return

            if self.stage == "done":
                if self.final_intrinsics is None and len(self.captures) >= max(self.seed_required + 1, 6):
                    self.final_intrinsics, self.final_rms = self._calibrate_from_captures(self.captures)
                    self.final_camera_matrix = camera_from_intrinsics(self.final_intrinsics)[0].astype(float).tolist()
                    self._write_session_summary()
                    self.message = "Final calibration recomputed."
                else:
                    self.message = "Calibration is already complete."

    def skip_target(self) -> None:
        with self.lock:
            if self.stage != "guided":
                return
            if self.current_target_idx in self.remaining_target_indices:
                self.remaining_target_indices.remove(self.current_target_idx)
            self.current_target_idx = self._choose_next_target()
            self.message = "Skipped current target. New target projected."

    def reset(self) -> None:
        with self.lock:
            self.stage = "seed"
            self.message = "Session reset. Collect five diverse seed poses."
            self.captures.clear()
            self.detected_descriptors.clear()
            self.seed_intrinsics = None
            self.final_intrinsics = None
            self.seed_rms = None
            self.final_rms = None
            self.final_camera_matrix = None
            self.eig_history.clear()
            self.final_projection_overlay_previews.clear()
            self.remaining_target_indices = list(range(len(self.target_bank)))
            self.current_target_idx = None
            self.latest_alignment_error = None

    def _write_session_summary(self) -> None:
        payload = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "board_type": self.args.board_type,
            "board_rows": int(self.args.board_rows),
            "board_cols": int(self.args.board_cols),
            "board_square_size": float(self.args.board_square_size),
            "board_marker_size": float(self.args.board_marker_size),
            "seed_required": int(self.seed_required),
            "next_required": int(self.next_required),
            "image_size": [self.width, self.height],
            "seed_intrinsics": None if self.seed_intrinsics is None else self.seed_intrinsics.tolist(),
            "seed_rms_reprojection_error": self.seed_rms,
            "final_intrinsics": None if self.final_intrinsics is None else self.final_intrinsics.tolist(),
            "final_camera_matrix": self.final_camera_matrix,
            "final_rms_reprojection_error": self.final_rms,
            "eig_history": self.eig_history,
            "captures": [record.__dict__ for record in self.captures],
        }
        if self.final_intrinsics is not None:
            overlays = []
            for record in self.captures:
                overlay_report = self._render_final_projection_overlay(pathlib.Path(record.image_path), self.final_intrinsics)
                if overlay_report is not None:
                    overlays.append(overlay_report)
            payload["final_projection_overlays"] = overlays
            self.final_projection_overlay_previews = overlays
        (self.output_dir / "live_session_summary.json").write_text(json.dumps(payload, indent=2), encoding="ascii")

    def pose_plot_payload(self) -> dict[str, Any]:
        with self.lock:
            captures = list(self.captures)
            intrinsics = None if self.final_intrinsics is None else np.asarray(self.final_intrinsics, dtype=np.float64)
            if intrinsics is None and self.seed_intrinsics is not None:
                intrinsics = np.asarray(self.seed_intrinsics, dtype=np.float64)
            if intrinsics is None and len(captures) >= 3:
                try:
                    intrinsics, _ = self._calibrate_from_captures(captures)
                except Exception:
                    intrinsics = None
            latest_detection = self.latest_detection
            remaining_target_indices = list(self.remaining_target_indices)
            current_target_idx = self.current_target_idx
            pose_revision = {
                "stage": self.stage,
                "num_captures": len(self.captures),
                "current_target_idx": self.current_target_idx,
                "remaining_targets": len(self.remaining_target_indices),
                "latest_detection_ok": bool(self.latest_detection.ok) if self.latest_detection is not None else False,
                "latest_detection_points": int(self.latest_detection.num_points) if self.latest_detection is not None else 0,
                "seed_ready": self.seed_intrinsics is not None,
                "final_ready": self.final_intrinsics is not None,
            }
        points_seed_x, points_seed_y, points_seed_z = [], [], []
        points_guided_x, points_guided_y, points_guided_z = [], [], []
        labels_seed, labels_guided = [], []
        target_x, target_y, target_z, target_labels = [], [], [], []
        if intrinsics is not None:
            for idx, record in enumerate(captures):
                meta_path = pathlib.Path(record.image_path).with_suffix(".json")
                if not meta_path.exists():
                    continue
                meta = json.loads(meta_path.read_text(encoding="ascii"))
                object_points = np.asarray(meta["object_points"], dtype=np.float64)
                image_points = np.asarray(meta["image_points"], dtype=np.float64)
                tvec = self._estimate_board_origin_camera(object_points, image_points, intrinsics)
                if tvec is None:
                    continue
                if record.stage == "seed":
                    points_seed_x.append(float(tvec[0]))
                    points_seed_y.append(float(tvec[1]))
                    points_seed_z.append(float(tvec[2]))
                    labels_seed.append(str(idx + 1))
                else:
                    points_guided_x.append(float(tvec[0]))
                    points_guided_y.append(float(tvec[1]))
                    points_guided_z.append(float(tvec[2]))
                    labels_guided.append(str(idx + 1))
        traces = [
            go.Scatter3d(
                x=[0.0], y=[0.0], z=[0.0],
                mode="markers+text",
                text=["camera"],
                textposition="top center",
                marker=dict(size=6, color="red"),
                name="camera",
            )
        ]
        if points_seed_x:
            traces.append(
                go.Scatter3d(
                    x=points_seed_x, y=points_seed_y, z=points_seed_z,
                    mode="markers+text",
                    text=labels_seed,
                    textposition="top center",
                    marker=dict(size=5, color="#42a5f5"),
                    name="seed poses",
                )
            )
        if points_guided_x:
            traces.append(
                go.Scatter3d(
                    x=points_guided_x, y=points_guided_y, z=points_guided_z,
                    mode="markers+text",
                    text=labels_guided,
                    textposition="top center",
                    marker=dict(size=5, color="#66bb6a"),
                    name="guided poses",
                )
            )
        if intrinsics is not None and latest_detection is not None and latest_detection.ok:
            current_tvec = self._estimate_board_origin_camera(
                np.asarray(latest_detection.object_points, dtype=np.float64),
                np.asarray(latest_detection.image_points, dtype=np.float64),
                intrinsics,
            )
            if current_tvec is not None:
                traces.append(
                    go.Scatter3d(
                        x=[float(current_tvec[0])],
                        y=[float(current_tvec[1])],
                        z=[float(current_tvec[2])],
                        mode="markers+text",
                        text=["live board"],
                        textposition="top center",
                        marker=dict(size=7, color="#ffa726", symbol="diamond"),
                        name="live board",
                    )
                )
        if intrinsics is not None:
            candidate_indices = []
            if current_target_idx is not None:
                candidate_indices.append(current_target_idx)
            for idx in remaining_target_indices:
                if idx == current_target_idx:
                    continue
                candidate_indices.append(idx)
                if len(candidate_indices) >= min(self.next_required, 6):
                    break
            for rank, idx in enumerate(candidate_indices, start=1):
                tvec = self._estimate_target_origin_camera(self.target_bank[idx], intrinsics)
                if tvec is None:
                    continue
                target_x.append(float(tvec[0]))
                target_y.append(float(tvec[1]))
                target_z.append(float(tvec[2]))
                target_labels.append(f"next {rank}")
        if target_x:
            traces.append(
                go.Scatter3d(
                    x=target_x, y=target_y, z=target_z,
                    mode="markers+text",
                    text=target_labels,
                    textposition="top center",
                    marker=dict(size=5, color="#ab47bc", symbol="cross"),
                    name="next targets",
                )
            )
        layout = {
            "title": {"text": "Chosen Board Poses Relative to Camera"},
            "margin": {"l": 0, "r": 0, "b": 0, "t": 40},
            "uirevision": "poses3d-live",
            "scene": {
                "xaxis": {"title": "X (camera frame)", "showbackground": True, "backgroundcolor": "#161616", "gridcolor": "#444"},
                "yaxis": {"title": "Y (camera frame)", "showbackground": True, "backgroundcolor": "#161616", "gridcolor": "#444"},
                "zaxis": {"title": "Z (camera frame)", "showbackground": True, "backgroundcolor": "#161616", "gridcolor": "#444"},
                "aspectmode": "data",
            },
            "legend": {"orientation": "h", "y": 1.02, "x": 0.0},
        }
        fig = go.Figure(data=traces, layout=layout)
        payload = fig.to_plotly_json()
        payload["revision"] = pose_revision
        return payload

    def _is_diverse_enough(self, detection: DetectionResult) -> bool:
        """Return True if detection's pose is sufficiently different from recent captures."""
        if detection.board_quad is None:
            return False
        desc = quad_descriptor(detection.board_quad, self.image_size)
        if not self.detected_descriptors:
            return True
        # Require min distance from ALL existing captures to avoid any duplicates
        min_dist = float(min(np.linalg.norm(desc - d) for d in self.detected_descriptors))
        return min_dist >= 0.05

    def _capture_loop(self) -> None:
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            detection = self._detect_board(frame)
            annotated = frame.copy()
            alignment_error = None
            target_quad = None
            if self.stage == "guided" and self.current_target_idx is not None:
                target_quad = self.target_bank[self.current_target_idx]
                annotated = render_texture_overlay(annotated, self.texture_gray, target_quad, alpha=0.38)
                if detection.ok and detection.board_quad is not None:
                    alignment_error = float(np.mean(np.linalg.norm(order_quad(detection.board_quad) - order_quad(target_quad), axis=1)))
            elif self.stage == "done" and self.final_intrinsics is not None and detection.ok:
                annotated = self._overlay_live_reprojection(annotated, detection, self.final_intrinsics)
            if detection.ok:
                annotated = self._draw_detection_overlay(annotated, detection)
            if detection.ok and detection.board_quad is not None:
                quad = order_quad(detection.board_quad).astype(np.int32)
                cv2.polylines(annotated, [quad], True, (0, 255, 0), 2, cv2.LINE_AA)

            # ── Auto-capture ──────────────────────────────────────────────────
            auto_triggered = False
            auto_align_ready = False
            now_t = time.time()
            cooldown_remaining = max(0.0, self.auto_capture_cooldown_s - (now_t - self.last_auto_capture_time))
            if self.auto_capture_enabled and detection.ok and detection.num_points >= 20 and self.stage in ("seed", "guided"):
                cooldown_ok = cooldown_remaining < 0.01
                diverse_ok = self._is_diverse_enough(detection)
                if self.stage == "seed" and cooldown_ok and diverse_ok:
                    self.capture_current()
                    self.last_auto_capture_time = now_t
                    auto_triggered = True
                elif self.stage == "guided":
                    auto_align_ready = (alignment_error is not None and
                                        alignment_error < self.auto_capture_threshold_px)
                    if cooldown_ok and diverse_ok and auto_align_ready:
                        self.capture_current()
                        self.last_auto_capture_time = now_t
                        auto_triggered = True
            if auto_triggered:
                self.last_capture_flash_time = now_t

            # ── OSD ───────────────────────────────────────────────────────────
            # Green border when aligned and ready to capture
            if auto_align_ready and cooldown_remaining < 0.01:
                cv2.rectangle(annotated, (0, 0), (self.width - 1, self.height - 1), (0, 255, 0), 8, cv2.LINE_AA)
            elif auto_align_ready:
                cv2.rectangle(annotated, (0, 0), (self.width - 1, self.height - 1), (0, 200, 100), 4, cv2.LINE_AA)

            # Flash "CAPTURED!" overlay
            if now_t - self.last_capture_flash_time < 0.8:
                cv2.putText(annotated, "CAPTURED!", (self.width // 2 - 120, self.height // 2),
                            cv2.FONT_HERSHEY_DUPLEX, 2.0, (0, 255, 0), 4, cv2.LINE_AA)

            # Progress bar at bottom
            total_now = len(self.captures)
            bar_w = int(self.width * total_now / max(self.total_captures_target, 1))
            cv2.rectangle(annotated, (0, self.height - 12), (bar_w, self.height - 1), (0, 200, 80), -1)
            cv2.rectangle(annotated, (0, self.height - 12), (self.width - 1, self.height - 1), (80, 80, 80), 1)

            status_lines = [
                f"stage: {self.stage}  [{total_now}/{self.total_captures_target}]",
                f"detection: {detection.label}",
            ]
            if self.stage == "seed":
                seed_count = len([c for c in self.captures if c.stage == "seed"])
                status_lines.append(f"seed: {seed_count}/{self.seed_required}")
                if self.auto_capture_enabled:
                    if detection.ok:
                        status_lines.append("auto: hold board steady — capturing" if cooldown_remaining < 0.01
                                            else f"auto: next in {cooldown_remaining:.1f}s")
                    else:
                        status_lines.append("auto: show board to camera")
            elif self.stage == "guided":
                if alignment_error is not None:
                    thresh = self.auto_capture_threshold_px
                    color_hint = "ALIGNED" if alignment_error < thresh else f"align error: {alignment_error:.0f}px (need <{thresh:.0f})"
                    status_lines.append(color_hint)
                if self.auto_capture_enabled and cooldown_remaining > 0:
                    status_lines.append(f"auto cooldown: {cooldown_remaining:.1f}s")
            y = 28
            for line in status_lines:
                cv2.putText(annotated, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2, cv2.LINE_AA)
                y += 26
            with self.lock:
                self.latest_frame_bgr = frame.copy()
                self.latest_detection = detection
                self.latest_alignment_error = alignment_error
                self.latest_annotated_jpeg = jpeg_bytes(annotated, quality=82)
            time.sleep(max(1.0 / max(self.args.fps, 1.0), 0.03))

    def frame_jpeg(self) -> bytes:
        with self.lock:
            return self.latest_annotated_jpeg or jpeg_bytes(np.zeros((self.height, self.width, 3), dtype=np.uint8))

    def status_payload(self) -> dict[str, Any]:
        with self.lock:
            detection_status = self.latest_detection.label if self.latest_detection is not None else "starting"
            target_total = self.seed_required if self.stage == "seed" else self.next_required
            recent = [
                {
                    "label": f"{record.stage}:{pathlib.Path(record.image_path).name}",
                    "jpeg_base64": record.jpeg_base64,
                    "full_image_url": f"/api/file?path={pathlib.Path(record.image_path).as_posix()}",
                }
                for record in self.captures[-6:]
            ]
            return {
                "stage": self.stage,
                "num_captures": len(self.captures) if self.stage == "seed" else len([c for c in self.captures if c.stage == "guided"]),
                "target_total": target_total,
                "detection_status": detection_status,
                "alignment_error_px": self.latest_alignment_error,
                "seed_rms": self.seed_rms,
                "final_rms": self.final_rms,
                "current_min_eig": None if not self.eig_history else self.eig_history[-1]["min_eig"],
                "current_max_eig": None if not self.eig_history else self.eig_history[-1]["max_eig"],
                "eig_history": self.eig_history,
                "final_camera_matrix": self.final_camera_matrix,
                "camera_input_dims": [self.width, self.height],
                "message": self.message,
                "recent_captures": recent,
                "final_projection_overlays": self.final_projection_overlay_previews,
                "image_plane_coverage_jpeg_base64": self._render_image_plane_coverage_preview(),
                "final_residual_heatmap_jpeg_base64": self._render_final_residual_heatmap_preview(),
            }

    def close(self) -> None:
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cap.release()


def make_handler(session: LiveCalibrationSession):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except BrokenPipeError:
                return

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                body = HTML_PAGE.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except BrokenPipeError:
                    return
                return
            if parsed.path == "/frame.jpg":
                body = session.frame_jpeg()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except BrokenPipeError:
                    return
                return
            if parsed.path == "/api/poses3d_json":
                self._send_json(session.pose_plot_payload())
                return
            if parsed.path == "/api/status":
                self._send_json(session.status_payload())
                return
            if parsed.path == "/api/file":
                params = parse_qs(parsed.query)
                file_arg = params.get("path", [None])[0]
                if not file_arg:
                    self.send_error(HTTPStatus.BAD_REQUEST)
                    return
                file_path = pathlib.Path(file_arg).expanduser().resolve()
                if not under_root(file_path, session.output_dir):
                    self.send_error(HTTPStatus.FORBIDDEN)
                    return
                if not file_path.exists() or not file_path.is_file():
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                body = file_path.read_bytes()
                suffix = file_path.suffix.lower()
                content_type = "image/jpeg" if suffix in (".jpg", ".jpeg") else "application/octet-stream"
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except BrokenPipeError:
                    return
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/api/capture":
                session.capture_current()
                self._send_json({"ok": True})
                return
            if parsed.path == "/api/calibrate":
                session.calibrate_now()
                self._send_json({"ok": True})
                return
            if parsed.path == "/api/next_target":
                session.skip_target()
                self._send_json({"ok": True})
                return
            if parsed.path == "/api/reset":
                session.reset()
                self._send_json({"ok": True})
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--output-dir", default="live_calibration_demo_outputs")
    parser.add_argument("--board-type", choices=("charuco", "aprilgrid"), default="charuco")
    parser.add_argument("--board-rows", type=int, default=11)
    parser.add_argument("--board-cols", type=int, default=15)
    parser.add_argument("--board-square-size", type=float, default=0.021)
    parser.add_argument("--board-marker-size", type=float, default=0.016)
    parser.add_argument("--aruco-dict", default="DICT_4X4_250")
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--next-k", type=int, default=10)
    parser.add_argument("--aprilgrid-subpix-window", type=int, default=5)
    parser.add_argument("--aprilgrid-subpix-max-iters", type=int, default=80)
    parser.add_argument("--aprilgrid-subpix-epsilon", type=float, default=0.01)
    parser.add_argument("--aprilgrid-max-subpix-displacement2", type=float, default=9.0)
    parser.add_argument("--total-captures", type=int, default=None,
                        help="Total frames to collect (seed + guided). Default: seed-count + next-k.")
    parser.add_argument("--auto-capture", action="store_true", default=True,
                        help="Auto-capture when board aligns with ghost target (default: on)")
    parser.add_argument("--no-auto-capture", dest="auto_capture", action="store_false",
                        help="Disable auto-capture; require manual button press")
    parser.add_argument("--auto-threshold-px", type=float, default=60.0,
                        help="Alignment error (px) below which auto-capture fires (default: 60)")
    parser.add_argument("--auto-cooldown", type=float, default=2.5,
                        help="Minimum seconds between auto-captures (default: 2.5)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session = LiveCalibrationSession(args)
    server = ThreadingHTTPServer(("127.0.0.1", int(args.port)), make_handler(session))
    print(f"Live calibration demo running at http://127.0.0.1:{args.port}")
    print(f"Board type: {args.board_type}")
    print(f"Output dir: {pathlib.Path(args.output_dir).expanduser().resolve()}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        session.close()


if __name__ == "__main__":
    main()
