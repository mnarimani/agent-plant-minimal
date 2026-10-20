/* Offline Model Explorer: pure SVG, no network assets and no model execution. */
"use strict";

// Same-origin Streamlit iframes can read the app's actual rendered theme.
// Outside the app (or across origins), CSS follows the system preference.
function followAppTheme() {
  if (window.parent === window) return;
  try {
    const host = window.parent.document;
    let queued = false;
    function sync() {
      queued = false;
      const app = host.querySelector('[data-testid="stApp"]');
      if (!app) return;
      const color = window.parent.getComputedStyle(app).backgroundColor;
      const channels = color.match(/[\d.]+/g);
      if (!channels || channels.length < 3 || (channels.length > 3 && Number(channels[3]) === 0)) return;
      const [r, g, b] = channels.slice(0, 3).map(Number);
      const theme = .2126 * r + .7152 * g + .0722 * b < 128 ? "dark" : "light";
      if (document.documentElement.dataset.theme !== theme) document.documentElement.dataset.theme = theme;
    }
    function schedule() {
      if (!queued) { queued = true; requestAnimationFrame(sync); }
    }
    const observer = new MutationObserver(schedule);
    observer.observe(host.documentElement, {
      subtree: true, attributes: true, attributeFilter: ["class", "style", "data-theme"], childList: true,
    });
    sync();
    window.addEventListener("pagehide", () => observer.disconnect(), {once: true});
  } catch (_) { /* Cross-origin embeds retain the standalone CSS theme. */ }
}
followAppTheme();
const packageData = JSON.parse(document.getElementById("model-data").textContent);
let viewName = packageData.views ? "overview" : "full";
let model = packageData.views ? {...packageData, ...packageData.views[viewName]} : packageData;
const ns = "http://www.w3.org/2000/svg";
const graph = document.getElementById("graph"), viewport = document.getElementById("viewport");
const inspector = document.getElementById("inspector"), rendered = new Map();
let selected = null, panel = "inputs", layoutName = "native", diagramDrawn = false;
let view = [], fitMode = true, drag = null;

function html(tag, text, parent, cls) {
  const result = document.createElement(tag);
  if (text !== undefined) result.textContent = text;
  if (cls) result.className = cls;
  if (parent) parent.appendChild(result);
  return result;
}
function svg(tag, attrs, parent, text) {
  const result = document.createElementNS(ns, tag);
  for (const [key, value] of Object.entries(attrs)) result.setAttribute(key, value);
  if (text !== undefined) result.textContent = text;
  if (parent) parent.appendChild(result);
  return result;
}
function format(value) { return typeof value === "string" ? value : JSON.stringify(value); }
function field(parent, name, value) { html("dt", name, parent); html("dd", format(value), parent); }
function mapping(parent, symbol, name) {
  const row = html("div", undefined, parent, "mapping"); html("strong", symbol, row); html("span", name, row);
}
function kind(node) {
  if (node.type === "Inport") return "input";
  if (node.type === "Outport") return "output";
  if (["Integrator", "TransferFcn", "StateflowChart"].includes(node.type)) return "state";
  return "compute";
}
function showBlock() {
  html("h2", "Block properties", inspector);
  if (selected) html("h3", selected.name, inspector);
  if (!selected) {
    html("p", "Open Diagram and select a block to inspect its parameters and connected signals.", inspector, "hint");
    html("h3", "Navigation", inspector);
    html("p", "Drag the canvas to pan. Use + / − or Ctrl + wheel to zoom. Fit restores the whole diagram.", inspector, "hint");
    html("h3", "Layouts", inspector);
    html("p", "Block positions, dimensions, orientation and saved line segments come from the original system XML. From and Goto retain their original tag routing.", inspector, "hint");
    return;
  }
  html("span", selected.type === "MatlabFunction" ? "MATLAB Function" : selected.type, inspector, "pill");
  if (selected.child_view) {
    const open = html("button", "Open subsystem", inspector);
    open.addEventListener("click", () => switchView(selected.child_view));
  }
  const fields = html("dl", undefined, inspector);
  const source = selected.source_layout || {};
  if (source.rotation) field(fields, "Original block rotation", `${source.rotation}°`);
  if (source.mirror) field(fields, "Original orientation", "Mirrored");
  for (const [name, value] of Object.entries(selected.parameters)) {
    if (["script", "expression", "chart_xml"].includes(name)) continue;
    field(fields, name.replaceAll("_", " "), value);
  }
  const code = selected.parameters.script ?? selected.parameters.expression ?? selected.parameters.chart_xml;
  if (code !== undefined) { html("h3", "Code / equation", inspector); html("pre", code, inspector); }
  const states = (model.dynamics?.states || []).filter(state => state.block_id === selected.id);
  if (states.length) {
    html("h3", "Dynamic states", inspector);
    states.forEach(state => mapping(inspector, state.symbol, `initial = ${state.initial_expression}`));
  }
  html("h3", "Connected signals", inspector);
  const connections = model.edges.filter(edge => edge.source.block_id === selected.id || edge.destination.block_id === selected.id);
  const byId = new Map(model.nodes.map(node => [node.id, node]));
  for (const edge of connections) {
    const from = byId.get(edge.source.block_id), to = byId.get(edge.destination.block_id);
    html("p", `${from.name} → ${to.name}`, inspector, "hint");
  }
}
function showSimulation() {
  html("h2", "Simulation settings", inspector);
  html("p", "Defaults extracted from the model. Use the Simulation tab to change settings and run comparisons.", inspector, "hint");
  const fields = html("dl", undefined, inspector);
  for (const [key, value] of Object.entries(model.settings)) field(fields, key.replaceAll("_", " "), value);
}
function showPorts(direction) {
  html("h2", direction === "inputs" ? "Model inputs" : "Model outputs", inspector);
  const values = model[direction] || [], prefix = direction === "inputs" ? "u" : "y";
  if (!values.length) html("p", `No root ${direction}.`, inspector, "hint");
  values.forEach((name, index) => mapping(inspector, `${prefix}${index + 1}`, name));
}
function showParameters() {
  html("h2", "Model parameters", inspector);
  const parameters = model.parameters || [];
  if (!parameters.length) html("p", "No assigned local parameters were extracted.", inspector, "hint");
  const fields = html("dl", undefined, inspector);
  parameters.forEach(parameter => field(fields, parameter.block_name, parameter.name));
}
function showDynamics() {
  html("h2", "Dynamic model", inspector);
  const report = model.dynamics;
  if (!report) { html("p", "Reconvert the model to extract its dynamic equations.", inspector, "hint"); return; }
  if (report.status !== "success") {
    html("p", "Dynamic equations could not be extracted.", inspector, "error");
    (report.errors || []).forEach(error => html("p", `${error.block_name || "Model"}: ${error.message}`, inspector, "error"));
    return;
  }
  html("span", report.time_dependent ? "xdot = f(t, x, u) · y = g(t, x, u)" : "xdot = f(x, u) · y = g(x, u)", inspector, "pill");
  html("h3", "State vector", inspector);
  if (!report.states.length) html("p", "Algebraic model; the state vector is empty.", inspector, "hint");
  report.states.forEach(state => mapping(inspector, state.symbol, `${state.block_name}; initial = ${state.initial_expression}`));
  html("h3", "Input / output mapping", inspector);
  [...report.inputs, ...report.outputs].forEach(item => mapping(inspector, item.symbol, item.name));
  html("h3", "Equations", inspector);
  html("pre", [...report.f.map(item => `d${item.state}/dt = ${item.expression}`), ...report.g.map(item => `${item.output} = ${item.expression}`)].join("\n"), inspector);
  for (const realization of report.transfer_realizations || []) {
    const details = html("details", undefined, inspector);
    html("summary", `${realization.block_name} · state matrices`, details);
    html("pre", JSON.stringify({states: realization.states, A: realization.A, B: realization.B, C: realization.C, D: realization.D}, null, 2), details);
  }
}
function switchPanel(name) {
  panel = name; inspector.replaceChildren();
  const isDiagram = name === "diagram";
  if (!isDiagram) {
    document.body.classList.remove("focus");
    document.getElementById("focus").setAttribute("aria-pressed", "false");
  }
  document.querySelector(".canvas").hidden = !isDiagram;
  document.querySelector("main").classList.toggle("details-only", !isDiagram);
  if (isDiagram && !diagramDrawn) { diagramDrawn = true; draw(); }
  document.querySelectorAll("[data-panel]").forEach(button => button.setAttribute("aria-pressed", String(button.dataset.panel === name)));
  ({diagram: showBlock, simulation: showSimulation, inputs: () => showPorts("inputs"), outputs: () => showPorts("outputs"), parameters: showParameters, dynamics: showDynamics})[name]();
  if (isDiagram && fitMode) fit();
}
function selectNode(node, center = false) {
  selected = node;
  const related = new Set([node.id]);
  model.edges.forEach(edge => { if (edge.source.block_id === node.id || edge.destination.block_id === node.id) { related.add(edge.source.block_id); related.add(edge.destination.block_id); } });
  for (const [id, group] of rendered) {
    group.classList.toggle("selected", id === node.id); group.classList.toggle("dimmed", !related.has(id));
    group.setAttribute("aria-pressed", String(id === node.id));
  }
  graph.querySelectorAll(".wire").forEach(path => {
    const edge = model.edges[Number(path.dataset.edge)];
    const connected = edge.source.block_id === node.id || edge.destination.block_id === node.id;
    path.classList.toggle("related", connected); path.classList.toggle("dimmed", !connected);
  });
  if (center) {
    const geometry = model.layouts[layoutName].nodes[node.id];
    fitMode = false; const width = Math.min(view[2], 520), height = width * viewport.clientHeight / viewport.clientWidth;
    view = [geometry.x + geometry.width / 2 - width / 2, geometry.y + geometry.height / 2 - height / 2, width, height]; applyView();
  }
  switchPanel("diagram");
}
function polynomial(coefficients) {
  const terms = [];
  (coefficients || [1]).forEach((coefficient, index, array) => {
    if (!coefficient) return;
    const power = array.length - index - 1, absolute = Math.abs(coefficient);
    const number = power && absolute === 1 ? "" : String(Number(absolute.toPrecision(4)));
    const term = number + (power ? (power === 1 ? "s" : `s${power === 2 ? "²" : power === 3 ? "³" : "^" + power}`) : "");
    terms.push((terms.length ? coefficient < 0 ? " − " : " + " : coefficient < 0 ? "−" : "") + term);
  });
  return terms.join("") || "0";
}
function icon(node, geometry, group) {
  const {x, y, width: w, height: h} = geometry, cx = x + w / 2, cy = y + h / 2;
  const text = (value, yy = cy + 5, size = 16) => svg("text", {x: cx, y: yy, class: "symbol", style: `font-size:${size}px`}, group, value);
  if (node.type === "TransferFcn" && !node.parameters.numerator) {
    text("Transfer Fcn", cy + 5, 12);
  } else if (["TransferFcn", "Integrator"].includes(node.type)) {
    const numerator = node.type === "Integrator" ? "1" : polynomial(node.parameters.numerator);
    const denominator = node.type === "Integrator" ? "s" : polynomial(node.parameters.denominator);
    [text(numerator, cy - 9, 15), text(denominator, cy + 21, 15)].forEach(item => {
      if (item.textContent.length * 8 > w - 24) { item.setAttribute("textLength", w - 24); item.setAttribute("lengthAdjust", "spacingAndGlyphs"); }
    });
    svg("line", {x1: x + 13, x2: x + w - 13, y1: cy, y2: cy, class: "symbol-line"}, group);
  } else if (node.type === "MatlabFunction") { text("f(u)", cy + 1, 22); text("MATLAB", cy + 23, 10); }
  else if (node.type === "Sum") { if (!geometry.ports.input.length) text("Σ", cy + 6, 24); }
  else if (node.type === "Subsystem") { text("Subsystem", cy - 2, 15); text("Open", cy + 21, 11); }
  else if (["From", "Goto", "GotoTagVisibility"].includes(node.type)) text(`[${node.parameters.tag}]`, cy + 5, 14);
  else if (node.type === "Inport") text(`u${node.source_number || (model.inputs || []).indexOf(node.parameters.input_key) + 1}`);
  else if (node.type === "Outport") text(`y${node.source_number || (model.outputs || []).indexOf(node.name) + 1}`);
  else if (node.type === "Gain") text(format(node.parameters.gain ?? node.parameters.source_parameters?.Gain ?? "Gain"));
  else if (node.type === "Constant") text(format(node.parameters.value ?? node.parameters.source_parameters?.Value ?? "Constant"));
  else if (node.type === "Mux") text("⋮", cy + 6, 26);
  else if (node.type === "Step") svg("path", {d: `M ${x+15} ${cy+12} H ${cx} V ${cy-12} H ${x+w-15}`, class: "symbol-line"}, group);
  else if (node.type === "Saturate") svg("path", {d: `M ${x+16} ${cy+14} H ${cx-12} L ${cx+12} ${cy-14} H ${x+w-16}`, class: "symbol-line"}, group);
  else text(node.type === "Fcn" ? "f(u)" : node.type, cy + 5, 12);
}
function draw() {
  if (!model.nodes.length) {
    graph.replaceChildren(); rendered.clear();
    svg("text", {x: 200, y:100, class:"symbol", style:"font-size:14px"}, graph, "No blocks are stored in this XML system.");
    fit(); return;
  }
  graph.replaceChildren(); rendered.clear();
  const layout = model.layouts[layoutName], defs = svg("defs", {}, graph);
  const pattern = svg("pattern", {id: "grid", width: 20, height: 20, patternUnits: "userSpaceOnUse"}, defs);
  svg("circle", {cx: 1, cy: 1, r: .8, fill: "var(--grid)"}, pattern);
  const marker = svg("marker", {id: "arrow", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: "auto"}, defs);
  svg("path", {d: "M 0 0 L 10 5 L 0 10 z", fill: "var(--wire)"}, marker);
  svg("rect", {x: layout.bounds[0], y: layout.bounds[1], width: layout.bounds[2], height: layout.bounds[3], fill: "var(--surface)"}, graph);
  svg("rect", {x: layout.bounds[0], y: layout.bounds[1], width: layout.bounds[2], height: layout.bounds[3], fill: "url(#grid)", "pointer-events": "none"}, graph);
  const wireLayer = svg("g", {}, graph), junctions = new Map();
  for (const route of layout.edges) {
    const edge = model.edges[route.index];
    const path = svg("path", {d: route.points.map((point, index) => `${index ? "L" : "M"} ${point.join(" ")}`).join(" "), class: "wire" + (edge.kind === "signal" ? "" : " control"), "marker-end": "url(#arrow)", "data-edge": route.index}, wireLayer);
    svg("title", {}, path, `${model.nodes.find(n => n.id === edge.source.block_id).name} → ${model.nodes.find(n => n.id === edge.destination.block_id).name}`);
    for (const point of route.points.slice(1, -1)) {
      const key = `${edge.source.block_id}:${edge.source.port}:${point.join(",")}`;
      const item = junctions.get(key) || {point, count: 0}; item.count++; junctions.set(key, item);
    }
  }
  for (const {point, count} of junctions.values()) if (count > 1) svg("circle", {cx: point[0], cy: point[1], r: 3, class: "junction"}, graph);
  for (const node of model.nodes) {
    const g = layout.nodes[node.id], group = svg("g", {class: "node", "data-block-id": node.id, "data-kind": kind(node), role: "button", tabindex: 0, "aria-label": node.name, "aria-pressed": "false"}, graph);
    const {x, y, width: w, height: h} = g;
    svg("title", {}, group, `${node.name} · ${node.type}`);
    if (node.type === "Sum" && node.source_layout?.icon_shape !== "rectangular") svg("circle", {cx: x+w/2, cy: y+h/2, r: w/2, class: "body"}, group);
    else if (node.type === "From") svg("polygon", {points:`${x},${y} ${x+w-12},${y} ${x+w},${y+h/2} ${x+w-12},${y+h} ${x},${y+h}`,class:"body"}, group);
    else if (node.type === "Goto") svg("polygon", {points:`${x+12},${y} ${x+w},${y} ${x+w},${y+h} ${x+12},${y+h} ${x},${y+h/2}`,class:"body"}, group);
    else if (node.type === "Gain") {
      const side = g.ports.output[0]?.side || "right";
      const points = {right:`${x},${y} ${x+w},${y+h/2} ${x},${y+h}`,
        left:`${x+w},${y} ${x},${y+h/2} ${x+w},${y+h}`,
        top:`${x},${y+h} ${x+w/2},${y} ${x+w},${y+h}`,
        bottom:`${x},${y} ${x+w/2},${y+h} ${x+w},${y}`}[side];
      svg("polygon", {points, class:"body"}, group);
    }
    else svg("rect", {x, y, width: w, height: h, rx: ["Inport", "Outport"].includes(node.type) ? h/2 : 7, class: "body"}, group);
    icon(node, g, group);
    const labelX = g.label_side === "right" ? x+w+12 : x+w/2;
    const labelY = g.label_side === "top" ? y-12 : g.label_side === "right" ? y+h/2+4 : y+h+20;
    const label = svg("text", {x: labelX, y: labelY, class: "node-label", "text-anchor": g.label_side === "right" ? "start" : "middle"}, group, g.label);
    if (g.label.length * 7 > g.label_width) { label.setAttribute("textLength", g.label_width); label.setAttribute("lengthAdjust", "spacingAndGlyphs"); }
    for (const direction of ["input", "output"]) for (const point of g.ports[direction]) {
      svg("circle", {cx: point.x, cy: point.y, r: 2.6, class: "port"}, group);
      if (node.type === "Sum" && direction === "input") {
        const sign = node.parameters.signs?.[point.port] < 0 ? "−" : "+";
        const offset = {left: [8, 4], right: [-8, 4], top: [0, 11], bottom: [0, -6]}[point.side];
        svg("text", {x: point.x+offset[0], y: point.y+offset[1], class: "symbol", style: "font-size:11px"}, group, sign);
      }
    }
    group.addEventListener("click", () => selectNode(node));
    if (node.child_view) group.addEventListener("dblclick", () => switchView(node.child_view));
    group.addEventListener("keydown", event => { if (["Enter", " "].includes(event.key)) { event.preventDefault(); selectNode(node); } });
    rendered.set(node.id, group);
  }
  if (selected) selectNode(selected);
  fit();
}
function applyView() {
  graph.setAttribute("viewBox", view.join(" "));
  document.getElementById("zoom-level").textContent = `${Math.round(100 * viewport.clientWidth / view[2])}% · drag to pan`;
}
function fit() {
  if (!viewport.clientWidth || !viewport.clientHeight) return;
  const bounds = model.layouts[layoutName].bounds, ratio = viewport.clientWidth / Math.max(1, viewport.clientHeight);
  const w = Math.max(bounds[2], bounds[3] * ratio), h = w / ratio;
  view = [bounds[0] - (w-bounds[2])/2, bounds[1] - (h-bounds[3])/2, w, h]; fitMode = true; applyView();
}
function zoom(factor, clientX, clientY) {
  const rect = viewport.getBoundingClientRect();
  const rx = clientX === undefined ? .5 : (clientX-rect.left)/rect.width, ry = clientY === undefined ? .5 : (clientY-rect.top)/rect.height;
  const width = Math.max(90, Math.min(model.layouts[layoutName].bounds[2]*8, view[2]/factor));
  const height = width * rect.height/rect.width;
  view = [view[0]+rx*(view[2]-width), view[1]+ry*(view[3]-height), width, height]; fitMode = false; applyView();
}
document.getElementById("model-name").textContent = packageData.model_name;
document.title = `${packageData.model_name} — Model Explorer`;
const layoutSelect = document.getElementById("layout");
const systemSelect = document.getElementById("system");
for (const [key, section] of Object.entries(packageData.views || {full: {title: "Full model"}})) {
  const option = html("option", section.title, systemSelect); option.value = key;
}
systemSelect.value = viewName;
function updateLayouts() {
  layoutSelect.replaceChildren();
  for (const name of Object.keys(model.layouts)) {
    const option = html("option", name === "native" ? "SLX layout" : "Signal flow", layoutSelect); option.value = name;
  }
  layoutName = "native"; layoutSelect.value = layoutName;
  document.getElementById("graph-summary").textContent = `${model.nodes.length} visible blocks · ${model.edges.length} connections`;
}
function switchView(key) {
  if (!packageData.views?.[key]) return;
  viewName = key; systemSelect.value = key;
  model = {...packageData, ...packageData.views[key]}; selected = null;
  updateLayouts(); draw(); switchPanel("diagram");
}
updateLayouts();
systemSelect.addEventListener("change", () => switchView(systemSelect.value));
layoutSelect.addEventListener("change", () => { layoutName = layoutSelect.value; draw(); });

document.querySelectorAll("[data-panel]").forEach(button => button.addEventListener("click", () => switchPanel(button.dataset.panel)));
document.getElementById("fit").addEventListener("click", fit);
document.getElementById("zoom-in").addEventListener("click", () => zoom(1.25));
document.getElementById("zoom-out").addEventListener("click", () => zoom(.8));
document.getElementById("focus").addEventListener("click", event => { document.body.classList.toggle("focus"); event.target.setAttribute("aria-pressed", String(document.body.classList.contains("focus"))); fit(); });
viewport.addEventListener("wheel", event => { if (event.ctrlKey || event.metaKey) { event.preventDefault(); zoom(event.deltaY < 0 ? 1.12 : 1/1.12, event.clientX, event.clientY); } }, {passive: false});
viewport.addEventListener("pointerdown", event => { if (event.target.closest(".node")) return; drag = {x: event.clientX, y: event.clientY, view: [...view]}; viewport.setPointerCapture(event.pointerId); });
viewport.addEventListener("pointermove", event => { if (!drag) return; fitMode = false; view = [drag.view[0]-(event.clientX-drag.x)*drag.view[2]/viewport.clientWidth, drag.view[1]-(event.clientY-drag.y)*drag.view[3]/viewport.clientHeight, drag.view[2], drag.view[3]]; applyView(); });
viewport.addEventListener("pointerup", () => { drag = null; }); viewport.addEventListener("pointercancel", () => { drag = null; });
new ResizeObserver(() => { if (panel === "diagram") { if (fitMode) fit(); else applyView(); } }).observe(viewport);
const search = document.getElementById("search"), matches = document.getElementById("matches");
search.addEventListener("input", () => {
  matches.replaceChildren(); const value = search.value.trim().toLowerCase();
  const found = value ? model.nodes.filter(node => node.name.toLowerCase().includes(value) || node.type.toLowerCase().includes(value)) : [];
  found.forEach(node => { const button = html("button", node.name, matches); button.addEventListener("click", () => { selectNode(node, true); search.value = ""; matches.hidden = true; }); });
  if (value && !found.length) html("p", "No matching blocks", matches, "hint"); matches.hidden = !value;
});
search.addEventListener("keydown", event => { if (event.key === "Escape") { matches.hidden = true; search.value = ""; } });
const overlay = document.getElementById("preview-overlay"), previewButton = document.getElementById("source-preview");
previewButton.disabled = !model.source_preview;
if (model.source_preview) document.getElementById("thumbnail").src = model.source_preview;
previewButton.addEventListener("click", () => { overlay.hidden = false; document.getElementById("close-preview").focus(); });
function closePreview() { overlay.hidden = true; previewButton.focus(); }
document.getElementById("close-preview").addEventListener("click", closePreview);
overlay.addEventListener("click", event => { if (event.target === overlay) closePreview(); });
document.addEventListener("keydown", event => { if (event.key === "Escape" && !overlay.hidden) closePreview(); });
document.getElementById("save-svg").addEventListener("click", () => {
  const copy = graph.cloneNode(true), bounds = model.layouts[layoutName].bounds;
  copy.removeAttribute("id");
  copy.setAttribute("viewBox", bounds.join(" ")); copy.setAttribute("width", bounds[2]); copy.setAttribute("height", bounds[3]);
  const sheet = [...document.styleSheets].map(sheet => [...sheet.cssRules].map(rule => rule.cssText).join("\n")).join("\n");
  svg("style", {}, copy, sheet);
  const url = URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(copy)], {type: "image/svg+xml;charset=utf-8"}));
  const link = html("a", undefined); link.href = url; link.download = `${model.model_name}-${layoutName}.svg`; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
switchPanel("inputs");
