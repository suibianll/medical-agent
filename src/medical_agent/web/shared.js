"use strict";

const SVG_NS = "http://www.w3.org/2000/svg";

export function asText(value, fallback = "") {
  if (typeof value === "string") return value.trim();
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  return fallback;
}

export function asArray(value) {
  if (Array.isArray(value)) return value;
  if (value && typeof value === "object") return Object.values(value);
  return [];
}

export function unique(values) {
  return [...new Set(values.filter(Boolean))];
}

export function identifier(value) {
  if (value && typeof value === "object") {
    return asText(value.id || value.key || value.ref || value.node || value.name);
  }
  return asText(value).replace(/^\[|\]$/g, "");
}

export function referenceIds(value) {
  if (!value) return [];
  if (typeof value === "string") {
    const bracketed = [...value.matchAll(/\[([^\]\s]{1,80})\]/g)].map((match) => match[1]);
    return bracketed.length
      ? unique(bracketed.map(identifier))
      : unique(value.split(/[,，、\s]+/).map(identifier));
  }
  if (Array.isArray(value)) return unique(value.flatMap(referenceIds));
  if (typeof value === "object") return unique(Object.values(value).flatMap(referenceIds));
  return [];
}

export function truncateText(value, maximum = 240) {
  const text = asText(value).replace(/\s+/g, " ");
  return text.length > maximum ? `${text.slice(0, Math.max(1, maximum - 1))}…` : text;
}

export function splitSentences(value) {
  const text = asText(value).replace(/\s+/g, " ");
  if (!text) return [];
  const sentences = text.match(/[^。！？.!?]+[。！？.!?]+(?:[”’"')\]}）]+)?|[^。！？.!?]+$/g);
  return (sentences || [text]).map((sentence) => sentence.trim()).filter(Boolean);
}

export function makeElement(tagName, className = "", text = "") {
  const node = document.createElement(tagName);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}

export function makeSvgElement(tagName, attributes = {}) {
  const node = document.createElementNS(SVG_NS, tagName);
  Object.entries(attributes).forEach(([name, value]) => {
    node.setAttribute(name, String(value));
  });
  return node;
}

export function graphNodeType(value, terminalType = "answer") {
  const type = asText(value, "reference").toLowerCase();
  if (/(patient|case|record)/.test(type)) return "patient";
  if (/(knowledge|guideline|document|source)/.test(type)) return "knowledge";
  if (/(evidence|citation|fact)/.test(type)) return "evidence";
  if (/(task|plan)/.test(type)) return "task";
  if (/(claim|conclusion)/.test(type)) return "claim";
  if (/(report|answer|response)/.test(type)) return terminalType;
  return "reference";
}

export function graphNodeLabel(raw, fallback) {
  if (typeof raw === "string") return raw;
  const item = raw && typeof raw === "object" ? raw : {};
  return asText(item.label || item.name || item.text || item.title || item.content, fallback);
}

function graphLayer(type) {
  if (["patient", "knowledge", "evidence", "reference"].includes(type)) return 0;
  if (type === "task") return 1;
  if (type === "claim") return 2;
  return 3;
}

export function graphLayout(nodes, options) {
  const {
    width,
    minimumHeight,
    nodeHeight,
    rowGap,
    topPadding,
    xByLayer,
    widths
  } = options;
  const columns = [[], [], [], []];
  nodes.forEach((node) => columns[graphLayer(node.type)].push(node));
  const maximum = Math.max(1, ...columns.map((column) => column.length));
  const height = Math.max(minimumHeight, maximum * (nodeHeight + rowGap) + topPadding * 2);
  const positions = new Map();
  columns.forEach((column, layer) => {
    const occupied = column.length * nodeHeight + Math.max(0, column.length - 1) * rowGap;
    let y = Math.max(topPadding, (height - occupied) / 2) + nodeHeight / 2;
    column.forEach((node) => {
      positions.set(node.id, {
        x: xByLayer[layer],
        y,
        width: widths[node.type] || widths.default,
        height: nodeHeight
      });
      y += nodeHeight + rowGap;
    });
  });
  return { width, height, positions };
}

export function getErrorMessage(payload, fallback) {
  if (typeof payload === "string") return payload;
  if (!payload || typeof payload !== "object") return fallback;
  const error = payload.error;
  if (typeof error === "string") return error;
  if (error && typeof error === "object") {
    return asText(error.message || error.detail || error.code, fallback);
  }
  return asText(payload.message || payload.detail, fallback);
}

export async function requestJson(url, options = {}) {
  const { allowEmpty = false, ...fetchOptions } = options;
  const { headers: customHeaders, ...requestOptions } = fetchOptions;
  const response = await fetch(url, {
    ...requestOptions,
    headers: { Accept: "application/json", ...(customHeaders || {}) }
  });
  const raw = await response.text();
  let body = null;
  if (raw) {
    try {
      body = JSON.parse(raw);
    } catch {
      body = raw;
    }
  }
  if (!response.ok) {
    throw new Error(getErrorMessage(body, `请求失败（HTTP ${response.status}）`));
  }
  if ((!body || typeof body !== "object") && !allowEmpty) {
    throw new Error("服务端未返回有效数据。");
  }
  return body || {};
}
