(() => {
  "use strict";

  const KNOWLEDGE_URL = "/api/knowledge";
  const KNOWLEDGE_IMPORT_URL = "/api/knowledge/import";
  const CHAT_URL = "/api/chat";
  const MAX_FILE_BYTES = 500_000;
  const MAX_GRAPH_NODES = 48;
  const SVG_NS = "http://www.w3.org/2000/svg";

  const elements = {
    knowledgeForm: document.getElementById("knowledge-form"),
    knowledgeName: document.getElementById("knowledge-name"),
    knowledgeFile: document.getElementById("knowledge-file"),
    knowledgeContent: document.getElementById("knowledge-content"),
    knowledgeError: document.getElementById("knowledge-error"),
    importKnowledge: document.getElementById("import-knowledge"),
    refreshKnowledge: document.getElementById("refresh-knowledge"),
    knowledgeList: document.getElementById("knowledge-list"),
    knowledgeCount: document.getElementById("knowledge-count"),
    knowledgeState: document.getElementById("knowledge-state"),
    chatForm: document.getElementById("chat-form"),
    chatLog: document.getElementById("chat-log"),
    chatMessage: document.getElementById("chat-message"),
    patientContext: document.getElementById("patient-context"),
    sendMessage: document.getElementById("send-message"),
    chatStatus: document.getElementById("chat-status"),
    loadDemo: document.getElementById("load-demo"),
    graphCaption: document.getElementById("graph-caption"),
    evidenceGraph: document.getElementById("evidence-graph"),
    runStatus: document.getElementById("run-status"),
    selectedEvidence: document.getElementById("selected-evidence"),
    evidenceList: document.getElementById("evidence-list"),
    evidenceCount: document.getElementById("evidence-count")
  };

  const syntheticDemo = {
    patientRecord: "合成病例：68 岁，近两周乏力。记录显示 eGFR 约 42 mL/min/1.73m²，既往有药物过敏史，近期肾功能尚未复查。该内容仅用于界面演示。",
    message: "请基于已导入资料，梳理这个合成病例中需要优先核实的用药安全信息，并列出引用依据。"
  };

  let chatHistory = [];
  let knowledgeDocuments = [];
  let graphState = emptyGraphState();
  let activeChatRequest = false;
  let assistantTurnSequence = 0;

  function emptyGraphState() {
    return {
      nodes: [],
      edges: [],
      nodeById: new Map(),
      evidenceById: new Map(),
      claimsById: new Map(),
      activeTurn: null
    };
  }

  function asText(value, fallback = "") {
    if (typeof value === "string") return value.trim();
    if (typeof value === "number" || typeof value === "boolean") return String(value);
    return fallback;
  }

  function asArray(value) {
    if (Array.isArray(value)) return value;
    if (value && typeof value === "object") return Object.values(value);
    return [];
  }

  function unique(values) {
    return [...new Set(values.filter(Boolean))];
  }

  function identifier(value) {
    if (value && typeof value === "object") {
      return asText(value.id || value.key || value.ref || value.node || value.name);
    }
    return asText(value).replace(/^\[|\]$/g, "");
  }

  function extractReferenceIds(value) {
    if (!value) return [];
    if (typeof value === "string") {
      const bracketed = [...value.matchAll(/\[([^\]\s]{1,80})\]/g)].map((match) => match[1]);
      if (bracketed.length) return unique(bracketed.map(identifier));
      return unique(value.split(/[,，、\s]+/).map(identifier));
    }
    if (Array.isArray(value)) return unique(value.flatMap(extractReferenceIds));
    if (typeof value === "object") return unique(Object.values(value).flatMap(extractReferenceIds));
    return [];
  }

  function getErrorMessage(payload, fallback) {
    if (typeof payload === "string") return payload;
    if (!payload || typeof payload !== "object") return fallback;
    const error = payload.error;
    if (typeof error === "string") return error;
    if (error && typeof error === "object") return asText(error.message || error.detail || error.code, fallback);
    return asText(payload.message || payload.detail, fallback);
  }

  async function requestJson(url, options = {}) {
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

  function makeElement(tagName, className = "", text = "") {
    const node = document.createElement(tagName);
    if (className) node.className = className;
    if (text) node.textContent = text;
    return node;
  }

  function makeSvgElement(tagName, attributes = {}) {
    const node = document.createElementNS(SVG_NS, tagName);
    Object.entries(attributes).forEach(([name, value]) => {
      node.setAttribute(name, String(value));
    });
    return node;
  }

  function normalizeDocuments(payload) {
    const rawDocuments = Array.isArray(payload)
      ? payload
      : asArray(payload.documents || payload.items || payload.knowledge || payload.data);

    return rawDocuments.map((raw, index) => {
      if (typeof raw === "string") {
        return { id: `document-${index + 1}`, name: raw, meta: "已导入资料", preview: "" };
      }
      const item = raw && typeof raw === "object" ? raw : {};
      const name = asText(item.name || item.title || item.filename || item.source, `未命名资料 ${index + 1}`);
      const metaBits = [];
      const chunks = item.chunks ?? item.chunk_count ?? item.chunkCount;
      if (chunks != null) metaBits.push(`${chunks} 个片段`);
      if (item.updated_at || item.updatedAt || item.created_at || item.createdAt) metaBits.push("已导入");
      return {
        id: identifier(item.id || item.document_id || item.documentId) || `document-${index + 1}`,
        name,
        meta: metaBits.join(" · ") || "已导入资料",
        preview: asText(item.preview || item.excerpt || item.content || item.text)
      };
    });
  }

  function renderKnowledgeDocuments() {
    elements.knowledgeList.replaceChildren();
    elements.knowledgeCount.textContent = String(knowledgeDocuments.length);

    if (!knowledgeDocuments.length) {
      elements.knowledgeState.hidden = false;
      elements.knowledgeState.textContent = "尚无资料。导入一份脱敏文本后即可开始检索。";
      return;
    }

    elements.knowledgeState.hidden = true;
    knowledgeDocuments.forEach((document) => {
      const item = makeElement("li");
      const name = makeElement("strong", "knowledge-document-name", document.name);
      name.title = document.name;
      const meta = makeElement("span", "knowledge-document-meta", document.meta);
      if (document.preview) meta.title = document.preview;
      item.append(name, meta);
      elements.knowledgeList.append(item);
    });
  }

  async function loadKnowledge() {
    elements.refreshKnowledge.disabled = true;
    elements.knowledgeState.hidden = false;
    elements.knowledgeState.textContent = "正在读取知识库…";
    try {
      const response = await requestJson(KNOWLEDGE_URL);
      knowledgeDocuments = normalizeDocuments(response);
      renderKnowledgeDocuments();
    } catch (error) {
      knowledgeDocuments = [];
      elements.knowledgeList.replaceChildren();
      elements.knowledgeCount.textContent = "0";
      elements.knowledgeState.hidden = false;
      elements.knowledgeState.textContent = `暂时无法读取知识库：${asText(error.message, "未知错误")}`;
    } finally {
      elements.refreshKnowledge.disabled = false;
    }
  }

  function readTextFile(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onerror = () => reject(new Error("无法读取该文件。"));
      reader.onload = () => resolve(asText(reader.result));
      reader.readAsText(file, "utf-8");
    });
  }

  function setKnowledgeError(message = "") {
    elements.knowledgeError.textContent = message;
  }

  async function importKnowledge(event) {
    event.preventDefault();
    setKnowledgeError();
    const file = elements.knowledgeFile.files && elements.knowledgeFile.files[0];
    const pastedText = elements.knowledgeContent.value.trim();

    if (file && pastedText) {
      setKnowledgeError("请一次选择一种资料来源：文件或粘贴文本。");
      return;
    }
    if (!file && !pastedText) {
      setKnowledgeError("请选择文本文件或粘贴资料正文。");
      return;
    }
    if (file && file.size > MAX_FILE_BYTES) {
      setKnowledgeError("文件超过 500 KB，请先提取需要的脱敏内容后再导入。");
      return;
    }

    elements.importKnowledge.disabled = true;
    const originalButtonText = elements.importKnowledge.textContent;
    elements.importKnowledge.textContent = "正在导入…";
    try {
      const content = file ? (await readTextFile(file)).trim() : pastedText;
      if (!content) throw new Error("资料内容为空，未导入。");
      const name = elements.knowledgeName.value.trim() || (file ? file.name : "未命名资料");
      await requestJson(KNOWLEDGE_IMPORT_URL, {
        method: "POST",
        allowEmpty: true,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, content })
      });
      elements.knowledgeForm.reset();
      elements.knowledgeState.hidden = false;
      elements.knowledgeState.textContent = "导入成功，正在刷新列表…";
      await loadKnowledge();
    } catch (error) {
      setKnowledgeError(asText(error.message, "导入失败，请稍后重试。"));
    } finally {
      elements.importKnowledge.disabled = false;
      elements.importKnowledge.textContent = originalButtonText;
    }
  }

  function normalizeEvidence(rawEvidence) {
    const values = Array.isArray(rawEvidence)
      ? rawEvidence
      : asArray(rawEvidence && (rawEvidence.items || rawEvidence.evidence || rawEvidence.sources));
    return values.map((raw, index) => {
      if (typeof raw === "string") {
        return { id: `E${index + 1}`, source: "资料来源", excerpt: raw, kind: "evidence" };
      }
      const item = raw && typeof raw === "object" ? raw : {};
      return {
        id: identifier(item.id || item.evidence_id || item.evidenceId || item.ref) || `E${index + 1}`,
        source: asText(item.source || item.title || item.document || item.name, "资料来源"),
        excerpt: asText(item.content || item.text || item.excerpt || item.quote || item.detail),
        kind: asText(item.kind || item.type || item.category, "evidence")
      };
    });
  }

  function normalizeClaims(rawClaims) {
    return asArray(rawClaims).map((raw, index) => {
      if (typeof raw === "string") {
        return { id: `C${index + 1}`, text: raw, refs: [], status: "supported" };
      }
      const item = raw && typeof raw === "object" ? raw : {};
      return {
        id: identifier(item.id || item.claim_id || item.claimId) || `C${index + 1}`,
        text: asText(item.text || item.claim || item.conclusion || item.answer || item.statement, "未提供结论文本"),
        refs: unique(extractReferenceIds(item.refs || item.references || item.citations || item.evidence_ids || item.evidenceIds)),
        status: asText(item.status || item.evaluation || item.state, "supported").toLowerCase()
      };
    });
  }

  function responseAnswer(response, claims) {
    const direct = response.answer ?? response.response ?? response.message;
    if (typeof direct === "string" && direct.trim()) return direct.trim();
    const report = response.report;
    if (typeof report === "string" && report.trim()) return report.trim();
    if (report && typeof report === "object") {
      const reportText = asText(report.answer || report.summary || report.text || report.content);
      if (reportText) return reportText;
    }
    if (claims.length) return claims.map((claim) => claim.text).join("\n");
    return "本轮未返回可展示的文本回答。";
  }

  function citationsInText(value) {
    return unique([...String(value || "").matchAll(/\[([^\]\s]{1,80})\]/g)].map((match) => identifier(match[1])));
  }

  function allTurnReferences(turn) {
    return unique([
      ...citationsInText(turn.answer),
      ...turn.claims.flatMap((claim) => claim.refs)
    ]);
  }

  function makeCitationButton(referenceId, turn) {
    const button = makeElement("button", "citation-button", `[${referenceId}]`);
    button.type = "button";
    button.dataset.referenceId = referenceId;
    button.dataset.turnId = turn.id;
    button.setAttribute("aria-label", `查看证据引用 ${referenceId}`);
    button.addEventListener("click", () => activateTurnGraph(turn, referenceId));
    return button;
  }

  function appendAnswerWithCitations(container, answer, turn) {
    const citationPattern = /\[([^\]\s]{1,80})\]/g;
    let cursor = 0;
    let match;
    while ((match = citationPattern.exec(answer))) {
      if (match.index > cursor) container.append(document.createTextNode(answer.slice(cursor, match.index)));
      container.append(makeCitationButton(identifier(match[1]), turn));
      cursor = match.index + match[0].length;
    }
    if (cursor < answer.length) container.append(document.createTextNode(answer.slice(cursor)));
    if (!answer) container.textContent = "本轮未返回可展示的文本回答。";
  }

  function statusLabel(status) {
    const normalized = asText(status, "").toLowerCase();
    if (/(pass|success|complete|done|supported)/.test(normalized)) return "已完成";
    if (/(review|warning|manual|blocked|abstain)/.test(normalized)) return "需复核";
    if (/(error|fail|reject)/.test(normalized)) return "异常";
    return normalized ? status : "已返回";
  }

  function appendUserMessage(content) {
    const article = makeElement("article", "message message-user");
    const avatar = makeElement("div", "message-avatar", "我");
    avatar.setAttribute("aria-hidden", "true");
    const body = makeElement("div", "message-body");
    const meta = makeElement("div", "message-meta");
    meta.append(makeElement("strong", "", "你"), makeElement("span", "", "刚刚"));
    body.append(meta, makeElement("p", "", content));
    article.append(avatar, body);
    elements.chatLog.append(article);
    scrollChatToBottom();
  }

  function appendAssistantMessage(turn) {
    const article = makeElement("article", "message message-assistant");
    article.dataset.turnId = turn.id;
    const avatar = makeElement("div", "message-avatar", "证");
    avatar.setAttribute("aria-hidden", "true");
    const body = makeElement("div", "message-body");
    const meta = makeElement("div", "message-meta");
    meta.append(
      makeElement("strong", "", "证据链助手"),
      makeElement("span", "", statusLabel(turn.status))
    );
    const answer = makeElement("p", "answer-content");
    appendAnswerWithCitations(answer, turn.answer, turn);
    body.append(meta, answer);

    if (turn.claims.length) {
      const claimList = makeElement("div", "claim-list");
      turn.claims.forEach((claim) => {
        const item = makeElement(
          "section",
          `claim-item${/(review|repair|warning|unsupported|fail)/.test(claim.status) ? " is-review" : ""}`
        );
        const label = makeElement("div", "claim-label");
        label.append(
          makeElement("span", "", `${claim.id} · 结论`),
          makeElement("span", "", statusLabel(claim.status))
        );
        item.append(label, makeElement("p", "", claim.text));
        if (claim.refs.length) {
          const references = makeElement("div", "claim-references");
          claim.refs.forEach((referenceId) => references.append(makeCitationButton(referenceId, turn)));
          item.append(references);
        }
        claimList.append(item);
      });
      body.append(claimList);
    } else {
      const refs = allTurnReferences(turn);
      if (refs.length) {
        const references = makeElement("div", "claim-references");
        refs.forEach((referenceId) => references.append(makeCitationButton(referenceId, turn)));
        body.append(references);
      }
    }

    article.append(avatar, body);
    elements.chatLog.append(article);
    scrollChatToBottom();
  }

  function appendPendingMessage() {
    const article = makeElement("article", "message message-assistant message-pending");
    article.id = "pending-answer";
    const avatar = makeElement("div", "message-avatar", "证");
    avatar.setAttribute("aria-hidden", "true");
    const body = makeElement("div", "message-body");
    const meta = makeElement("div", "message-meta");
    meta.append(makeElement("strong", "", "证据链助手"), makeElement("span", "", "正在分析"));
    const indicator = makeElement("span", "typing-indicator");
    indicator.setAttribute("aria-label", "正在生成回答");
    indicator.append(makeElement("i"), makeElement("i"), makeElement("i"));
    body.append(meta, indicator);
    article.append(avatar, body);
    elements.chatLog.append(article);
    scrollChatToBottom();
  }

  function removePendingMessage() {
    document.getElementById("pending-answer")?.remove();
  }

  function scrollChatToBottom() {
    elements.chatLog.scrollTop = elements.chatLog.scrollHeight;
  }

  function setChatBusy(isBusy) {
    activeChatRequest = isBusy;
    elements.sendMessage.disabled = isBusy;
    elements.chatMessage.disabled = isBusy;
    elements.loadDemo.disabled = isBusy;
  }

  function setChatStatus(message, isError = false) {
    elements.chatStatus.textContent = message;
    elements.chatStatus.classList.toggle("is-error", isError);
  }

  function updateRunStatus(status, run) {
    const normalized = asText(status || (run && (run.status || run.state)), "").toLowerCase();
    elements.runStatus.className = "run-status";
    if (/(pass|success|complete|done|supported)/.test(normalized)) {
      elements.runStatus.classList.add("is-passed");
    } else if (/(review|warning|manual|blocked|abstain)/.test(normalized)) {
      elements.runStatus.classList.add("is-review");
    } else if (/(error|fail|reject)/.test(normalized)) {
      elements.runStatus.classList.add("is-error");
    } else if (normalized === "running") {
      elements.runStatus.classList.add("is-running");
    } else {
      elements.runStatus.classList.add("is-idle");
    }
    elements.runStatus.textContent = statusLabel(status || (run && (run.status || run.state)) || "等待对话");
  }

  function buildHistoryPayload() {
    return chatHistory.map((turn) => ({
      role: turn.role,
      content: turn.role === "assistant" ? turn.answer : turn.content
    }));
  }

  async function sendChatMessage(event) {
    event.preventDefault();
    if (activeChatRequest) return;
    const message = elements.chatMessage.value.trim();
    if (!message) {
      setChatStatus("请先输入问题。", true);
      elements.chatMessage.focus();
      return;
    }

    const history = buildHistoryPayload();
    const patientRecord = elements.patientContext.value.trim();
    const userTurn = { role: "user", content: message };
    chatHistory.push(userTurn);
    appendUserMessage(message);
    elements.chatMessage.value = "";
    setChatBusy(true);
    setChatStatus("正在检索知识库并生成可追溯回答…");
    updateRunStatus("running");
    appendPendingMessage();

    try {
      const response = await requestJson(CHAT_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, patientRecord, history })
      });
      const claims = normalizeClaims(response.claims);
      const evidence = normalizeEvidence(response.evidence);
      const assistantTurn = {
        id: `assistant-${++assistantTurnSequence}`,
        role: "assistant",
        answer: responseAnswer(response, claims),
        claims,
        evidence,
        graph: response.graph,
        run: response.run || {},
        status: asText(response.status || (response.run && (response.run.status || response.run.state)), "completed")
      };
      chatHistory.push(assistantTurn);
      removePendingMessage();
      appendAssistantMessage(assistantTurn);
      activateTurnGraph(assistantTurn);
      updateRunStatus(assistantTurn.status, assistantTurn.run);
      const runId = asText(assistantTurn.run.id || assistantTurn.run.runId);
      setChatStatus(runId ? `已完成本轮分析（运行 ${runId.slice(0, 8)}）。` : "已完成本轮分析，可点击引用查看证据。");
    } catch (error) {
      if (chatHistory.at(-1) === userTurn) chatHistory.pop();
      removePendingMessage();
      updateRunStatus("error");
      setChatStatus(`本轮未完成：${asText(error.message, "请求失败，请稍后重试。")}`, true);
    } finally {
      setChatBusy(false);
      elements.chatMessage.focus();
    }
  }

  function graphNodeType(value) {
    const type = asText(value, "reference").toLowerCase();
    if (/(patient|case|record)/.test(type)) return "patient";
    if (/(knowledge|guideline|document|source)/.test(type)) return "knowledge";
    if (/(evidence|citation|fact)/.test(type)) return "evidence";
    if (/(task|plan)/.test(type)) return "task";
    if (/(claim|conclusion)/.test(type)) return "claim";
    if (/(report|answer|response)/.test(type)) return "answer";
    return "reference";
  }

  function graphNodeLabel(raw, fallback) {
    if (typeof raw === "string") return raw;
    if (!raw || typeof raw !== "object") return fallback;
    return asText(raw.label || raw.name || raw.text || raw.title || raw.content, fallback);
  }

  function normalizeGraph(rawGraph, claims, evidence) {
    const nodes = [];
    const edges = [];
    const byId = new Map();
    const edgeKeys = new Set();

    function addNode(raw, fallbackId, fallbackType = "reference", fallbackLabel = "关联节点") {
      const id = identifier(raw && typeof raw === "object" ? raw.id || raw.node_id || raw.nodeId : raw) || fallbackId;
      if (!id) return null;
      if (byId.has(id)) return byId.get(id);
      const source = raw && typeof raw === "object" ? raw : {};
      const node = {
        id,
        type: graphNodeType(source.type || source.kind || fallbackType),
        label: graphNodeLabel(source, fallbackLabel || id),
        status: asText(source.status || source.state)
      };
      byId.set(id, node);
      nodes.push(node);
      return node;
    }

    function addEdge(rawSource, rawTarget, type = "supports") {
      const source = identifier(rawSource);
      const target = identifier(rawTarget);
      if (!source || !target || source === target) return;
      if (!byId.has(source)) addNode({ id: source, label: source }, source, "reference", source);
      if (!byId.has(target)) addNode({ id: target, label: target }, target, "reference", target);
      const key = `${source}→${target}`;
      if (edgeKeys.has(key)) return;
      edgeKeys.add(key);
      edges.push({ id: key, source, target, type: asText(type, "supports") });
    }

    const raw = rawGraph && typeof rawGraph === "object" ? rawGraph : {};
    const rawNodes = asArray(raw.nodes || raw.vertices || raw.items);
    rawNodes.forEach((node, index) => addNode(node, `N${index + 1}`));
    const rawEdges = asArray(raw.edges || raw.links || raw.relationships);
    rawEdges.forEach((edge) => {
      if (!edge || typeof edge !== "object") return;
      addEdge(edge.from || edge.source || edge.start || edge.parent, edge.to || edge.target || edge.end || edge.child, edge.type || edge.relation);
    });

    evidence.forEach((item) => {
      addNode(
        { id: item.id, type: item.kind, label: `${item.id}：${item.source}` },
        item.id,
        item.kind,
        `${item.id}：${item.source}`
      );
    });
    claims.forEach((claim) => {
      addNode({ id: claim.id, type: "claim", label: claim.text, status: claim.status }, claim.id, "claim", claim.text);
      claim.refs.forEach((referenceId) => {
        if (!byId.has(referenceId)) {
          addNode({ id: referenceId, type: "evidence", label: referenceId }, referenceId, "evidence", referenceId);
        }
        addEdge(referenceId, claim.id, "supports");
      });
    });

    let answerNode = nodes.find((node) => node.type === "answer");
    if (!answerNode && (claims.length || evidence.length)) {
      answerNode = addNode({ id: "ANSWER", type: "answer", label: "本轮回答" }, "ANSWER", "answer", "本轮回答");
    }
    if (answerNode) {
      claims.forEach((claim) => addEdge(claim.id, answerNode.id, "reported_in"));
      if (!claims.length) evidence.forEach((item) => addEdge(item.id, answerNode.id, "supports"));
    }

    const visibleNodes = nodes.slice(0, MAX_GRAPH_NODES);
    const visibleIds = new Set(visibleNodes.map((node) => node.id));
    return {
      nodes: visibleNodes,
      edges: edges.filter((edge) => visibleIds.has(edge.source) && visibleIds.has(edge.target))
    };
  }

  function graphLayer(type) {
    if (type === "patient" || type === "knowledge" || type === "evidence" || type === "reference") return 0;
    if (type === "task") return 1;
    if (type === "claim") return 2;
    return 3;
  }

  function nodeWidth(type) {
    if (type === "answer") return 155;
    if (type === "claim") return 196;
    return 180;
  }

  function labelLines(label, limit = 17) {
    const compact = asText(label, "关联节点").replace(/\s+/g, " ");
    if (compact.length <= limit) return [compact];
    return [compact.slice(0, limit), `${compact.slice(limit, limit * 2 - 1)}…`];
  }

  function graphLayout(nodes) {
    const columns = [[], [], [], []];
    nodes.forEach((node) => columns[graphLayer(node.type)].push(node));
    const maximum = Math.max(1, ...columns.map((column) => column.length));
    const nodeHeight = 70;
    const rowGap = 20;
    const height = Math.max(274, maximum * (nodeHeight + rowGap) + 40);
    const xByLayer = [105, 320, 540, 762];
    const position = new Map();
    columns.forEach((column, layer) => {
      const occupied = column.length * nodeHeight + Math.max(0, column.length - 1) * rowGap;
      let y = Math.max(30, (height - occupied) / 2) + nodeHeight / 2;
      column.forEach((node) => {
        position.set(node.id, { x: xByLayer[layer], y, width: nodeWidth(node.type), height: nodeHeight });
        y += nodeHeight + rowGap;
      });
    });
    return { width: 860, height, position };
  }

  function drawGraph(turn) {
    const graph = normalizeGraph(turn.graph, turn.claims, turn.evidence);
    const nodeById = new Map(graph.nodes.map((node) => [node.id, node]));
    const evidenceById = new Map(turn.evidence.map((item) => [item.id, item]));
    const claimsById = new Map(turn.claims.map((item) => [item.id, item]));
    graphState = { ...graph, nodeById, evidenceById, claimsById, activeTurn: turn };

    elements.evidenceGraph.replaceChildren();
    renderEvidenceList(turn.evidence);
    if (!graph.nodes.length) {
      elements.evidenceGraph.append(makeElement("p", "graph-empty", "本轮没有返回可绘制的证据关系。"));
      elements.graphCaption.textContent = "本轮回答未包含可绘制的结论或证据节点。";
      resetSelectedEvidence();
      return;
    }

    const layout = graphLayout(graph.nodes);
    const svg = makeSvgElement("svg", {
      viewBox: `0 0 ${layout.width} ${layout.height}`,
      role: "group",
      "aria-labelledby": "evidence-svg-title evidence-svg-desc"
    });
    const title = makeSvgElement("title", { id: "evidence-svg-title" });
    title.textContent = "本轮回答的证据链图";
    const description = makeSvgElement("desc", { id: "evidence-svg-desc" });
    description.textContent = `包含 ${graph.nodes.length} 个节点和 ${graph.edges.length} 条关系。点击节点可查看来源及关联说明。`;
    const defs = makeSvgElement("defs");
    const marker = makeSvgElement("marker", {
      id: "evidence-arrow",
      viewBox: "0 0 10 10",
      refX: "8.5",
      refY: "5",
      markerWidth: "5",
      markerHeight: "5",
      orient: "auto-start-reverse"
    });
    marker.append(makeSvgElement("path", { d: "M 0 0 L 10 5 L 0 10 z", fill: "#b5c8bc" }));
    defs.append(marker);
    svg.append(title, description, defs);

    graph.edges.forEach((edge) => {
      const source = layout.position.get(edge.source);
      const target = layout.position.get(edge.target);
      if (!source || !target) return;
      const startX = source.x + source.width / 2;
      const endX = target.x - target.width / 2;
      const direction = endX >= startX ? 1 : -1;
      const bend = Math.max(42, Math.abs(endX - startX) * 0.45) * direction;
      const path = makeSvgElement("path", {
        class: "graph-edge",
        d: `M ${startX} ${source.y} C ${startX + bend} ${source.y}, ${endX - bend} ${target.y}, ${endX} ${target.y}`,
        "data-source": edge.source,
        "data-target": edge.target
      });
      svg.append(path);
    });

    graph.nodes.forEach((node) => {
      const position = layout.position.get(node.id);
      const group = makeSvgElement("g", {
        class: `graph-node type-${node.type}`,
        "data-node-id": node.id,
        role: "button",
        tabindex: "0",
        "aria-label": `${node.id}：${node.label}`
      });
      const nodeTitle = makeSvgElement("title");
      nodeTitle.textContent = `${node.id}：${node.label}`;
      const shape = makeSvgElement("rect", {
        class: "node-shape",
        x: position.x - position.width / 2,
        y: position.y - position.height / 2,
        width: position.width,
        height: position.height,
        rx: "11"
      });
      const idLabel = makeSvgElement("text", {
        class: "node-id",
        x: position.x - position.width / 2 + 12,
        y: position.y - position.height / 2 + 17
      });
      idLabel.textContent = node.id;
      const label = makeSvgElement("text", {
        class: "node-label",
        x: position.x - position.width / 2 + 12,
        y: position.y - 4
      });
      labelLines(node.label).forEach((line, index) => {
        const span = makeSvgElement("tspan", { x: position.x - position.width / 2 + 12, dy: index === 0 ? 0 : 16 });
        span.textContent = line;
        label.append(span);
      });
      group.append(nodeTitle, shape, idLabel, label);
      group.addEventListener("click", () => selectGraphNode(node.id));
      group.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          selectGraphNode(node.id);
        }
      });
      svg.append(group);
    });

    elements.evidenceGraph.append(svg);
    elements.graphCaption.textContent = `本轮返回 ${graph.nodes.length} 个节点、${graph.edges.length} 条关系。点击节点或回答中的引用可追溯来源。`;
    resetSelectedEvidence();
  }

  function renderEvidenceList(evidence) {
    elements.evidenceList.replaceChildren();
    elements.evidenceCount.textContent = String(evidence.length);
    if (!evidence.length) {
      const item = makeElement("li", "list-state", "本轮回答未返回单独的证据条目。");
      elements.evidenceList.append(item);
      return;
    }
    evidence.forEach((item) => {
      const listItem = makeElement("li");
      const button = makeElement("button");
      button.type = "button";
      button.append(
        makeElement("span", "evidence-id", item.id),
        document.createTextNode(item.source)
      );
      if (item.excerpt) button.append(makeElement("span", "evidence-source", item.excerpt));
      button.addEventListener("click", () => selectGraphNode(item.id));
      listItem.append(button);
      elements.evidenceList.append(listItem);
    });
  }

  function resetSelectedEvidence() {
    elements.selectedEvidence.textContent = "选择图中的节点或回答中的引用，可查看资料来源与关联说明。";
  }

  function selectGraphNode(id) {
    const node = graphState.nodeById.get(id);
    if (!node) {
      elements.selectedEvidence.textContent = `未找到引用 ${id} 的详细资料。`;
      return;
    }
    elements.evidenceGraph.querySelectorAll(".graph-node").forEach((candidate) => {
      candidate.classList.toggle("is-selected", candidate.dataset.nodeId === id);
    });
    elements.evidenceGraph.querySelectorAll(".graph-edge").forEach((edge) => {
      edge.classList.toggle("is-highlighted", edge.dataset.source === id || edge.dataset.target === id);
    });
    document.querySelectorAll(".citation-button").forEach((button) => {
      button.classList.toggle(
        "is-active",
        button.dataset.turnId === graphState.activeTurn?.id && button.dataset.referenceId === id
      );
    });

    const evidence = graphState.evidenceById.get(id);
    const claim = graphState.claimsById.get(id);
    const detail = document.createDocumentFragment();
    detail.append(makeElement("strong", "", `${node.id} · ${node.label}`));
    if (evidence) {
      detail.append(makeElement("p", "detail-source", `来源：${evidence.source}`));
      if (evidence.excerpt) detail.append(makeElement("p", "", evidence.excerpt));
    } else if (claim) {
      const refs = claim.refs.length ? `引用：${claim.refs.map((ref) => `[${ref}]`).join(" ")}` : "未附带引用标识";
      detail.append(makeElement("p", "", refs));
    } else {
      const related = graphState.edges
        .filter((edge) => edge.source === id || edge.target === id)
        .map((edge) => `${edge.source} → ${edge.target}`);
      detail.append(makeElement("p", "", related.length ? `关联：${related.join("；")}` : "当前节点没有可展示的关联说明。"));
    }
    elements.selectedEvidence.replaceChildren(detail);
  }

  function activateTurnGraph(turn, referenceId = "") {
    if (!turn || turn.role !== "assistant") return;
    drawGraph(turn);
    if (referenceId) selectGraphNode(referenceId);
  }

  function loadSyntheticDemo() {
    elements.patientContext.value = syntheticDemo.patientRecord;
    elements.chatMessage.value = syntheticDemo.message;
    const contextDetails = document.querySelector(".patient-context");
    if (contextDetails) contextDetails.open = true;
    setChatStatus("已载入安全合成病例和示例问题；可修改后发送。");
    elements.chatMessage.focus();
  }

  elements.knowledgeForm.addEventListener("submit", importKnowledge);
  elements.refreshKnowledge.addEventListener("click", loadKnowledge);
  elements.chatForm.addEventListener("submit", sendChatMessage);
  elements.loadDemo.addEventListener("click", loadSyntheticDemo);
  loadKnowledge();
})();
