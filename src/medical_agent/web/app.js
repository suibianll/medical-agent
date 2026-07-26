import {
  asArray,
  asText,
  graphLayout,
  graphNodeLabel,
  graphNodeType,
  identifier,
  makeElement,
  makeSvgElement,
  requestJson,
  referenceIds,
  splitSentences,
  truncateText,
  unique
} from "./shared.js";
import { createExecutionView, stageTitle } from "./execution-view.js";

(() => {
  "use strict";

  const KNOWLEDGE_URL = "/api/knowledge";
  const KNOWLEDGE_IMPORT_URL = "/api/knowledge/import";
  const CHAT_STREAM_URL = "/api/chat/stream";
  const HEALTH_URL = "/api/health";
  const MAX_FILE_BYTES = 500_000;
  const MAX_GRAPH_NODES = 48;

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
    reportTemplate: document.getElementById("report-template"),
    reportTitle: document.getElementById("report-title"),
    sendMessage: document.getElementById("send-message"),
    chatStatus: document.getElementById("chat-status"),
    loadDemo: document.getElementById("load-demo"),
    modelMode: document.getElementById("model-mode"),
    modelDetail: document.getElementById("model-detail"),
    modelProfile: document.getElementById("model-profile"),
    graphCaption: document.getElementById("graph-caption"),
    evidenceGraph: document.getElementById("evidence-graph"),
    runStatus: document.getElementById("run-status"),
    openEvidencePage: document.getElementById("open-evidence-page"),
    selectedEvidence: document.getElementById("selected-evidence"),
    evidenceList: document.getElementById("evidence-list"),
    evidenceCount: document.getElementById("evidence-count"),
    citationPreview: document.getElementById("citation-preview"),
    executionStage: document.getElementById("execution-stage"),
    taskProgress: document.getElementById("task-progress"),
    executionTrace: document.getElementById("execution-trace")
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
  const executionView = createExecutionView(elements);
  let modelState = { mode: "unknown", provider: "", name: "" };
  let modelProfiles = [];

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

  function setModelState(raw, isError = false) {
    const source = raw && typeof raw === "object" ? raw : {};
    const mode = asText(source.mode || source.model_mode || source.modelMode, "unknown").toLowerCase();
    modelState = {
      mode: mode === "real" ? "real" : mode === "demo" ? "demo" : "unknown",
      provider: asText(source.provider),
      name: asText(source.name || source.model || source.model_name || source.modelName)
    };
    elements.modelMode.className = `model-badge is-${isError ? "error" : modelState.mode}`;
    elements.modelMode.textContent = isError
      ? "模型状态未知"
      : modelState.mode === "real"
        ? "真实模型"
        : modelState.mode === "demo"
          ? "演示模型"
          : "模型未配置";
    const detail = [modelState.provider, modelState.name].filter(Boolean).join(" · ");
    elements.modelDetail.textContent = detail || (isError ? "服务未连接" : "本地工作台");
  }

  function setModelCatalog(rawCatalog, fallbackModel = {}) {
    const catalog = rawCatalog && typeof rawCatalog === "object" ? rawCatalog : {};
    modelProfiles = asArray(catalog.profiles).filter((profile) => profile && typeof profile === "object");
    elements.modelProfile.replaceChildren();
    if (!modelProfiles.length) {
      const option = makeElement("option", "", asText(fallbackModel.name, "当前模型"));
      option.value = "";
      elements.modelProfile.append(option);
      elements.modelProfile.disabled = true;
      setModelState(fallbackModel);
      return;
    }
    modelProfiles.forEach((profile) => {
      const option = makeElement("option", "", asText(profile.label || profile.name || profile.id, "模型"));
      option.value = identifier(profile.id);
      option.textContent = `${asText(profile.label || profile.name || profile.id, "模型")} · ${profile.mode === "demo" ? "演示" : "真实"}`;
      elements.modelProfile.append(option);
    });
    elements.modelProfile.disabled = false;
    const defaultProfile = identifier(catalog.default);
    if (modelProfiles.some((profile) => identifier(profile.id) === defaultProfile)) {
      elements.modelProfile.value = defaultProfile;
    }
    const selected = modelProfiles.find((profile) => identifier(profile.id) === elements.modelProfile.value) || modelProfiles[0];
    if (selected) setModelState(selected);
  }

  function selectedModelProfile() {
    return identifier(elements.modelProfile?.value);
  }

  async function loadHealth() {
    try {
      const health = await requestJson(HEALTH_URL);
      setModelCatalog(health.models, health.model || health);
    } catch {
      elements.modelProfile.disabled = true;
      setModelState({}, true);
    }
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
        return { id: `E${index + 1}`, source: "资料来源", locator: "", excerpt: raw, kind: "evidence" };
      }
      const item = raw && typeof raw === "object" ? raw : {};
      return {
        id: identifier(item.id || item.evidence_id || item.evidenceId || item.ref) || `E${index + 1}`,
        source: asText(item.source || item.title || item.document || item.name, "资料来源"),
        locator: asText(item.locator || item.location || item.section || item.document_id || item.documentId),
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
        refs: referenceIds(item.refs || item.references || item.citations || item.evidence_ids || item.evidenceIds),
        taskId: identifier(item.task_id || item.taskId),
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

  function evidenceForReference(turn, referenceId) {
    return (turn.evidence || []).find((item) => item.id === referenceId) || null;
  }

  function hideCitationPreview() {
    if (elements.citationPreview) elements.citationPreview.hidden = true;
  }

  function showCitationPreview(button, turn, referenceId) {
    const preview = elements.citationPreview;
    if (!preview) return;
    const evidence = evidenceForReference(turn, referenceId);
    const source = evidence?.source || "本轮证据引用";
    const locator = evidence?.locator || "定位信息未单独返回";
    const excerpt = evidence?.excerpt || "该引用的原文摘要将在右侧证据详情中显示。";
    preview.replaceChildren(
      makeElement("strong", "", `[${referenceId}] · ${source}`),
      makeElement("span", "citation-preview-locator", `定位：${locator}`),
      makeElement("span", "", truncateText(excerpt, 360))
    );
    preview.hidden = false;
    const trigger = button.getBoundingClientRect();
    const tip = preview.getBoundingClientRect();
    const viewportWidth = document.documentElement.clientWidth;
    const left = Math.max(8, Math.min(trigger.left + window.scrollX, window.scrollX + viewportWidth - tip.width - 8));
    preview.style.left = `${left}px`;
    preview.style.top = `${trigger.bottom + window.scrollY + 7}px`;
  }

  function makeCitationButton(referenceId, turn) {
    const button = makeElement("button", "citation-button", `[${referenceId}]`);
    button.type = "button";
    button.dataset.referenceId = referenceId;
    button.dataset.turnId = turn.id;
    button.setAttribute("aria-label", `查看证据引用 ${referenceId}`);
    button.setAttribute("aria-describedby", "citation-preview");
    const previewEvidence = () => {
      activateTurnGraph(turn, referenceId);
      showCitationPreview(button, turn, referenceId);
    };
    button.addEventListener("mouseenter", previewEvidence);
    button.addEventListener("focus", previewEvidence);
    button.addEventListener("mouseleave", hideCitationPreview);
    button.addEventListener("blur", hideCitationPreview);
    button.addEventListener("click", previewEvidence);
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
      const claimList = makeElement("div", "claim-list sentence-proof-compact");
      turn.claims.forEach((claim) => {
        splitSentences(claim.text).forEach((sentence, sentenceIndex) => {
          const item = makeElement(
            "section",
            `claim-item${/(review|repair|warning|unsupported|fail)/.test(claim.status) || !claim.refs.length ? " is-review" : ""}`
          );
          const label = makeElement("div", "claim-label");
          const origin = claim.taskId ? `${claim.id}.${sentenceIndex + 1} · T${String(claim.taskId).replace(/^T/i, "")}` : `${claim.id}.${sentenceIndex + 1}`;
          label.append(
            makeElement("span", "", origin),
            makeElement("span", "", claim.refs.length ? "引用已关联" : "缺少引用")
          );
          item.append(label, makeElement("p", "", sentence));
          const references = makeElement("div", "claim-references");
          if (claim.refs.length) {
            claim.refs.forEach((referenceId) => references.append(makeCitationButton(referenceId, turn)));
          } else {
            references.append(makeElement("span", "uncited-message", "需要人工复核"));
          }
          item.append(references);
          claimList.append(item);
        });
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
    meta.append(
      makeElement("strong", "", "证据链助手"),
      makeElement("span", "pending-stage", "正在提交任务")
    );
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
    if (elements.reportTemplate) elements.reportTemplate.disabled = isBusy;
    if (elements.reportTitle) elements.reportTitle.disabled = isBusy;
    if (elements.modelProfile) elements.modelProfile.disabled = isBusy || !modelProfiles.length;
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

  function selectedReportTemplate() {
    const name = asText(elements.reportTemplate?.value, "evidence_summary");
    const title = asText(elements.reportTitle?.value);
    return title ? { name, title } : name;
  }

  function setEvidencePageLink(runId, result = null) {
    const safeRunId = asText(runId);
    if (!safeRunId || !elements.openEvidencePage) return;
    const target = new URL("/evidence.html", window.location.origin);
    target.searchParams.set("runId", safeRunId);
    elements.openEvidencePage.href = target.pathname + target.search;
    elements.openEvidencePage.hidden = false;
    if (result && typeof result === "object") {
      try {
        sessionStorage.setItem(`medical-agent-run:${safeRunId}`, JSON.stringify(result));
        sessionStorage.setItem(
          `medical-agent-run-context:${safeRunId}`,
          JSON.stringify({
            reportTemplate: selectedReportTemplate(),
            modelProfile: selectedModelProfile()
          })
        );
      } catch {
        // Session storage is only a convenience fallback for the evidence page.
      }
    }
  }

  function parseSseBlock(block) {
    let eventName = "message";
    const dataLines = [];
    block.split(/\r?\n/).forEach((line) => {
      if (!line || line.startsWith(":")) return;
      const separator = line.indexOf(":");
      const field = separator >= 0 ? line.slice(0, separator) : line;
      const value = separator >= 0 ? line.slice(separator + 1).replace(/^ /, "") : "";
      if (field === "event") eventName = value || eventName;
      if (field === "data") dataLines.push(value);
    });
    if (!dataLines.length) return null;
    const raw = dataLines.join("\n");
    if (raw === "[DONE]") return { eventName: "done", payload: {} };
    try {
      return { eventName, payload: JSON.parse(raw) };
    } catch {
      return { eventName, payload: { message: raw } };
    }
  }

  function streamEventResult(eventName, payload) {
    const data = payload && typeof payload === "object" ? payload : {};
    const kind = `${asText(eventName)} ${asText(data.type || data.event || data.kind)}`.toLowerCase();
    if (!/(result|final|complete|done)/.test(kind)) return null;
    if (data.result && typeof data.result === "object") return data.result;
    if (data.response && typeof data.response === "object") return data.response;
    if (data.data && typeof data.data === "object") return data.data;
    if (data.claims || data.evidence || data.run || data.answer || data.report) return data;
    return null;
  }

  function streamEventError(eventName, payload) {
    const data = payload && typeof payload === "object" ? payload : {};
    const kind = `${asText(eventName)} ${asText(data.type || data.event || data.kind)}`.toLowerCase();
    if (!/(error|fail|reject)/.test(kind)) return "";
    return getErrorMessage(data, "流式任务未完成。");
  }

  function updatePendingStage(stage) {
    const label = document.querySelector("#pending-answer .pending-stage");
    if (label) label.textContent = stageTitle(stage);
  }

  async function requestChatStream(payload) {
    const response = await fetch(CHAT_STREAM_URL, {
      method: "POST",
      headers: {
        Accept: "text/event-stream",
        "Content-Type": "application/json",
        "Cache-Control": "no-cache"
      },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      const raw = await response.text();
      let errorPayload = raw;
      try {
        errorPayload = raw ? JSON.parse(raw) : {};
      } catch {}
      throw new Error(getErrorMessage(errorPayload, `请求失败（HTTP ${response.status}）`));
    }

    const contentType = response.headers.get("content-type") || "";
    if (!contentType.includes("text/event-stream") || !response.body) {
      throw new Error("服务端未返回 SSE 实时事件流。");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    let result = null;

    const consumeBlock = (block) => {
      const event = parseSseBlock(block);
      if (!event) return;
      const failure = streamEventError(event.eventName, event.payload);
      if (failure) throw new Error(failure);
      const terminal = streamEventResult(event.eventName, event.payload);
      if (terminal) {
        result = terminal;
        return;
      }
      if (event.eventName !== "done") {
        executionView.applyProgressEvent(event.payload, event.eventName);
        updatePendingStage(asText(event.payload?.stage || event.eventName));
        setChatStatus(`正在${stageTitle(event.payload?.stage || event.eventName)}…`);
      }
    };

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const blocks = buffer.split(/\r?\n\r?\n/);
        buffer = blocks.pop() || "";
        blocks.forEach(consumeBlock);
      }
      buffer += decoder.decode();
      if (buffer.trim()) consumeBlock(buffer);
    } finally {
      reader.releaseLock();
    }

    if (!result) throw new Error("实时任务连接已结束，但未收到最终回答。");
    return result;
  }

  function buildAssistantTurn(response) {
    const claims = normalizeClaims(response.claims);
    const evidence = normalizeEvidence(response.evidence);
    return {
      id: `assistant-${++assistantTurnSequence}`,
      role: "assistant",
      answer: responseAnswer(response, claims),
      claims,
      evidence,
      graph: response.graph,
      run: response.run || {},
      status: asText(response.status || (response.run && (response.run.status || response.run.state)), "completed")
    };
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
    executionView.reset();
    executionView.applyProgressEvent({ stage: "submit", message: "问题已提交，正在等待任务规划。" }, "progress");
    setChatStatus("正在连接任务编排并生成可追溯回答…");
    updateRunStatus("running");
    appendPendingMessage();

    try {
      const response = await requestChatStream({
        message,
        patientRecord,
        history,
        reportTemplate: selectedReportTemplate(),
        modelProfile: selectedModelProfile()
      });
      executionView.hydrateFromResult(response);
      const assistantTurn = buildAssistantTurn(response);
      chatHistory.push(assistantTurn);
      removePendingMessage();
      appendAssistantMessage(assistantTurn);
      activateTurnGraph(assistantTurn);
      updateRunStatus(assistantTurn.status, assistantTurn.run);
      if (response.run?.model) setModelState(response.run.model);
      const runId = asText(assistantTurn.run.id || assistantTurn.run.runId);
      setEvidencePageLink(runId, response);
      setChatStatus(runId ? `已完成本轮分析（运行 ${runId.slice(0, 8)}）。` : "已完成本轮分析，可点击引用查看证据。");
    } catch (error) {
      if (chatHistory.at(-1) === userTurn) chatHistory.pop();
      removePendingMessage();
      executionView.setStage("error");
      executionView.recordTrace("执行异常", asText(error.message, "请求失败，请稍后重试。"), "error");
      updateRunStatus("error");
      setChatStatus(`本轮未完成：${asText(error.message, "请求失败，请稍后重试。")}`, true);
    } finally {
      setChatBusy(false);
      elements.chatMessage.focus();
    }
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

  function labelLines(label, limit = 17) {
    const compact = asText(label, "关联节点").replace(/\s+/g, " ");
    if (compact.length <= limit) return [compact];
    return [compact.slice(0, limit), `${compact.slice(limit, limit * 2 - 1)}…`];
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

    const layout = graphLayout(graph.nodes, {
      width: 860,
      minimumHeight: 274,
      nodeHeight: 70,
      rowGap: 20,
      topPadding: 20,
      xByLayer: [105, 320, 540, 762],
      widths: { answer: 155, claim: 196, default: 180 }
    });
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
      const source = layout.positions.get(edge.source);
      const target = layout.positions.get(edge.target);
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
      const position = layout.positions.get(node.id);
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
  elements.modelProfile.addEventListener("change", () => {
    const selected = modelProfiles.find((profile) => identifier(profile.id) === selectedModelProfile());
    if (selected) {
      setModelState(selected);
      setChatStatus(`下一轮将使用：${asText(selected.label || selected.name || selected.id)}。`);
    }
  });
  executionView.reset();
  loadKnowledge();
  loadHealth();
})();
