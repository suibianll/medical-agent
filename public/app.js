(() => {
  "use strict";

  const API_URL = "/api/runs";
  const SVG_NS = "http://www.w3.org/2000/svg";
  const stepOrder = ["validate", "plan", "execute", "evaluate", "report"];

  const DEMO = {
    patientRecord: `【合成病例，仅用于演示】\n患者：62 岁，男性，身份信息已移除。\n主诉：近两周乏力、食欲下降，无胸痛、呼吸困难或少尿。\n既往史：2 型糖尿病、高血压。\n当前用药：二甲双胍 0.5 g，每日两次；缬沙坦 80 mg，每日一次。\n检查：肌酐 165 μmol/L；eGFR 38 mL/min/1.73m²；钾 4.7 mmol/L。\n已知药物过敏：无。`,
    request: "请基于病历和知识库，梳理当前用药与肾功能相关的注意事项；每项结论都应显示患者事实和知识库证据引用，并明确证据不足之处。",
    plan: [
      { id: 1, goal: "提取与肾功能、当前用药有关的患者事实", deps: [] },
      { id: 2, goal: "检索肾功能受损患者的相关用药依据", deps: [1] },
      { id: 3, goal: "评估当前药物的肾功能相关注意事项", deps: [1, 2] },
      { id: 4, goal: "汇总结论、局限性和后续信息需求", deps: [3] }
    ]
  };

  const $ = (selector) => document.querySelector(selector);
  const elements = {
    form: $("#run-form"),
    patientRecord: $("#patient-record"),
    request: $("#request"),
    plan: $("#plan-json"),
    loadDemo: $("#load-demo"),
    runButton: $("#run-button"),
    copyReport: $("#copy-report"),
    patientError: $("#patient-record-error"),
    requestError: $("#request-error"),
    planError: $("#plan-error"),
    runState: $("#run-state"),
    runSummary: $("#run-summary"),
    progressSteps: $("#progress-steps"),
    results: $("#results"),
    report: $("#report-content"),
    evidenceList: $("#evidence-list"),
    evidenceItems: $("#evidence-items"),
    dagCanvas: $("#dag-canvas"),
    dagDescription: $("#dag-description"),
    taskCount: $("#task-count"),
    taskList: $("#task-list"),
    graphCanvas: $("#evidence-graph"),
    graphSummary: $("#graph-summary")
  };

  let lastReportText = "";
  let activeTimers = [];

  function clear(element) {
    while (element.firstChild) element.removeChild(element.firstChild);
  }

  function text(value) {
    return value == null ? "" : String(value);
  }

  function normalizeArray(value) {
    if (Array.isArray(value)) return value;
    if (value && typeof value === "object") return Object.values(value);
    return [];
  }

  function setState(label, kind = "idle") {
    elements.runState.textContent = label;
    elements.runState.className = `status-pill status-${kind}`;
  }

  function resetSteps() {
    stepOrder.forEach((step) => {
      const item = elements.progressSteps.querySelector(`[data-step="${step}"]`);
      item?.classList.remove("is-active", "is-done", "is-error");
    });
  }

  function setSteps(current, state = "running") {
    const index = stepOrder.indexOf(current);
    stepOrder.forEach((step, stepIndex) => {
      const item = elements.progressSteps.querySelector(`[data-step="${step}"]`);
      if (!item) return;
      item.classList.remove("is-active", "is-done", "is-error");
      if (state === "error" && stepIndex === index) item.classList.add("is-error");
      else if (stepIndex < index || state === "success") item.classList.add("is-done");
      else if (stepIndex === index) item.classList.add("is-active");
    });
  }

  function clearTimers() {
    activeTimers.forEach(window.clearTimeout);
    activeTimers = [];
  }

  function startProgress() {
    clearTimers();
    setSteps("validate");
    const scheduled = [
      [500, "plan", "正在生成可执行任务计划…"],
      [1250, "execute", "正在检索知识并抽取可引用事实…"],
      [2400, "evaluate", "正在检查结论与证据链…"],
      [3700, "report", "正在生成报告和可视化…"]
    ];
    scheduled.forEach(([delay, step, message]) => {
      activeTimers.push(window.setTimeout(() => {
        setSteps(step);
        elements.runSummary.textContent = message;
      }, delay));
    });
  }

  function setBusy(busy) {
    elements.runButton.disabled = busy;
    elements.runButton.innerHTML = busy
      ? '<span class="button-icon" aria-hidden="true">◌</span>正在运行…'
      : '<span class="button-icon" aria-hidden="true">▶</span>运行演示';
    elements.report.setAttribute("aria-busy", String(busy));
  }

  function clearErrors() {
    elements.patientError.textContent = "";
    elements.requestError.textContent = "";
    elements.planError.textContent = "";
  }

  function parseInput() {
    const patientRecord = elements.patientRecord.value.trim();
    const request = elements.request.value.trim();
    const planText = elements.plan.value.trim();
    let valid = true;
    let plan;

    if (!patientRecord) {
      elements.patientError.textContent = "请填写脱敏后的患者病历。";
      valid = false;
    }
    if (!request) {
      elements.requestError.textContent = "请填写需要完成的分析请求。";
      valid = false;
    }
    if (planText) {
      try {
        plan = JSON.parse(planText);
        if (!Array.isArray(plan) && !(plan && Array.isArray(plan.tasks))) {
          throw new Error("任务计划应为任务数组，或包含 tasks 数组的对象。");
        }
      } catch (error) {
        elements.planError.textContent = `计划 JSON 无法解析：${error.message}`;
        valid = false;
      }
    }

    return valid ? { patientRecord, request, ...(plan ? { plan } : {}) } : null;
  }

  function createSvg(tag, attributes = {}) {
    const node = document.createElementNS(SVG_NS, tag);
    Object.entries(attributes).forEach(([name, value]) => node.setAttribute(name, String(value)));
    return node;
  }

  function addSvgText(parent, x, y, content, attributes = {}) {
    const svgText = createSvg("text", { x, y, ...attributes });
    svgText.textContent = content;
    parent.appendChild(svgText);
    return svgText;
  }

  function truncate(value, max = 25) {
    const source = text(value).replace(/\s+/g, " ").trim();
    return source.length > max ? `${source.slice(0, max - 1)}…` : source;
  }

  function citationClass(id) {
    if (/^P/i.test(id)) return "patient";
    if (/^K/i.test(id)) return "knowledge";
    return "unknown";
  }

  function appendCitation(parent, id) {
    const citation = document.createElement("span");
    citation.className = `citation ${citationClass(id)}`;
    citation.textContent = `[${id}]`;
    citation.title = `证据引用 ${id}`;
    parent.appendChild(citation);
  }

  function appendTextWithCitations(parent, value) {
    const source = text(value);
    const pattern = /\[([PK]\d+(?:\s*[,，]\s*[PK]\d+)*)\]/gi;
    let cursor = 0;
    let match;
    while ((match = pattern.exec(source)) !== null) {
      if (match.index > cursor) parent.appendChild(document.createTextNode(source.slice(cursor, match.index)));
      match[1].split(/\s*[,，]\s*/).forEach((id) => appendCitation(parent, id));
      cursor = pattern.lastIndex;
    }
    if (cursor < source.length) parent.appendChild(document.createTextNode(source.slice(cursor)));
  }

  function extractEvidence(response) {
    const candidates = [
      response?.report?.evidence,
      response?.report?.references,
      response?.run?.evidence,
      response?.evidence
    ];
    const seen = new Map();
    candidates.flatMap(normalizeArray).forEach((item) => {
      if (!item) return;
      if (typeof item === "string") return;
      const id = text(item.id || item.evidenceId || item.ref || item.key);
      if (!id) return;
      seen.set(id, {
        id,
        text: text(item.text || item.quote || item.content || item.snippet || item.fact),
        source: text(item.source || item.title || item.publisher || item.document || "来源信息未提供")
      });
    });
    return [...seen.values()];
  }

  function extractTasks(run) {
    const candidates = [
      run?.tasks,
      run?.plan?.tasks,
      Array.isArray(run?.plan) ? run.plan : null,
      run?.execution?.tasks,
      run?.taskResults
    ];
    const raw = candidates.map(normalizeArray).find((items) => items.length) || [];
    return raw.map((task, index) => {
      const id = text(task?.id ?? task?.taskId ?? index + 1);
      const deps = normalizeArray(task?.deps ?? task?.dependencies ?? task?.dependsOn)
        .map((dep) => text(typeof dep === "object" ? (dep.id ?? dep.taskId) : dep))
        .filter(Boolean);
      return {
        id,
        goal: text(task?.goal || task?.title || task?.objective || task?.name || `任务 ${id}`),
        deps,
        status: text(task?.status || task?.state || "待执行")
      };
    });
  }

  function extractClaims(report) {
    const raw = normalizeArray(report?.claims || report?.conclusions || report?.items);
    return raw.map((claim, index) => ({
      id: text(claim?.id || claim?.claimId || `C${index + 1}`),
      text: text(typeof claim === "string" ? claim : claim?.text || claim?.claim || claim?.content),
      refs: normalizeArray(claim?.refs || claim?.citations || claim?.evidenceIds)
        .map((ref) => text(typeof ref === "object" ? (ref.id || ref.evidenceId || ref.ref) : ref))
        .filter(Boolean),
      confidence: text(claim?.confidence || claim?.status || "")
    })).filter((claim) => claim.text);
  }

  function getReportText(report) {
    if (typeof report === "string") return report;
    if (!report || typeof report !== "object") return "";
    return text(report.markdown || report.text || report.body || report.content || report.summary);
  }

  function renderReport(report, evidence) {
    clear(elements.report);
    const reportText = getReportText(report);
    const claims = extractClaims(report);
    const title = report && typeof report === "object" ? text(report.title) : "";
    const summary = report && typeof report === "object" ? text(report.summary) : "";
    lastReportText = reportText || claims.map((claim) => `${claim.text} ${claim.refs.map((ref) => `[${ref}]`).join("")}`).join("\n");

    if (title) {
      const heading = document.createElement("h3");
      heading.textContent = title;
      elements.report.appendChild(heading);
    }
    if (summary && summary !== reportText) {
      const paragraph = document.createElement("p");
      appendTextWithCitations(paragraph, summary);
      elements.report.appendChild(paragraph);
    }

    if (claims.length) {
      claims.forEach((claim) => {
        const card = document.createElement("article");
        card.className = "report-claim";
        const paragraph = document.createElement("p");
        appendTextWithCitations(paragraph, claim.text);
        claim.refs.forEach((ref) => appendCitation(paragraph, ref));
        card.appendChild(paragraph);
        if (claim.confidence) {
          const meta = document.createElement("p");
          meta.className = "claim-meta";
          meta.textContent = `结论 ${claim.id} · ${claim.confidence}`;
          card.appendChild(meta);
        }
        elements.report.appendChild(card);
      });
    } else if (reportText) {
      reportText.split(/\n{2,}/).map((paragraph) => paragraph.trim()).filter(Boolean).forEach((line) => {
        const paragraph = document.createElement("p");
        paragraph.className = "report-line";
        appendTextWithCitations(paragraph, line.replace(/^#{1,6}\s+/, ""));
        elements.report.appendChild(paragraph);
      });
    } else {
      const empty = document.createElement("p");
      empty.className = "report-empty";
      empty.textContent = "服务端未返回可展示的报告文本。";
      elements.report.appendChild(empty);
    }

    clear(elements.evidenceItems);
    if (evidence.length) {
      evidence.forEach((item) => {
        const li = document.createElement("li");
        const id = document.createElement("span");
        id.className = "evidence-id";
        id.textContent = `[${item.id}]`;
        li.appendChild(id);
        li.appendChild(document.createTextNode(item.text || "证据文本未提供"));
        const source = document.createElement("span");
        source.className = "evidence-source";
        source.textContent = item.source;
        li.appendChild(source);
        elements.evidenceItems.appendChild(li);
      });
      elements.evidenceList.hidden = false;
    } else {
      elements.evidenceList.hidden = true;
    }
    elements.copyReport.disabled = !lastReportText;
  }

  function levelsForTasks(tasks) {
    const levels = new Map();
    const ids = new Set(tasks.map((task) => task.id));
    tasks.forEach((task) => {
      const visit = (id, trail = new Set()) => {
        if (levels.has(id)) return levels.get(id);
        if (trail.has(id)) return 0;
        const current = tasks.find((candidate) => candidate.id === id);
        if (!current) return 0;
        const nextTrail = new Set(trail);
        nextTrail.add(id);
        const value = current.deps.filter((dep) => ids.has(dep)).reduce((max, dep) => Math.max(max, visit(dep, nextTrail) + 1), 0);
        levels.set(id, value);
        return value;
      };
      visit(task.id);
    });
    return levels;
  }

  function renderTaskDag(tasks) {
    clear(elements.dagCanvas);
    clear(elements.taskList);
    if (!tasks.length) {
      elements.taskCount.textContent = "";
      elements.dagDescription.textContent = "服务端未返回任务计划。";
      const empty = document.createElement("p");
      empty.className = "report-empty";
      empty.textContent = "暂无任务依赖图。";
      elements.dagCanvas.appendChild(empty);
      return;
    }

    elements.taskCount.textContent = `${tasks.length} 个任务`;
    elements.dagDescription.textContent = "箭头由上游任务指向依赖其结果的下游任务。下方列表给出每项任务的文字说明。";
    const levels = levelsForTasks(tasks);
    const groups = new Map();
    tasks.forEach((task) => {
      const level = levels.get(task.id) || 0;
      groups.set(level, [...(groups.get(level) || []), task]);
    });
    const maxLevel = Math.max(...groups.keys());
    const largestGroup = Math.max(...[...groups.values()].map((group) => group.length));
    const width = Math.max(670, (maxLevel + 1) * 220 + 80);
    const height = Math.max(180, largestGroup * 100 + 72);
    const svg = createSvg("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": `共 ${tasks.length} 个任务的依赖图` });
    const defs = createSvg("defs");
    const marker = createSvg("marker", { id: "dag-arrow", viewBox: "0 0 10 10", refX: "9", refY: "5", markerWidth: "6", markerHeight: "6", orient: "auto-start-reverse" });
    marker.appendChild(createSvg("path", { d: "M 0 0 L 10 5 L 0 10 z", fill: "#799488" }));
    defs.appendChild(marker);
    svg.appendChild(defs);
    const nodePositions = new Map();
    [...groups.entries()].forEach(([level, group]) => {
      group.forEach((task, index) => {
        const x = 30 + level * 220;
        const y = 35 + index * 100 + (height - (group.length * 100)) / 2;
        nodePositions.set(task.id, { x, y });
      });
    });
    const edgeLayer = createSvg("g", { class: "dag-edges" });
    tasks.forEach((task) => task.deps.forEach((dep) => {
      const from = nodePositions.get(dep);
      const to = nodePositions.get(task.id);
      if (!from || !to) return;
      edgeLayer.appendChild(createSvg("line", {
        x1: from.x + 174, y1: from.y + 30, x2: to.x - 8, y2: to.y + 30,
        stroke: "#799488", "stroke-width": "1.8", "marker-end": "url(#dag-arrow)"
      }));
    }));
    svg.appendChild(edgeLayer);
    tasks.forEach((task) => {
      const position = nodePositions.get(task.id);
      const group = createSvg("g", { tabindex: "0", role: "img", "aria-label": `任务 ${task.id}：${task.goal}。依赖：${task.deps.join("、") || "无"}` });
      group.appendChild(createSvg("rect", { x: position.x, y: position.y, width: "174", height: "60", rx: "9", fill: "#f0e9f7", stroke: "#9c7abc" }));
      addSvgText(group, position.x + 11, position.y + 21, `T${task.id}`, { fill: "#5a397e", "font-size": "12", "font-weight": "700" });
      addSvgText(group, position.x + 11, position.y + 42, truncate(task.goal, 21), { fill: "#25202c", "font-size": "11" });
      const title = createSvg("title");
      title.textContent = `任务 ${task.id}：${task.goal}`;
      group.appendChild(title);
      svg.appendChild(group);
    });
    elements.dagCanvas.appendChild(svg);

    tasks.forEach((task) => {
      const li = document.createElement("li");
      li.className = "task-item";
      const id = document.createElement("span");
      id.className = "task-id";
      id.textContent = `T${task.id}`;
      const goal = document.createElement("span");
      goal.textContent = task.goal;
      const status = document.createElement("span");
      status.className = `task-status ${task.status.toLowerCase().replace(/\s+/g, "-")}`;
      status.textContent = task.status;
      const deps = document.createElement("span");
      deps.className = "task-deps";
      deps.textContent = task.deps.length ? `依赖：${task.deps.map((dep) => `T${dep}`).join("、")}` : "无上游依赖";
      li.append(id, goal, status, deps);
      elements.taskList.appendChild(li);
    });
  }

  function inferNodeType(node) {
    const explicit = text(node.type || node.kind || node.category).toLowerCase();
    if (/(task|任务)/.test(explicit)) return "task";
    if (/(patient|病历|patient_fact)/.test(explicit)) return "patient";
    if (/(evidence|source|知识|fact|证据)/.test(explicit)) return "evidence";
    if (/(claim|conclusion|结论)/.test(explicit)) return "claim";
    if (/(report|报告)/.test(explicit)) return "report";
    const id = text(node.id);
    if (/^T/i.test(id)) return "task";
    if (/^P/i.test(id)) return "patient";
    if (/^K/i.test(id)) return "evidence";
    if (/^C/i.test(id)) return "claim";
    if (/^R/i.test(id)) return "report";
    return "evidence";
  }

  function normalizeGraph(response, tasks, evidence) {
    const source = response?.graph || {};
    const rawNodes = normalizeArray(source.nodes || source.vertices || source.items);
    const rawEdges = normalizeArray(source.edges || source.links || source.relations);
    const nodes = new Map();
    rawNodes.forEach((node, index) => {
      const id = text(node?.id || node?.key || node?.nodeId || index + 1);
      if (!id) return;
      nodes.set(id, {
        id,
        label: text(node?.label || node?.name || node?.title || node?.text || node?.content || id),
        type: inferNodeType({ ...node, id }),
        detail: text(node?.detail || node?.source || node?.description || "")
      });
    });
    const edges = rawEdges.map((edge) => ({
      source: text(edge?.source || edge?.from || edge?.u || edge?.fromId),
      target: text(edge?.target || edge?.to || edge?.v || edge?.toId),
      label: text(edge?.label || edge?.type || edge?.relation || edge?.kind || "支持")
    })).filter((edge) => edge.source && edge.target);

    if (!nodes.size) {
      tasks.forEach((task) => nodes.set(`T${task.id}`, { id: `T${task.id}`, label: task.goal, type: "task", detail: "" }));
      evidence.forEach((item) => nodes.set(item.id, { id: item.id, label: item.text || item.id, type: /^P/i.test(item.id) ? "patient" : "evidence", detail: item.source }));
      const claims = extractClaims(response?.report);
      claims.forEach((claim) => {
        nodes.set(claim.id, { id: claim.id, label: claim.text, type: "claim", detail: "" });
        claim.refs.forEach((ref) => edges.push({ source: ref, target: claim.id, label: "支持" }));
      });
      if (claims.length) {
        nodes.set("R1", { id: "R1", label: "最终报告", type: "report", detail: "" });
        claims.forEach((claim) => edges.push({ source: claim.id, target: "R1", label: "形成" }));
      }
    }
    edges.forEach((edge) => {
      if (!nodes.has(edge.source)) nodes.set(edge.source, { id: edge.source, label: edge.source, type: inferNodeType({ id: edge.source }), detail: "" });
      if (!nodes.has(edge.target)) nodes.set(edge.target, { id: edge.target, label: edge.target, type: inferNodeType({ id: edge.target }), detail: "" });
    });
    return { nodes: [...nodes.values()], edges };
  }

  function graphColumn(type) {
    if (type === "task") return 0;
    if (type === "patient" || type === "evidence") return 1;
    if (type === "claim") return 2;
    if (type === "report") return 3;
    return 1;
  }

  function graphColor(type) {
    if (type === "task") return { fill: "#f0e9f7", stroke: "#9c7abc", text: "#513574" };
    if (type === "patient") return { fill: "#e1f1fa", stroke: "#8dc7e3", text: "#155879" };
    if (type === "claim") return { fill: "#dff3e8", stroke: "#7dba96", text: "#075235" };
    if (type === "report") return { fill: "#fff3d5", stroke: "#e6bb60", text: "#744900" };
    return { fill: "#e9f2f8", stroke: "#9ec2d9", text: "#1d638d" };
  }

  function renderEvidenceGraph(graph) {
    clear(elements.graphCanvas);
    clear(elements.graphSummary);
    if (!graph.nodes.length) {
      const empty = document.createElement("p");
      empty.className = "report-empty";
      empty.textContent = "暂无可视化证据链。运行完成后将展示任务、证据、结论和报告之间的关系。";
      elements.graphCanvas.appendChild(empty);
      return;
    }
    const groups = [[], [], [], []];
    graph.nodes.forEach((node) => groups[graphColumn(node.type)].push(node));
    const maxRows = Math.max(...groups.map((group) => group.length));
    const width = 1010;
    const height = Math.max(310, maxRows * 92 + 80);
    const svg = createSvg("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": `证据链图谱，包含 ${graph.nodes.length} 个节点和 ${graph.edges.length} 条关系` });
    const title = createSvg("title");
    title.textContent = "证据链图谱";
    svg.appendChild(title);
    const desc = createSvg("desc");
    desc.textContent = "图中节点依次表示任务、患者或知识证据、结论及最终报告；箭头指示支持或生成关系。";
    svg.appendChild(desc);
    const defs = createSvg("defs");
    const marker = createSvg("marker", { id: "evidence-arrow", viewBox: "0 0 10 10", refX: "9", refY: "5", markerWidth: "6", markerHeight: "6", orient: "auto-start-reverse" });
    marker.appendChild(createSvg("path", { d: "M 0 0 L 10 5 L 0 10 z", fill: "#789286" }));
    defs.appendChild(marker);
    svg.appendChild(defs);
    const positions = new Map();
    groups.forEach((group, column) => {
      const x = 25 + column * 250;
      const top = (height - group.length * 72) / 2;
      group.forEach((node, index) => positions.set(node.id, { x, y: top + index * 82 }));
    });
    const edgeLayer = createSvg("g");
    graph.edges.forEach((edge) => {
      const from = positions.get(edge.source);
      const to = positions.get(edge.target);
      if (!from || !to) return;
      const direction = to.x >= from.x ? 1 : -1;
      edgeLayer.appendChild(createSvg("line", {
        x1: from.x + (direction > 0 ? 206 : 0), y1: from.y + 29,
        x2: to.x + (direction > 0 ? -8 : 214), y2: to.y + 29,
        stroke: "#789286", "stroke-width": "1.5", "stroke-opacity": "0.86", "marker-end": "url(#evidence-arrow)"
      }));
      if (edge.label) {
        addSvgText(edgeLayer, (from.x + to.x + 206) / 2, (from.y + to.y) / 2 + 24, truncate(edge.label, 10), { fill: "#60766a", "font-size": "9", "text-anchor": "middle" });
      }
    });
    svg.appendChild(edgeLayer);
    graph.nodes.forEach((node) => {
      const pos = positions.get(node.id);
      const colors = graphColor(node.type);
      const group = createSvg("g", { tabindex: "0", role: "img", "aria-label": `${node.type} 节点 ${node.id}：${node.label}${node.detail ? `，${node.detail}` : ""}` });
      group.appendChild(createSvg("rect", { x: pos.x, y: pos.y, width: "206", height: "58", rx: "10", fill: colors.fill, stroke: colors.stroke, "stroke-width": "1.2" }));
      addSvgText(group, pos.x + 10, pos.y + 19, node.id, { fill: colors.text, "font-size": "11", "font-weight": "700" });
      addSvgText(group, pos.x + 10, pos.y + 40, truncate(node.label, 27), { fill: "#1f3128", "font-size": "10.5" });
      const nodeTitle = createSvg("title");
      nodeTitle.textContent = `${node.id}：${node.label}`;
      group.appendChild(nodeTitle);
      svg.appendChild(group);
      const summary = document.createElement("li");
      summary.textContent = `${node.id}（${node.type}）：${node.label}`;
      elements.graphSummary.appendChild(summary);
    });
    elements.graphCanvas.appendChild(svg);
  }

  function runDetails(run, overallStatus = "") {
    if (!run || typeof run !== "object") return "运行已完成。";
    const id = text(run.id || run.runId);
    const status = text(run.status || run.state || overallStatus);
    const round = run.round ?? run.iteration ?? run.repairRound;
    const fragments = ["运行已完成"];
    if (id) fragments.push(`运行 ID：${id}`);
    if (status) fragments.push(`状态：${status}`);
    if (round != null) fragments.push(`修复轮次：${round}`);
    return fragments.join(" · ");
  }

  function renderResponse(response) {
    const run = response?.run || {};
    const evidence = extractEvidence(response);
    const tasks = extractTasks(run);
    renderReport(response?.report, evidence);
    renderTaskDag(tasks);
    renderEvidenceGraph(normalizeGraph(response, tasks, evidence));
    elements.results.hidden = false;
    elements.results.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  async function submitRun(event) {
    event.preventDefault();
    clearErrors();
    const payload = parseInput();
    if (!payload) {
      setState("请检查输入", "error");
      elements.runSummary.textContent = "存在需要修正的字段；数据不会被提交。";
      return;
    }
    setBusy(true);
    setState("运行中", "running");
    elements.runSummary.textContent = "正在校验输入并提交运行请求…";
    startProgress();
    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify(payload)
      });
      const body = await response.json().catch(() => null);
      if (!response.ok) {
        const reason = text(body?.error?.message || body?.message || body?.error || `请求失败（HTTP ${response.status}）`);
        throw new Error(reason);
      }
      if (!body || typeof body !== "object") throw new Error("服务端未返回有效 JSON。\n");
      clearTimers();
      setSteps("report", "success");
      const status = text(body?.run?.status || body?.run?.state || body?.status).toLowerCase();
      const needsReview = /(review|manual|abstain|blocked|warning)/.test(status);
      setState(needsReview ? "需人工复核" : "已完成", needsReview ? "warning" : "success");
      elements.runSummary.textContent = runDetails(body.run, body.status);
      renderResponse(body);
    } catch (error) {
      clearTimers();
      setSteps("execute", "error");
      setState("运行失败", "error");
      elements.runSummary.textContent = `未能完成运行：${error.message || "未知错误"}`;
    } finally {
      setBusy(false);
    }
  }

  function loadDemo() {
    elements.patientRecord.value = DEMO.patientRecord;
    elements.request.value = DEMO.request;
    elements.plan.value = JSON.stringify(DEMO.plan, null, 2);
    const details = document.querySelector(".plan-details");
    if (details) details.open = true;
    clearErrors();
    resetSteps();
    setState("已载入合成示例", "idle");
    elements.runSummary.textContent = "已填入合成病例和示例 DAG；可修改后运行。";
    elements.patientRecord.focus();
  }

  async function copyReport() {
    if (!lastReportText) return;
    try {
      await navigator.clipboard.writeText(lastReportText);
      const original = elements.copyReport.textContent;
      elements.copyReport.textContent = "已复制";
      window.setTimeout(() => { elements.copyReport.textContent = original; }, 1600);
    } catch {
      elements.runSummary.textContent = "浏览器未授予复制权限；请手动选择报告文本复制。";
    }
  }

  elements.form.addEventListener("submit", submitRun);
  elements.loadDemo.addEventListener("click", loadDemo);
  elements.copyReport.addEventListener("click", copyReport);
})();
