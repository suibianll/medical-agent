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

(() => {
  "use strict";

  const RUNS_API = "/api/runs/";
  const MAX_GRAPH_NODES = 72;

  const elements = {
    loaderForm: document.getElementById("run-loader-form"),
    runId: document.getElementById("run-id"),
    loadRun: document.getElementById("load-run"),
    pageStatus: document.getElementById("run-page-status"),
    reportRunStatus: document.getElementById("report-run-status"),
    reportTemplate: document.getElementById("evidence-report-template"),
    reportTitle: document.getElementById("evidence-report-title"),
    updateReportView: document.getElementById("update-report-view"),
    reportViewNote: document.getElementById("report-view-note"),
    reportText: document.getElementById("report-text"),
    reportCitationPreview: document.getElementById("report-citation-preview"),
    sentenceCoverage: document.getElementById("sentence-coverage"),
    sentenceProofCount: document.getElementById("sentence-proof-count"),
    sentenceProofList: document.getElementById("sentence-proof-list"),
    auditCount: document.getElementById("audit-count"),
    auditLog: document.getElementById("run-audit-log"),
    dagCount: document.getElementById("dag-count"),
    dagCaption: document.getElementById("dag-caption"),
    taskDependencyMap: document.getElementById("task-dependency-map"),
    runDag: document.getElementById("run-dag"),
    nodeDetail: document.getElementById("run-node-detail"),
    evidenceCount: document.getElementById("run-evidence-count"),
    evidenceList: document.getElementById("run-evidence-list")
  };

  let currentResult = null;
  let currentRunId = "";
  let graphState = emptyGraphState();

  function emptyGraphState() {
    return {
      nodes: [],
      edges: [],
      nodeById: new Map(),
      evidenceById: new Map(),
      claimById: new Map(),
      taskById: new Map()
    };
  }

  function resultFromPayload(payload) {
    if (payload && typeof payload === "object" && payload.result && typeof payload.result === "object") {
      return payload.result;
    }
    return payload && typeof payload === "object" ? payload : {};
  }

  function statusKey(value) {
    const status = asText(value, "").toLowerCase();
    if (/(error|fail|reject)/.test(status)) return "error";
    if (/(review|warning|manual|blocked|abstain|needs_repair)/.test(status)) return "review";
    if (/(pass|success|complete|done|supported)/.test(status)) return "passed";
    if (/(run|start|progress|pending)/.test(status)) return "running";
    return "idle";
  }

  function statusLabel(value) {
    const key = statusKey(value);
    if (key === "passed") return "已完成";
    if (key === "review") return "需复核";
    if (key === "error") return "异常";
    if (key === "running") return "执行中";
    return asText(value, "等待加载");
  }

  function setRunStatus(status) {
    const key = statusKey(status);
    elements.reportRunStatus.className = `run-status is-${key}`;
    elements.reportRunStatus.textContent = statusLabel(status);
  }

  function setPageStatus(message, isError = false) {
    elements.pageStatus.textContent = message;
    elements.pageStatus.classList.toggle("is-error", isError);
  }

  function runUrl(runId, suffix = "") {
    return `${RUNS_API}${encodeURIComponent(runId)}${suffix}`;
  }

  function readStoredRun(runId) {
    try {
      const raw = sessionStorage.getItem(`medical-agent-run:${runId}`);
      return raw ? JSON.parse(raw) : null;
    } catch {
      return null;
    }
  }

  function applyStoredReportContext(runId) {
    try {
      const raw = sessionStorage.getItem(`medical-agent-run-context:${runId}`);
      const context = raw ? JSON.parse(raw) : null;
      const template = context?.reportTemplate;
      if (typeof template === "string") elements.reportTemplate.value = template;
      if (template && typeof template === "object") {
        if (asText(template.name)) elements.reportTemplate.value = template.name;
        if (asText(template.title)) elements.reportTitle.value = template.title;
      }
    } catch {
      // The result endpoint remains the source of truth when session storage is unavailable.
    }
  }

  async function fetchRunEvents(runId, result) {
    try {
      const payload = await requestJson(runUrl(runId, "/events"));
      return asArray(payload.events || payload.items || payload.data);
    } catch {
      const run = result?.run && typeof result.run === "object" ? result.run : {};
      return asArray(result?.events || result?.audit_events || run.events || run.audit_events);
    }
  }

  function normalizeEvidence(rawEvidence) {
    return asArray(rawEvidence).map((raw, index) => {
      if (typeof raw === "string") {
        return { id: `E${index + 1}`, source: "资料来源", locator: "", text: raw, kind: "evidence" };
      }
      const item = raw && typeof raw === "object" ? raw : {};
      return {
        id: identifier(item.id || item.evidence_id || item.evidenceId || item.ref) || `E${index + 1}`,
        source: asText(item.source || item.title || item.document || item.name, "资料来源"),
        locator: asText(item.locator || item.location || item.section || item.document_id || item.documentId),
        text: asText(item.text || item.content || item.excerpt || item.quote || item.detail),
        kind: asText(item.kind || item.type || item.category, "evidence"),
        span: item.span && typeof item.span === "object" ? item.span : {},
        contentHash: asText(item.content_hash || item.contentHash),
        metadata: item.metadata && typeof item.metadata === "object" ? item.metadata : {}
      };
    });
  }

  function normalizeClaims(rawClaims) {
    return asArray(rawClaims).map((raw, index) => {
      if (typeof raw === "string") return { id: `C${index + 1}`, text: raw, refs: [], taskId: "", status: "supported" };
      const claim = raw && typeof raw === "object" ? raw : {};
      return {
        id: identifier(claim.id || claim.claim_id || claim.claimId) || `C${index + 1}`,
        text: asText(claim.text || claim.claim || claim.conclusion || claim.summary || claim.answer, "未提供结论文本"),
        refs: unique(referenceIds(claim.refs || claim.references || claim.citations || claim.evidence_ids || claim.evidenceIds)),
        supportEdges: asArray(claim.support_edges || claim.supportEdges),
        supportSummary: claim.support_summary && typeof claim.support_summary === "object" ? claim.support_summary : {},
        taskId: identifier(claim.task_id || claim.taskId),
        status: asText(claim.status || claim.evaluation || claim.state, "supported")
      };
    });
  }

  function normalizeTasks(result) {
    const run = result?.run && typeof result.run === "object" ? result.run : {};
    const rawTasks = asArray(run.tasks || result?.tasks || (run.plan && run.plan.tasks));
    return rawTasks.map((raw, index) => {
      const task = raw && typeof raw === "object" ? raw : {};
      const id = identifier(task.id || task.task_id || task.taskId) || String(index + 1);
      const rawDeps = task.deps ?? task.dependencies ?? task.depends_on ?? task.dependsOn;
      const deps = typeof rawDeps === "string" ? referenceIds(rawDeps) : asArray(rawDeps).map(identifier).filter(Boolean);
      return {
        id,
        goal: asText(task.goal || task.title || task.name || task.description, `子任务 ${id}`),
        deps,
        status: asText(task.status || task.state, "completed"),
        error: asText(task.error)
      };
    });
  }

  function taskNodeId(value) {
    return `T${String(value).replace(/^T/i, "")}`;
  }

  function taskLevels(tasks) {
    const byId = new Map(tasks.map((task) => [taskNodeId(task.id), task]));
    const memo = new Map();
    const visiting = new Set();
    function levelFor(task) {
      const id = taskNodeId(task.id);
      if (memo.has(id)) return memo.get(id);
      if (visiting.has(id)) return 0;
      visiting.add(id);
      const parentLevels = task.deps
        .map((dependency) => byId.get(taskNodeId(dependency)))
        .filter(Boolean)
        .map(levelFor);
      visiting.delete(id);
      const level = parentLevels.length ? Math.max(...parentLevels) + 1 : 0;
      memo.set(id, level);
      return level;
    }
    tasks.forEach(levelFor);
    return memo;
  }

  function renderTaskDependencyMap(tasks, claims) {
    elements.taskDependencyMap.replaceChildren();
    if (!tasks.length) {
      elements.taskDependencyMap.append(makeElement("p", "graph-empty", "当前运行没有可展示的子任务。"));
      return;
    }
    const levels = taskLevels(tasks);
    const maximumLevel = Math.max(0, ...levels.values());
    const claimsByTask = new Map();
    claims.forEach((claim) => {
      const taskId = taskNodeId(claim.taskId || "");
      if (taskId === "T") return;
      if (!claimsByTask.has(taskId)) claimsByTask.set(taskId, []);
      claimsByTask.get(taskId).push(claim);
    });
    for (let level = 0; level <= maximumLevel; level += 1) {
      const column = makeElement("section", "task-wave");
      column.setAttribute("aria-label", `执行阶段 ${level + 1}`);
      const waveHeader = makeElement("div", "task-wave-header");
      waveHeader.append(
        makeElement("span", "task-wave-number", String(level + 1)),
        makeElement("span", "task-wave-label", level === 0 ? "起始任务" : `第 ${level + 1} 阶段`)
      );
      column.append(waveHeader);
      tasks.filter((task) => levels.get(taskNodeId(task.id)) === level).forEach((task) => {
        const id = taskNodeId(task.id);
        const button = makeElement("button", `task-lineage-card is-${statusKey(task.status)}`);
        button.type = "button";
        button.dataset.taskId = id;
        button.setAttribute("aria-label", `${id} 子 Agent：${task.goal}`);
        const header = makeElement("span", "task-lineage-header");
        header.append(
          makeElement("strong", "", `${id} · 子 Agent`),
          makeElement("span", "task-lineage-status", statusLabel(task.status))
        );
        button.append(header, makeElement("span", "task-lineage-goal", task.goal));
        if (task.deps.length) {
          button.append(makeElement("span", "task-lineage-deps", `依赖 ${task.deps.map(taskNodeId).join("、")}`));
        } else {
          button.append(makeElement("span", "task-lineage-deps", "无前置依赖"));
        }
        const produced = claimsByTask.get(id) || [];
        if (produced.length) {
          const refs = unique(produced.flatMap((claim) => claim.refs));
          button.append(makeElement("span", "task-lineage-output", `产出 ${produced.length} 条结论 · ${refs.length} 个依据`));
        }
        button.addEventListener("click", () => selectGraphNode(id));
        column.append(button);
      });
      elements.taskDependencyMap.append(column);
    }
  }

  function normalizeGraph(result, tasks, claims, evidence) {
    const rawGraph = result?.graph && typeof result.graph === "object" ? result.graph : {};
    const nodes = [];
    const edges = [];
    const byId = new Map();
    const edgeKeys = new Set();

    function addNode(raw, fallbackId, fallbackType = "reference", fallbackLabel = "关联节点") {
      const source = raw && typeof raw === "object" ? raw : { id: raw };
      const id = identifier(source.id || source.node_id || source.nodeId) || identifier(fallbackId);
      if (!id) return null;
      if (byId.has(id)) return byId.get(id);
      const node = {
        id,
        type: graphNodeType(source.type || source.kind || fallbackType, "report"),
        label: graphNodeLabel(source, fallbackLabel || id),
        status: asText(source.status || source.state),
        supportSummary: source.support_summary && typeof source.support_summary === "object" ? source.support_summary : {}
      };
      byId.set(id, node);
      nodes.push(node);
      return node;
    }

    function addEdge(rawSource, rawTarget, type = "supports", metadata = {}) {
      const source = identifier(rawSource);
      const target = identifier(rawTarget);
      if (!source || !target || source === target) return;
      if (!byId.has(source)) addNode({ id: source, label: source }, source);
      if (!byId.has(target)) addNode({ id: target, label: target }, target);
      const key = `${source}→${target}→${asText(type, "supports")}`;
      if (edgeKeys.has(key)) return;
      edgeKeys.add(key);
      edges.push({
        id: key,
        source,
        target,
        type: asText(type, "supports"),
        relation: asText(metadata.relation || type, "supports"),
        claimId: identifier(metadata.claim_id || metadata.claimId),
        verifier: asText(metadata.verifier),
        verifierScore: metadata.verifier_score ?? metadata.verifierScore
      });
    }

    asArray(rawGraph.nodes || rawGraph.vertices || rawGraph.items).forEach((node, index) => addNode(node, `N${index + 1}`));
    asArray(rawGraph.edges || rawGraph.links || rawGraph.relationships).forEach((edge) => {
      if (!edge || typeof edge !== "object") return;
      addEdge(
        edge.from || edge.source || edge.start || edge.parent,
        edge.to || edge.target || edge.end || edge.child,
        edge.type || edge.relation,
        edge
      );
    });

    tasks.forEach((task) => {
      const taskId = `T${String(task.id).replace(/^T/i, "")}`;
      addNode({ id: taskId, type: "task", label: `${taskId}：${task.goal}`, status: task.status }, taskId, "task");
      task.deps.forEach((dep) => addEdge(`T${String(dep).replace(/^T/i, "")}`, taskId, "depends_on"));
    });
    evidence.forEach((item) => addNode(
      { id: item.id, type: item.kind, label: `${item.id}：${item.source}` },
      item.id,
      item.kind,
      `${item.id}：${item.source}`
    ));
    claims.forEach((claim) => {
      addNode({ id: claim.id, type: "claim", label: claim.text, status: claim.status, support_summary: claim.supportSummary }, claim.id, "claim", claim.text);
      if (claim.taskId) addEdge(`T${String(claim.taskId).replace(/^T/i, "")}`, claim.id, "produces");
      const supportEdges = claim.supportEdges.length
        ? claim.supportEdges
        : claim.refs.map((ref) => ({ evidence_id: ref, relation: "supports" }));
      supportEdges.forEach((supportEdge) => {
        const ref = identifier(supportEdge?.evidence_id || supportEdge?.evidenceId);
        if (ref) addEdge(ref, claim.id, supportEdge.relation || "supports", { ...supportEdge, claim_id: claim.id });
      });
    });
    if (claims.length || nodes.length) {
      addNode({ id: "REPORT", type: "report", label: "最终报告" }, "REPORT", "report", "最终报告");
      claims.forEach((claim) => addEdge(claim.id, "REPORT", "reported_in"));
    }

    const visibleNodes = nodes.slice(0, MAX_GRAPH_NODES);
    const visibleIds = new Set(visibleNodes.map((node) => node.id));
    return { nodes: visibleNodes, edges: edges.filter((edge) => visibleIds.has(edge.source) && visibleIds.has(edge.target)) };
  }

  function nodeLabelLines(value, limit = 18) {
    const compact = asText(value, "关联节点").replace(/\s+/g, " ");
    if (compact.length <= limit) return [compact];
    return [compact.slice(0, limit), `${compact.slice(limit, limit * 2 - 1)}…`];
  }

  function renderRunGraph(result, tasks, claims, evidence) {
    const graph = normalizeGraph(result, tasks, claims, evidence);
    graphState = {
      ...graph,
      nodeById: new Map(graph.nodes.map((node) => [node.id, node])),
      evidenceById: new Map(evidence.map((item) => [item.id, item])),
      claimById: new Map(claims.map((item) => [item.id, item])),
      taskById: new Map(tasks.map((item) => [`T${String(item.id).replace(/^T/i, "")}`, item]))
    };

    elements.runDag.replaceChildren();
    elements.dagCount.textContent = String(graph.nodes.length);
    if (!graph.nodes.length) {
      elements.runDag.append(makeElement("p", "graph-empty", "当前运行未返回可绘制的任务或证据关系。"));
      elements.dagCaption.textContent = "没有可用的图节点。";
      resetNodeDetail();
      return;
    }

    const layout = graphLayout(graph.nodes, {
      width: 930,
      minimumHeight: 292,
      nodeHeight: 72,
      rowGap: 22,
      topPadding: 22,
      xByLayer: [110, 344, 586, 826],
      widths: { report: 150, claim: 206, default: 184 }
    });
    const svg = makeSvgElement("svg", {
      viewBox: `0 0 ${layout.width} ${layout.height}`,
      role: "group",
      "aria-labelledby": "run-dag-svg-title run-dag-svg-desc"
    });
    const title = makeSvgElement("title", { id: "run-dag-svg-title" });
    title.textContent = "任务依赖与证据支持关系";
    const description = makeSvgElement("desc", { id: "run-dag-svg-desc" });
    description.textContent = `显示 ${graph.nodes.length} 个节点和 ${graph.edges.length} 条关联。`;
    const defs = makeSvgElement("defs");
    const marker = makeSvgElement("marker", {
      id: "run-dag-arrow",
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
      const bend = Math.max(44, Math.abs(endX - startX) * 0.45) * direction;
      svg.append(makeSvgElement("path", {
        class: `run-graph-edge edge-${edge.type}`,
        d: `M ${startX} ${source.y} C ${startX + bend} ${source.y}, ${endX - bend} ${target.y}, ${endX} ${target.y}`,
        "data-source": edge.source,
        "data-target": edge.target
      }));
    });

    graph.nodes.forEach((node) => {
      const position = layout.positions.get(node.id);
      const group = makeSvgElement("g", {
        class: `run-graph-node type-${node.type}`,
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
      nodeLabelLines(node.label).forEach((line, index) => {
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
    elements.runDag.append(svg);
    const conflictCount = Number(result?.graph?.summary?.conflict_count || 0);
    elements.dagCaption.textContent = conflictCount
      ? `当前运行包含 ${graph.nodes.length} 个节点、${graph.edges.length} 条关联，其中 ${conflictCount} 个结论存在支持与冲突证据。选择节点查看关系。`
      : `当前运行包含 ${graph.nodes.length} 个节点、${graph.edges.length} 条关联。选择节点可查看状态、来源和支持关系。`;
    resetNodeDetail();
  }

  function resetNodeDetail() {
    elements.nodeDetail.textContent = "选择图中的节点，查看它的状态、来源和支持关系。";
  }

  function selectGraphNode(nodeId) {
    const node = graphState.nodeById.get(nodeId);
    if (!node) return;
    elements.runDag.querySelectorAll(".run-graph-node").forEach((candidate) => {
      candidate.classList.toggle("is-selected", candidate.dataset.nodeId === nodeId);
    });
    elements.runDag.querySelectorAll(".run-graph-edge").forEach((edge) => {
      edge.classList.toggle("is-highlighted", edge.dataset.source === nodeId || edge.dataset.target === nodeId);
    });
    elements.taskDependencyMap.querySelectorAll(".task-lineage-card").forEach((card) => {
      card.classList.toggle("is-selected", card.dataset.taskId === nodeId);
    });
    elements.sentenceProofList.querySelectorAll(".sentence-proof-item").forEach((item) => {
      const references = (item.dataset.references || "").split(" ").filter(Boolean);
      item.classList.toggle("is-focused", references.includes(nodeId) || item.dataset.taskId === nodeId);
    });

    const fragment = document.createDocumentFragment();
    fragment.append(makeElement("strong", "", `${node.id} · ${node.label}`));
    const evidence = graphState.evidenceById.get(nodeId);
    const claim = graphState.claimById.get(nodeId);
    const task = graphState.taskById.get(nodeId);
    if (evidence) {
      fragment.append(makeElement("p", "detail-source", `来源：${evidence.source}`));
      if (evidence.locator) fragment.append(makeElement("p", "", `定位：${evidence.locator}`));
      if (evidence.span && evidence.span.id) {
        fragment.append(makeElement("p", "", `证据片段：${evidence.span.id}（${evidence.span.start ?? 0}-${evidence.span.end ?? 0}）`));
      }
      if (evidence.contentHash) fragment.append(makeElement("p", "", `内容哈希：${evidence.contentHash.slice(0, 16)}…`));
      if (evidence.text) fragment.append(makeElement("p", "", evidence.text));
    } else if (claim) {
      fragment.append(makeElement("p", "", claim.refs.length ? `引用：${claim.refs.map((ref) => `[${ref}]`).join(" ")}` : "该结论未附引用标识。"));
      fragment.append(makeElement("p", "", `状态：${statusLabel(claim.status)}`));
      if (claim.supportEdges.length) {
        const relations = claim.supportEdges.map((edge) => `${asText(edge.evidence_id || edge.evidenceId)}：${asText(edge.relation, "supports")}`);
        fragment.append(makeElement("p", "", `验证边：${relations.join("；")}`));
      }
      const claimRelations = graphState.edges
        .filter((edge) => edge.target === nodeId && ["supports", "qualifies", "contradicts"].includes(edge.type))
        .map((edge) => edge.type);
      if (claim.supportSummary.has_conflict || (claimRelations.includes("contradicts") && claimRelations.some((relation) => ["supports", "qualifies"].includes(relation)))) {
        fragment.append(makeElement("p", "graph-conflict-note", "该结论同时存在支持与冲突证据，需要人工复核。"));
      }
    } else if (task) {
      fragment.append(makeElement("p", "", `状态：${statusLabel(task.status)}`));
      if (task.deps.length) fragment.append(makeElement("p", "", `依赖：${task.deps.map((dep) => `T${String(dep).replace(/^T/i, "")}`).join("、")}`));
      if (task.error) fragment.append(makeElement("p", "", `错误：${task.error}`));
    } else {
      const relationships = graphState.edges
        .filter((edge) => edge.source === nodeId || edge.target === nodeId)
        .map((edge) => `${edge.source} → ${edge.target}`);
      fragment.append(makeElement("p", "", relationships.length ? `关联：${relationships.join("；")}` : "该节点没有可展示的关联。"));
    }
    elements.nodeDetail.replaceChildren(fragment);
  }

  function renderEvidenceList(evidence) {
    elements.evidenceList.replaceChildren();
    elements.evidenceCount.textContent = String(evidence.length);
    if (!evidence.length) {
      elements.evidenceList.append(makeElement("li", "list-state", "本次运行未返回单独的证据条目。"));
      return;
    }
    evidence.forEach((item) => {
      const listItem = makeElement("li");
      const button = makeElement("button");
      button.type = "button";
      button.append(makeElement("span", "evidence-id", item.id), document.createTextNode(item.source));
      if (item.locator) button.append(makeElement("span", "evidence-source", `定位：${item.locator}`));
      if (item.text) button.append(makeElement("span", "evidence-source", truncateText(item.text, 220)));
      button.addEventListener("click", () => selectGraphNode(item.id));
      listItem.append(button);
      elements.evidenceList.append(listItem);
    });
  }

  function stageLabel(value) {
    const stage = asText(value, "").toLowerCase();
    if (/(plan|规划)/.test(stage)) return "任务规划";
    if (/(query|retriev|search|检索)/.test(stage)) return "知识检索";
    if (/(extract|fact|提取)/.test(stage)) return "信息提取";
    if (/(synth|claim|生成)/.test(stage)) return "结论生成";
    if (/(evaluat|verify|audit|核验|评估)/.test(stage)) return "证据核验";
    if (/(repair|revise|修正)/.test(stage)) return "迭代修正";
    if (/(complete|done|result|final|完成)/.test(stage)) return "任务完成";
    if (/(error|fail|reject)/.test(stage)) return "执行异常";
    if (/(task)/.test(stage)) return "子任务执行";
    return "运行事件";
  }

  function auditTime(value) {
    const date = new Date(value || Date.now());
    if (Number.isNaN(date.getTime())) return "刚刚";
    return new Intl.DateTimeFormat("zh-CN", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false }).format(date);
  }

  function safeAuditDetail(raw) {
    const event = raw && typeof raw === "object" ? raw : {};
    const details = [];
    if (asText(event.message)) details.push(asText(event.message));
    const queries = asArray(event.queries).filter((item) => typeof item === "string").slice(0, 3);
    if (queries.length) details.push(`查询：${queries.map((query) => truncateText(query, 100)).join("；")}`);
    const evidenceIds = asArray(event.evidence_ids || event.evidenceIds).map(identifier).filter(Boolean).slice(0, 8);
    if (evidenceIds.length) details.push(`证据：${evidenceIds.join("、")}`);
    const facts = asArray(event.facts).slice(0, 3).map((fact) => {
      if (!fact || typeof fact !== "object") return "";
      const ref = identifier(fact.ref || fact.evidence_id || fact.evidenceId);
      const summary = asText(fact.summary || fact.text || fact.fact);
      return ref && summary ? `${ref}：${truncateText(summary, 120)}` : summary || ref;
    }).filter(Boolean);
    if (facts.length) details.push(`事实：${facts.join("；")}`);
    const claims = asArray(event.claims).slice(0, 4).map((claim, index) => {
      const refs = referenceIds(claim && typeof claim === "object" ? claim.refs : claim);
      return refs.length ? `结论${index + 1}→${refs.join("、")}` : "";
    }).filter(Boolean);
    if (claims.length) details.push(claims.join("；"));
    const evaluation = event.evaluation && typeof event.evaluation === "object" ? event.evaluation : null;
    if (evaluation) {
      const issueCodes = asArray(evaluation.issue_codes || evaluation.issues).map((issue) => typeof issue === "string" ? issue : asText(issue?.code)).filter(Boolean).slice(0, 4);
      details.push(evaluation.pass === true ? "证据链通过自动核验" : issueCodes.length ? `待修正：${issueCodes.join("、")}` : "证据链正在或尚未通过核验");
    }
    return details.length ? details.join(" · ") : "服务端未返回可展示的安全摘要。";
  }

  function renderAuditLog(events, result) {
    const safeEvents = asArray(events).filter((event) => event && typeof event === "object");
    if (!safeEvents.length) {
      const run = result?.run && typeof result.run === "object" ? result.run : {};
      const tasks = normalizeTasks(result);
      if (tasks.length) {
        safeEvents.push({
          stage: "completed",
          timestamp: run.created_at,
          message: `已加载 ${tasks.length} 个任务的最终状态。`,
          evaluation: run.evaluation
        });
      }
    }
    elements.auditLog.replaceChildren();
    elements.auditCount.textContent = String(safeEvents.length);
    if (!safeEvents.length) {
      elements.auditLog.append(makeElement("li", "trace-empty", "该运行暂未保存可展示的审计事件。"));
      return;
    }
    safeEvents.forEach((event) => {
      const status = asText(event.status || event.stage);
      const item = makeElement("li", `run-audit-item is-${statusKey(status)}`);
      const header = makeElement("div", "run-audit-header");
      const time = makeElement("time", "trace-time", auditTime(event.timestamp || event.created_at));
      if (asText(event.timestamp || event.created_at)) time.dateTime = asText(event.timestamp || event.created_at);
      header.append(makeElement("span", "trace-stage", stageLabel(event.stage)), time);
      item.append(header, makeElement("p", "run-audit-detail", safeAuditDetail(event)));
      elements.auditLog.append(item);
    });
  }

  function reportTemplateLabel(template) {
    if (template === "task_trace") return "任务与证据链追踪";
    if (template === "handoff") return "人工复核交接单";
    return "证据核验摘要";
  }

  function applyResultReportTemplate(result) {
    const report = result?.report && typeof result.report === "object" ? result.report : {};
    const template = report.template ?? result?.reportTemplate;
    if (typeof template === "string" && template) {
      elements.reportTemplate.value = template;
      return;
    }
    if (template && typeof template === "object") {
      if (asText(template.name)) elements.reportTemplate.value = template.name;
      if (asText(template.title)) elements.reportTitle.value = template.title;
    }
  }

  function reportTextFromResult(result) {
    const report = result?.report;
    if (typeof report === "string" && report.trim()) return report.trim();
    if (report && typeof report === "object") {
      const text = asText(report.text || report.markdown || report.content || report.answer || report.summary);
      if (text) return text;
    }
    const claims = normalizeClaims(result?.claims);
    const evidence = normalizeEvidence(result?.evidence);
    const lines = ["# 医疗 Agent 证据报告", "", "## 结论"];
    if (claims.length) {
      claims.forEach((claim) => lines.push(`- ${claim.text} ${claim.refs.map((ref) => `[${ref}]`).join(" ")}`));
    } else {
      lines.push("- 当前运行未形成可引用结论。\n");
    }
    lines.push("", "## 证据明细");
    evidence.forEach((item) => lines.push(`- [${item.id}] ${item.source}${item.locator ? `（${item.locator}）` : ""}`));
    return lines.join("\n");
  }

  function reportEvidenceForReference(referenceId) {
    return graphState.evidenceById.get(referenceId)
      || normalizeEvidence(currentResult?.evidence).find((item) => item.id === referenceId)
      || null;
  }

  function hideReportCitationPreview() {
    if (elements.reportCitationPreview) elements.reportCitationPreview.hidden = true;
  }

  function showReportCitationPreview(button, referenceId) {
    const preview = elements.reportCitationPreview;
    if (!preview) return;
    const evidence = reportEvidenceForReference(referenceId);
    const source = evidence?.source || "本报告证据引用";
    const locator = evidence?.locator || "定位信息未单独返回";
    const excerpt = evidence?.text || "该引用的原文摘要可在右侧证据明细中查看。";
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

  function makeReportCitationButton(referenceId) {
    const button = makeElement("button", "citation-button", `[${referenceId}]`);
    button.type = "button";
    button.setAttribute("aria-label", `查看报告中的证据引用 ${referenceId}`);
    button.setAttribute("aria-describedby", "report-citation-preview");
    const preview = () => {
      selectGraphNode(referenceId);
      showReportCitationPreview(button, referenceId);
    };
    button.addEventListener("mouseenter", preview);
    button.addEventListener("focus", preview);
    button.addEventListener("mouseleave", hideReportCitationPreview);
    button.addEventListener("blur", hideReportCitationPreview);
    button.addEventListener("click", preview);
    return button;
  }

  function makeSentenceEvidenceButton(referenceId) {
    const evidence = reportEvidenceForReference(referenceId);
    const label = evidence?.source ? `[${referenceId}] ${truncateText(evidence.source, 34)}` : `[${referenceId}]`;
    const button = makeElement("button", "sentence-evidence-button", label);
    button.type = "button";
    button.setAttribute("aria-label", `查看该句依据 ${referenceId}${evidence?.source ? `，${evidence.source}` : ""}`);
    button.setAttribute("aria-describedby", "report-citation-preview");
    const preview = () => {
      selectGraphNode(referenceId);
      showReportCitationPreview(button, referenceId);
    };
    button.addEventListener("mouseenter", preview);
    button.addEventListener("focus", preview);
    button.addEventListener("mouseleave", hideReportCitationPreview);
    button.addEventListener("blur", hideReportCitationPreview);
    button.addEventListener("click", preview);
    return button;
  }

  function renderSentenceProof(claims) {
    elements.sentenceProofList.replaceChildren();
    const rows = claims.flatMap((claim) => splitSentences(claim.text).map((sentence, index) => ({
      id: `${claim.id}.${index + 1}`,
      claimId: claim.id,
      sentence,
      refs: claim.refs,
      taskId: claim.taskId,
      status: claim.status
    })));
    const supported = rows.filter((row) => row.refs.length > 0).length;
    elements.sentenceProofCount.textContent = `${rows.length} 句`;
    elements.sentenceCoverage.className = `coverage-badge ${rows.length && supported === rows.length ? "is-complete" : "is-review"}`;
    elements.sentenceCoverage.textContent = rows.length
      ? `引用覆盖 ${supported}/${rows.length}`
      : "无结论句";
    if (!rows.length) {
      elements.sentenceProofList.append(makeElement("li", "proof-empty", "当前运行没有形成可展示的结论句。"));
      return;
    }
    rows.forEach((row) => {
      const item = makeElement("li", `sentence-proof-item${row.refs.length ? "" : " is-uncited"}`);
      item.dataset.references = row.refs.join(" ");
      item.dataset.taskId = row.taskId ? taskNodeId(row.taskId) : "";
      const header = makeElement("div", "sentence-proof-header");
      const provenance = makeElement("div", "sentence-provenance");
      provenance.append(makeElement("strong", "", row.id));
      if (row.taskId) provenance.append(makeElement("span", "", `来自 ${taskNodeId(row.taskId)}`));
      const status = makeElement("span", `sentence-status is-${row.refs.length ? "supported" : "uncited"}`, row.refs.length ? "有依据" : "缺少引用");
      header.append(provenance, status);
      const text = makeElement("p", "sentence-proof-text", row.sentence);
      const sources = makeElement("div", "sentence-source-row");
      if (row.refs.length) {
        row.refs.forEach((referenceId) => sources.append(makeSentenceEvidenceButton(referenceId)));
      } else {
        sources.append(makeElement("span", "uncited-message", "该句没有可核验的证据编号，需要人工复核。"));
      }
      item.append(header, text, sources);
      elements.sentenceProofList.append(item);
    });
  }

  function appendReportInline(container, text) {
    const citationPattern = /\[([PK]\d+)\]/g;
    let cursor = 0;
    let match;
    while ((match = citationPattern.exec(text))) {
      if (match.index > cursor) container.append(document.createTextNode(text.slice(cursor, match.index)));
      container.append(makeReportCitationButton(match[1]));
      cursor = match.index + match[0].length;
    }
    if (cursor < text.length) container.append(document.createTextNode(text.slice(cursor)));
  }

  function renderReportText(text) {
    elements.reportText.replaceChildren();
    if (!text) {
      elements.reportText.textContent = "尚未加载文本报告。";
      return;
    }
    const documentView = makeElement("article", "report-document");
    const lines = String(text).split(/\r?\n/);
    let firstContent = true;
    lines.forEach((rawLine, index) => {
      const line = rawLine.trim();
      if (!line || /^-{4,}$/.test(line)) return;
      let element;
      if (firstContent) {
        element = makeElement("h3", "report-document-title");
        firstContent = false;
      } else if (/^-{4,}$/.test((lines[index + 1] || "").trim())) {
        element = makeElement("h4", "report-section-title");
      } else if (/^安全提示[:：]/.test(line)) {
        element = makeElement("aside", "report-safety-line");
      } else if (/^\[[PK]\d+\]\s*/.test(line)) {
        element = makeElement("h5", "report-evidence-title");
      } else if (/^-\s+/.test(line)) {
        element = makeElement("p", "report-list-line");
        element.append(makeElement("span", "report-list-mark", "•"));
      } else {
        element = makeElement("p", "report-paragraph");
      }
      const contentTarget = element.classList.contains("report-list-line")
        ? makeElement("span", "report-list-content")
        : element;
      appendReportInline(contentTarget, line.replace(/^-\s+/, ""));
      if (contentTarget !== element) element.append(contentTarget);
      documentView.append(element);
    });
    elements.reportText.append(documentView);
  }

  function renderReport() {
    if (!currentResult) {
      elements.reportText.textContent = "尚未加载文本报告。";
      return;
    }
    const template = asText(elements.reportTemplate.value, "evidence_summary");
    const customTitle = asText(elements.reportTitle.value);
    const text = reportTextFromResult(currentResult);
    const archivedTemplate = currentResult?.report && typeof currentResult.report === "object"
      ? currentResult.report.template
      : currentResult?.reportTemplate;
    const archivedTitle = asText(
      archivedTemplate && typeof archivedTemplate === "object" ? archivedTemplate.title : ""
    );
    const focus = template === "task_trace"
      ? "阅读重点：任务依赖、各阶段状态与证据支持关系。"
      : template === "handoff"
        ? "阅读重点：待补充信息、异常状态与人工复核交接。"
        : "阅读重点：结论、引用完整性与证据核验状态。";
    const localTitleNote = customTitle && customTitle !== archivedTitle
      ? ` 已修改标题为“${customTitle}”；该修改只会在发起新运行时传给服务端。`
      : archivedTitle
        ? ` 归档运行使用的报告标题：${archivedTitle}。`
        : "";
    elements.reportViewNote.textContent = `${focus} 当前阅读模板：${reportTemplateLabel(template)}。此操作不会重新生成模型内容；下方直接呈现归档的完整文本报告。${localTitleNote}`;
    renderReportText(text);
  }

  function renderRun(result, events) {
    currentResult = result;
    const run = result?.run && typeof result.run === "object" ? result.run : {};
    const status = asText(result.status || run.status, "completed");
    const tasks = normalizeTasks(result);
    const claims = normalizeClaims(result.claims);
    const evidence = normalizeEvidence(result.evidence);
    setRunStatus(status);
    renderAuditLog(events, result);
    renderSentenceProof(claims);
    renderTaskDependencyMap(tasks, claims);
    renderRunGraph(result, tasks, claims, evidence);
    renderEvidenceList(evidence);
    renderReport();
  }

  async function loadRun(runId) {
    const safeRunId = asText(runId);
    if (!safeRunId) {
      setPageStatus("请先输入运行编号。", true);
      elements.runId.focus();
      return;
    }
    currentRunId = safeRunId;
    elements.loadRun.disabled = true;
    setPageStatus("正在读取运行归档和安全审计事件…");
    try {
      let result;
      let usedSessionFallback = false;
      try {
        result = resultFromPayload(await requestJson(runUrl(safeRunId)));
      } catch (error) {
        result = readStoredRun(safeRunId);
        if (!result) throw error;
        usedSessionFallback = true;
      }
      applyResultReportTemplate(result);
      applyStoredReportContext(safeRunId);
      const events = await fetchRunEvents(safeRunId, result);
      renderRun(result, events);
      setPageStatus(
        usedSessionFallback
          ? "服务端短期归档已不可用，正在显示本次浏览会话保留的运行结果。"
          : `已加载运行 ${safeRunId.slice(0, 8)} 的任务、证据、审计日志和文本报告。`
      );
    } catch (error) {
      currentResult = null;
      graphState = emptyGraphState();
      setRunStatus("error");
      elements.reportText.textContent = "无法加载文本报告。";
      elements.runDag.replaceChildren(makeElement("p", "graph-empty", "无法加载运行证据关系。"));
      elements.evidenceList.replaceChildren(makeElement("li", "list-state", "无法加载证据明细。"));
      setPageStatus(`加载失败：${asText(error.message, "请检查运行编号或稍后重试。")}`, true);
    } finally {
      elements.loadRun.disabled = false;
    }
  }

  elements.loaderForm.addEventListener("submit", (event) => {
    event.preventDefault();
    loadRun(elements.runId.value);
  });
  elements.updateReportView.addEventListener("click", renderReport);
  elements.reportTemplate.addEventListener("change", renderReport);

  const params = new URLSearchParams(window.location.search);
  const initialRunId = asText(params.get("runId") || params.get("id"));
  if (initialRunId) {
    elements.runId.value = initialRunId;
    loadRun(initialRunId);
  }
})();
