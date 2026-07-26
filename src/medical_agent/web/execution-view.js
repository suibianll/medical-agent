"use strict";

import {
  asArray,
  asText,
  identifier,
  makeElement,
  referenceIds,
  truncateText,
  unique
} from "./shared.js";

const MAX_TRACE_ITEMS = 18;

function emptyState() {
  return {
    phase: "idle",
    tasks: new Map(),
    taskOrder: [],
    traces: [],
    traceKeys: new Set(),
    runId: ""
  };
}

export function stageTitle(value) {
  const stage = asText(value, "").toLowerCase();
  if (/(connect|submit|start)/.test(stage)) return "提交任务";
  if (/(plan|规划)/.test(stage)) return "任务规划";
  if (/(react|retriev|search|query|检索)/.test(stage)) return "知识检索";
  if (/(extract|fact|提取)/.test(stage)) return "信息提取";
  if (/(synth|claim|summar|生成)/.test(stage)) return "结论生成";
  if (/(evaluat|verify|audit|评估|核验)/.test(stage)) return "证据核验";
  if (/(model_call|token|cost|成本|调用)/.test(stage)) return "模型调用";
  if (/(repair|revise|修正)/.test(stage)) return "迭代修正";
  if (/(complete|finish|done|result|final|pass)/.test(stage)) return "任务完成";
  if (/(error|fail|reject)/.test(stage)) return "执行异常";
  return stage ? "任务执行" : "等待开始";
}

function stageTone(stage, status = "") {
  const normalized = `${asText(stage)} ${asText(status)}`.toLowerCase();
  if (/(error|fail|reject)/.test(normalized)) return "error";
  if (/(review|warning|manual|blocked|abstain)/.test(normalized)) return "review";
  if (/(complete|finish|done|result|final|pass|success)/.test(normalized)) return "complete";
  if (/(idle|wait)/.test(normalized)) return "idle";
  return "running";
}

function taskStateKey(value) {
  const status = asText(value, "planned").toLowerCase();
  if (/(error|fail|reject|blocked)/.test(status)) return "failed";
  if (/(review|warning|manual|abstain)/.test(status)) return "review";
  if (/(pass|success|complete|done|supported)/.test(status)) return "completed";
  if (/(run|start|retriev|extract|synth|evaluat|repair|progress)/.test(status)) return "running";
  return "planned";
}

function taskStateLabel(value) {
  const key = taskStateKey(value);
  if (key === "completed") return "已完成";
  if (key === "failed") return "失败";
  if (key === "review") return "需复核";
  if (key === "running") return "执行中";
  return "已规划";
}

function auditTimeLabel(timestamp) {
  const date = new Date(timestamp || Date.now());
  if (Number.isNaN(date.getTime())) return "刚刚";
  return new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false
  }).format(date);
}

function factSummary(value) {
  if (typeof value === "string") return value;
  const fact = value && typeof value === "object" ? value : {};
  const reference = identifier(
    fact.evidence_id || fact.evidenceId || fact.ref || fact.source_id || fact.sourceId
  );
  const text = asText(fact.summary || fact.text || fact.fact || fact.content || fact.excerpt || fact.value);
  return reference && text ? `${reference}：${text}` : text || reference;
}

function querySummary(value) {
  if (typeof value === "string") return value;
  const query = value && typeof value === "object" ? value : {};
  return asText(query.query || query.text || query.value || query.content);
}

function evidenceIdSummary(value) {
  if (typeof value === "string" || typeof value === "number") return String(value);
  const evidence = value && typeof value === "object" ? value : {};
  return identifier(evidence.id || evidence.evidence_id || evidence.evidenceId || evidence.ref);
}

function claimsReferenceSummary(value) {
  return asArray(value)
    .slice(0, 4)
    .map((raw, index) => {
      const claim = raw && typeof raw === "object" ? raw : {};
      const id = identifier(claim.id || claim.claim_id || claim.claimId) || `C${index + 1}`;
      const refs = referenceIds(
        claim.refs || claim.references || claim.citations || claim.evidence_ids || claim.evidenceIds
      );
      return `${id}${refs.length ? ` → ${refs.join("、")}` : "（未附引用）"}`;
    })
    .join("；");
}

function evaluationSummary(value) {
  if (typeof value === "string") return value;
  const evaluation = value && typeof value === "object" ? value : {};
  const pass = evaluation.pass ?? evaluation.passed ?? evaluation.ok;
  const issues = asArray(
    evaluation.issues || evaluation.errors || evaluation.unsupported_claims || evaluation.issue_codes
  )
    .slice(0, 3)
    .map((item) => {
      if (typeof item === "string") return item;
      const issue = item && typeof item === "object" ? item : {};
      return asText(issue.code || issue.message || issue.claim || issue.id);
    })
    .filter(Boolean);
  if (pass === true) return "证据链通过自动核验";
  if (pass === false) {
    return issues.length ? `发现待修正项：${issues.join("、")}` : "证据链未通过自动核验";
  }
  return issues.length ? `核验项：${issues.join("、")}` : asText(evaluation.status || evaluation.state);
}

export function createExecutionView(elements) {
  let state = emptyState();

  function dependencyIds(value) {
    if (typeof value === "string") return referenceIds(value);
    return unique(asArray(value).map(identifier));
  }

  function upsertTask(rawTask, fallbackId = "", fallbackStatus = "planned") {
    const raw = rawTask && typeof rawTask === "object" ? rawTask : { goal: asText(rawTask) };
    const id = identifier(raw.id || raw.task_id || raw.taskId) || identifier(fallbackId);
    if (!id) return null;
    const previous = state.tasks.get(id);
    const goal = asText(
      raw.goal || raw.title || raw.name || raw.description || raw.task,
      previous?.goal || `子任务 ${id}`
    );
    const rawDependencies = raw.deps ?? raw.dependencies ?? raw.depends_on ?? raw.dependsOn;
    const deps = rawDependencies == null ? (previous?.deps || []) : dependencyIds(rawDependencies);
    const status = asText(raw.status || raw.state || fallbackStatus, previous?.status || "planned");
    const task = { id, goal, deps, status, error: asText(raw.error, previous?.error || "") };
    if (!previous) state.taskOrder.push(id);
    state.tasks.set(id, task);
    return task;
  }

  function renderTasks() {
    elements.taskProgress.replaceChildren();
    if (!state.taskOrder.length) {
      elements.taskProgress.append(
        makeElement("li", "task-empty", "发送问题后将在此显示规划的子任务及其状态。")
      );
      return;
    }
    state.taskOrder.forEach((id) => {
      const task = state.tasks.get(id);
      if (!task) return;
      const item = makeElement("li", `task-item is-${taskStateKey(task.status)}`);
      const header = makeElement("div", "task-item-header");
      header.append(
        makeElement("span", "task-id", `T${String(task.id).replace(/^T/i, "")}`),
        makeElement("span", "task-state", taskStateLabel(task.status))
      );
      item.append(header, makeElement("p", "task-goal", truncateText(task.goal, 150)));
      if (task.deps.length) {
        const dependencyRow = makeElement("div", "task-dependency-row");
        dependencyRow.append(makeElement("span", "task-dependency-label", "前置依赖"));
        task.deps.forEach((dep) => dependencyRow.append(
          makeElement("span", "task-dependency-chip", `T${String(dep).replace(/^T/i, "")}`)
        ));
        dependencyRow.append(makeElement("span", "task-dependency-arrow", "→"));
        item.append(dependencyRow);
      } else {
        item.append(makeElement("p", "task-deps", "无前置依赖，可直接执行"));
      }
      if (task.error) {
        item.append(makeElement("p", "task-deps", `原因：${truncateText(task.error, 120)}`));
      }
      elements.taskProgress.append(item);
    });
  }

  function setStage(stage, status = "") {
    state.phase = asText(stage, "idle");
    elements.executionStage.className = `execution-stage is-${stageTone(stage, status)}`;
    elements.executionStage.textContent = stageTitle(stage);
  }

  function renderTrace() {
    elements.executionTrace.replaceChildren();
    if (!state.traces.length) {
      elements.executionTrace.append(
        makeElement("li", "trace-empty", "将显示任务计划、检索查询、证据编号、事实提取、引用核验和评估结果。")
      );
      return;
    }
    state.traces.forEach((trace) => {
      const item = makeElement("li", `trace-item${trace.tone ? ` is-${trace.tone}` : ""}`);
      const time = makeElement("time", "trace-time", trace.timeLabel);
      if (trace.timestamp) time.dateTime = trace.timestamp;
      item.append(
        makeElement("span", "trace-stage", trace.label),
        time,
        makeElement("span", "trace-detail", trace.detail)
      );
      elements.executionTrace.append(item);
    });
  }

  function recordTrace(label, detail, tone = "", timestamp = "") {
    const safeDetail = truncateText(detail, 360);
    if (!safeDetail) return;
    const key = `${label}|${safeDetail}`;
    if (state.traceKeys.has(key)) return;
    state.traceKeys.add(key);
    const effectiveTimestamp = asText(timestamp) || new Date().toISOString();
    state.traces.push({
      label,
      detail: safeDetail,
      tone,
      timestamp: effectiveTimestamp,
      timeLabel: auditTimeLabel(effectiveTimestamp)
    });
    if (state.traces.length > MAX_TRACE_ITEMS) state.traces.shift();
    renderTrace();
  }

  function valuesForSummary(value, formatter, maximum = 3) {
    return asArray(value)
      .slice(0, maximum)
      .map((item) => truncateText(formatter(item), 140))
      .filter(Boolean);
  }

  function applyTaskProgress(payload, stage) {
    const tasks = asArray(payload.tasks || (payload.plan && payload.plan.tasks));
    tasks.forEach((task, index) => upsertTask(task, `T${index + 1}`, "planned"));
    if (payload.task != null) {
      upsertTask(payload.task, payload.task_id || payload.taskId, payload.task_status || payload.status || stage);
    } else if (payload.task_id != null || payload.taskId != null) {
      upsertTask(
        { id: payload.task_id ?? payload.taskId, status: payload.task_status || payload.status || stage },
        "",
        payload.task_status || payload.status || stage
      );
    }
    renderTasks();
  }

  function appendSummary(payload, stage, timestamp = "") {
    const safeStage = stageTitle(stage);
    const planTasks = asArray(payload.tasks || (payload.plan && payload.plan.tasks));
    if (planTasks.length) recordTrace("任务计划", `已规划 ${planTasks.length} 个可追踪子任务。`, "", timestamp);
    const queries = valuesForSummary(payload.queries || payload.query, querySummary);
    if (queries.length) recordTrace("检索查询", queries.join("；"), "", timestamp);
    const evidenceIds = valuesForSummary(
      payload.evidence_ids || payload.evidenceIds || payload.evidence,
      evidenceIdSummary,
      6
    );
    if (evidenceIds.length) recordTrace("证据编号", evidenceIds.join("、"), "", timestamp);
    const facts = valuesForSummary(
      payload.facts || payload.extracted_facts || payload.extractedFacts,
      factSummary
    );
    if (facts.length) recordTrace("事实提取", facts.join("；"), "", timestamp);
    const claimRefs = claimsReferenceSummary(payload.claims || payload.claim_refs || payload.claimRefs);
    if (claimRefs) recordTrace("结论引用", claimRefs, "", timestamp);
    const evaluation = evaluationSummary(payload.evaluation || payload.audit);
    if (evaluation) recordTrace("证据评估", evaluation, /未通过|待修正/.test(evaluation) ? "warning" : "", timestamp);
    const metrics = payload.metrics && typeof payload.metrics === "object" ? payload.metrics : null;
    if (metrics) {
      if (metrics.stages && typeof metrics.stages === "object") {
        recordTrace("成本统计", `${asText(metrics.calls, "0")} 次调用；输入 ${asText(metrics.input_tokens, "-")} / 输出 ${asText(metrics.output_tokens, "-")} tokens`, "", timestamp);
      } else {
        const tokens = metrics.total_tokens != null ? `，${metrics.total_tokens} tokens` : "";
        const latency = metrics.latency_ms != null ? `，${metrics.latency_ms} ms` : "";
        recordTrace("模型调用", `${asText(metrics.stage, "unknown")}：${asText(metrics.model, "model")}${tokens}${latency}`, "", timestamp);
      }
    }
    const decision = payload.decision && typeof payload.decision === "object" ? payload.decision : null;
    if (decision && decision.outcome) recordTrace("风险路由", `${asText(decision.outcome)}（${asText(decision.risk_level, "standard")}）`, /escalation|defer/.test(decision.outcome) ? "warning" : "", timestamp);
    const retrieval = payload.retrieval && typeof payload.retrieval === "object" ? payload.retrieval : null;
    if (retrieval && retrieval.stop_reason) {
      recordTrace("检索停止", `${asText(retrieval.stop_reason)}；候选 ${asText(retrieval.candidate_count, "0")} 条`, "", timestamp);
    }
    if (payload.round != null) recordTrace("修正轮次", `第 ${payload.round} 轮证据核验或修正。`, "", timestamp);
    const hasStructured = planTasks.length || queries.length || evidenceIds.length || facts.length || claimRefs || evaluation;
    if (!hasStructured && payload.message) recordTrace(safeStage, asText(payload.message), "", timestamp);
  }

  function applyProgressEvent(payload, eventName = "progress") {
    const data = payload && typeof payload === "object" ? payload : { message: asText(payload) };
    const stage = asText(data.stage || data.phase || data.type || eventName, "progress");
    if (data.run_id || data.runId) state.runId = asText(data.run_id || data.runId);
    setStage(stage, data.status || data.state);
    applyTaskProgress(data, stage);
    appendSummary(data, stage, data.timestamp);
  }

  function hydrateFromResult(result) {
    const data = result && typeof result === "object" ? result : {};
    const run = data.run && typeof data.run === "object" ? data.run : {};
    state.runId = asText(run.id || run.run_id || data.run_id || data.runId, state.runId);
    const tasks = asArray(run.tasks || data.tasks || (run.plan && run.plan.tasks));
    tasks.forEach((task, index) => upsertTask(task, `T${index + 1}`, task.status || "completed"));
    const taskStates = run.task_states || run.taskStates || data.task_states;
    if (taskStates && typeof taskStates === "object" && !Array.isArray(taskStates)) {
      Object.entries(taskStates).forEach(([id, taskState]) => {
        const source = taskState && typeof taskState === "object" ? taskState : {};
        upsertTask(
          { id, ...(source.task && typeof source.task === "object" ? source.task : {}), status: source.status || source.state },
          id,
          source.status || source.state || "completed"
        );
      });
    }
    renderTasks();
    appendSummary({
      tasks,
      claims: data.claims,
      evidence: data.evidence,
      evaluation: run.evaluation || data.evaluation,
      metrics: run.model_usage,
      decision: run.decision,
      round: run.repair_history?.length || data.repair_history?.length || undefined
    }, "result");
    const status = asText(data.status || run.status || "completed");
    setStage(/review|warning|manual/.test(status) ? "review" : "complete", status);
  }

  function reset() {
    state = emptyState();
    setStage("idle");
    renderTasks();
    renderTrace();
  }

  return { applyProgressEvent, hydrateFromResult, recordTrace, reset, setStage };
}
