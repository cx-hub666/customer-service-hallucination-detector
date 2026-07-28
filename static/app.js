const state = { predictions: [], metrics: null };

const byId = (id) => document.getElementById(id);
const escapeHtml = (value) => String(value ?? "").replace(/[&<>'"]/g, (char) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;"
})[char]);

function setStatus(message, kind = "ready") {
  byId("systemStatus").textContent = message;
  const dot = document.querySelector(".status-dot");
  dot.className = `status-dot${kind === "ready" ? "" : ` ${kind}`}`;
}

function showToast(message, isError = false) {
  const toast = byId("toast");
  toast.textContent = message;
  toast.className = `toast show${isError ? " error" : ""}`;
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => { toast.className = "toast"; }, 3200);
}

const asPercent = (value) => value == null ? "--" : `${(Number(value) * 100).toFixed(1)}%`;

function renderMetrics(metrics) {
  state.metrics = metrics;
  ["precision", "recall", "f1", "accuracy"].forEach((key) => { byId(key).textContent = asPercent(metrics?.[key]); });
  const matrix = metrics?.confusion_matrix || {};
  ["tp", "tn", "fp", "fn"].forEach((key) => { byId(key).textContent = matrix[key] ?? "--"; });

  const chart = byId("categoryChart");
  const entries = Object.entries(metrics?.category_distribution || {});
  if (!entries.length) {
    chart.innerHTML = '<p class="muted">暂无分类数据</p>';
    return;
  }
  const maximum = Math.max(...entries.map(([, count]) => count), 1);
  chart.innerHTML = entries.map(([label, count]) => `
    <div class="bar-row">
      <span class="bar-label">${escapeHtml(label)}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${(count / maximum) * 100}%"></div></div>
      <span class="bar-value">${count}</span>
    </div>`).join("");
}

function populateCategories() {
  const select = byId("categoryFilter");
  const current = select.value;
  const categories = [...new Set(state.predictions.map((row) => row.category).filter(Boolean))];
  select.innerHTML = '<option value="all">全部分类</option>' + categories.map((category) => `<option value="${escapeHtml(category)}">${escapeHtml(category)}</option>`).join("");
  if ([...select.options].some((option) => option.value === current)) select.value = current;
}

function filteredRows() {
  const decision = byId("decisionFilter").value;
  const category = byId("categoryFilter").value;
  const query = byId("searchInput").value.trim().toLowerCase();
  return state.predictions.filter((row) => {
    if (decision !== "all" && String(row.is_hallucination) !== decision) return false;
    if (category !== "all" && row.category !== category) return false;
    if (query && ![row.id, row.user_question, row.reason, row.claims, row.evidence].some((value) => String(value ?? "").toLowerCase().includes(query))) return false;
    return true;
  });
}

function renderTable() {
  const rows = filteredRows();
  byId("resultCount").textContent = `显示 ${rows.length} / ${state.predictions.length} 条结果`;
  if (!rows.length) {
    byId("resultRows").innerHTML = '<tr><td colspan="7" class="empty-state">当前筛选条件下无结果</td></tr>';
    return;
  }
  byId("resultRows").innerHTML = rows.map((row) => `
    <tr>
      <td><strong>${escapeHtml(row.id)}</strong></td>
      <td><span class="pill ${row.is_hallucination ? "pill--danger" : "pill--normal"}">${row.is_hallucination ? "幻觉" : "正常"}</span></td>
      <td>${escapeHtml(row.category || "--")}</td>
      <td><span class="severity">${escapeHtml(row.severity)}</span></td>
      <td>${asPercent(row.confidence)}</td>
      <td class="reason-cell">${escapeHtml(row.reason)}</td>
      <td><button class="text-button" type="button" data-detail-id="${escapeHtml(row.id)}">详情</button></td>
    </tr>`).join("");
}

function renderAll(payload) {
  state.predictions = payload.predictions || [];
  populateCategories();
  renderMetrics(payload.metrics || null);
  renderTable();
}

function openDetail(id) {
  const row = state.predictions.find((item) => item.id === id);
  if (!row) return;
  byId("dialogTitle").textContent = `${row.id} · ${row.is_hallucination ? "幻觉" : "正常"}`;
  const details = [
    ["分类 / 严重程度", `${row.category || "无"} / ${row.severity}`],
    ["置信度 / 检测模式", `${asPercent(row.confidence)} / ${row.detection_mode}`],
    ["用户问题", row.user_question],
    ["客服主张", row.claims],
    ["知识库证据", row.evidence],
    ["判断原因", row.reason],
  ];
  byId("dialogBody").innerHTML = `<dl>${details.map(([label, value]) => `<div class="detail-row"><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd></div>`).join("")}</dl>`;
  byId("detailDialog").showModal();
}

async function loadResults() {
  try {
    const response = await fetch("/api/results");
    if (!response.ok) throw new Error("无法读取结果");
    renderAll(await response.json());
    setStatus("离线引擎就绪");
  } catch (error) {
    setStatus("结果加载失败", "error");
    showToast(error.message, true);
  }
}

async function runDetection() {
  const button = byId("runButton");
  const mode = byId("modeSelect").value;
  button.disabled = true;
  button.textContent = "运行中...";
  setStatus(`${mode.toUpperCase()} 检测运行中`, "busy");
  try {
    const response = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "检测失败");
    renderAll(payload);
    byId("lastRun").textContent = `${mode.toUpperCase()} · ${new Date().toLocaleString("zh-CN", { hour12: false })}`;
    setStatus(`${mode.toUpperCase()} 运行完成`);
    showToast(`完成 ${payload.predictions.length} 条回复检测`);
  } catch (error) {
    setStatus("检测失败", "error");
    showToast(error.message, true);
  } finally {
    button.disabled = false;
    button.textContent = "运行检测";
  }
}

byId("runButton").addEventListener("click", runDetection);
["decisionFilter", "categoryFilter"].forEach((id) => byId(id).addEventListener("change", renderTable));
byId("searchInput").addEventListener("input", renderTable);
byId("resultRows").addEventListener("click", (event) => {
  const button = event.target.closest("[data-detail-id]");
  if (button) openDetail(button.dataset.detailId);
});
byId("closeDialog").addEventListener("click", () => byId("detailDialog").close());
byId("detailDialog").addEventListener("click", (event) => {
  if (event.target === byId("detailDialog")) byId("detailDialog").close();
});

loadResults();
