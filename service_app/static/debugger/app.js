const CUSTOM_BUDGETS = [4, 8, 12];
const CACHED_EXAMPLES_PATHS =
  window.location.protocol === "file:"
    ? ["./cached_examples.json", "/static/debugger/cached_examples.json"]
    : ["/static/debugger/cached_examples.json", "./cached_examples.json"];

const state = {
  catalog: [],
  payload: null,
  scenarioId: null,
  budgetOptions: [],
  dataMode: "backend",
  selectedStrategyId: null,
  selectedEventIndex: 0,
  selectedCandidateId: null,
  customPayloads: {},
  prototypeCatalog: [],
  prototypePayloads: {},
  prototypeLoaded: false,
};

const elements = {
  scenarioSelect: document.getElementById("scenarioSelect"),
  budgetRange: document.getElementById("budgetRange"),
  budgetValue: document.getElementById("budgetValue"),
  refreshButton: document.getElementById("refreshButton"),
  promptText: document.getElementById("promptText"),
  promptMeta: document.getElementById("promptMeta"),
  groundTruth: document.getElementById("groundTruth"),
  strategyGrid: document.getElementById("strategyGrid"),
  timelineHint: document.getElementById("timelineHint"),
  timeline: document.getElementById("timeline"),
  stepTitle: document.getElementById("stepTitle"),
  decisionBox: document.getElementById("decisionBox"),
  signals: document.getElementById("signals"),
  candidates: document.getElementById("candidates"),
  candidateDetail: document.getElementById("candidateDetail"),
  treeContainer: document.getElementById("treeContainer"),
  globalPromptInput: document.getElementById("globalPromptInput"),
  providerSelect: document.getElementById("providerSelect"),
  modelIdInput: document.getElementById("modelIdInput"),
  modelApiKeyInput: document.getElementById("modelApiKeyInput"),
  singleQuestionInput: document.getElementById("singleQuestionInput"),
  singleGoldInput: document.getElementById("singleGoldInput"),
  runCustomButton: document.getElementById("runCustomButton"),
  resetDemoButton: document.getElementById("resetDemoButton"),
  customStatus: document.getElementById("customStatus"),
};

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json();
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json();
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function deepClone(value) {
  return JSON.parse(JSON.stringify(value));
}

function metricToPercent(signal) {
  const numericValue = Number(signal?.value ?? 0);
  if (Number.isNaN(numericValue)) {
    return 0;
  }

  const normalized = numericValue <= 1 ? numericValue * 100 : numericValue;
  const clampedValue = Math.max(0, Math.min(100, normalized));

  if (signal?.direction === "lower_better") {
    return 100 - clampedValue;
  }
  return clampedValue;
}

function formatMetric(value) {
  const numericValue = Number(value);
  if (Number.isNaN(numericValue)) {
    return String(value);
  }

  if (numericValue >= 1000) {
    return numericValue.toLocaleString();
  }

  if (Math.abs(numericValue) < 1) {
    return numericValue.toFixed(2);
  }

  return Number.isInteger(numericValue)
    ? String(numericValue)
    : numericValue.toFixed(2);
}

function getCurrentBudget() {
  const index = Number(elements.budgetRange.value || 0);
  return state.budgetOptions[index] ?? state.budgetOptions[0];
}

function pickNearestBudget(target, availableBudgets) {
  if (!availableBudgets?.length) {
    return null;
  }
  const expected = target ?? availableBudgets[0];
  return availableBudgets.reduce((best, current) => {
    const bestGap = Math.abs(best - expected);
    const currentGap = Math.abs(current - expected);
    return currentGap < bestGap ? current : best;
  }, availableBudgets[0]);
}

function normalizePrototypeBundle(rawBundle) {
  const payloads = rawBundle?.payloads || {};
  const scenarios = Array.isArray(rawBundle?.scenarios) ? rawBundle.scenarios : [];

  const normalizedScenarios = scenarios
    .map((scenario) => {
      const scenarioId = scenario?.id;
      if (!scenarioId || !payloads[scenarioId]) {
        return null;
      }

      const availableBudgets = Array.from(
        new Set(
          (Array.isArray(scenario.available_budgets)
            ? scenario.available_budgets
            : Object.keys(payloads[scenarioId])
          )
            .map((value) => Number(value))
            .filter((value) => Number.isFinite(value) && value > 0),
        ),
      ).sort((left, right) => left - right);

      if (!availableBudgets.length) {
        return null;
      }

      const defaultBudget = pickNearestBudget(
        scenario.default_budget,
        availableBudgets,
      );

      return {
        id: String(scenarioId),
        title: String(scenario.title || scenarioId),
        description: String(scenario.description || ""),
        available_budgets: availableBudgets,
        default_budget: defaultBudget,
      };
    })
    .filter((item) => item != null);

  return {
    scenarios: normalizedScenarios,
    payloads,
  };
}

async function fetchCachedExamplesJson() {
  let lastError = null;
  for (const path of CACHED_EXAMPLES_PATHS) {
    try {
      return await fetchJson(path);
    } catch (error) {
      lastError = error;
    }
  }

  throw (
    lastError ||
    new Error("Unable to load cached_examples.json for prototype mode.")
  );
}

async function ensurePrototypeDataLoaded() {
  if (state.prototypeLoaded) {
    return;
  }

  const rawBundle = await fetchCachedExamplesJson();
  const normalized = normalizePrototypeBundle(rawBundle);
  state.prototypeCatalog = normalized.scenarios;
  state.prototypePayloads = normalized.payloads;
  state.prototypeLoaded = true;
}

function getPrototypeScenarioPayload(scenarioId, budget) {
  const scenarioPayloads = state.prototypePayloads[scenarioId];
  if (!scenarioPayloads) {
    return null;
  }

  const availableBudgets = Object.keys(scenarioPayloads)
    .map((value) => Number(value))
    .filter((value) => Number.isFinite(value))
    .sort((left, right) => left - right);

  if (!availableBudgets.length) {
    return null;
  }

  const selectedBudget = pickNearestBudget(budget, availableBudgets);
  const payload = scenarioPayloads[String(selectedBudget)];
  if (!payload) {
    return null;
  }

  const clonedPayload = deepClone(payload);
  clonedPayload.selected_budget = selectedBudget;
  if (!clonedPayload.available_budgets?.length) {
    clonedPayload.available_budgets = availableBudgets;
  }
  return clonedPayload;
}

async function loadCatalog() {
  try {
    const catalogResponse = await fetchJson("/v1/debugger/demo/scenarios");
    state.dataMode = "backend";
    return catalogResponse.scenarios || [];
  } catch (error) {
    await ensurePrototypeDataLoaded();
    if (state.prototypeCatalog.length) {
      state.dataMode = "prototype";
      return deepClone(state.prototypeCatalog);
    }
    throw error;
  }
}

async function loadPayloadForScenario(scenarioId, budget) {
  if (state.dataMode === "custom") {
    const customRuns = state.customPayloads[scenarioId] || {};
    const payload =
      customRuns[String(budget)] || customRuns[String(CUSTOM_BUDGETS[1])];
    if (payload) {
      return deepClone(payload);
    }
    throw new Error(`Custom scenario not found: ${scenarioId}`);
  }

  if (state.dataMode === "prototype") {
    await ensurePrototypeDataLoaded();
    const localPayload = getPrototypeScenarioPayload(scenarioId, budget);
    if (localPayload) {
      return localPayload;
    }
  }

  try {
    const payload = await fetchJson(
      `/v1/debugger/demo/scenarios/${scenarioId}?budget=${budget}`,
    );
    state.dataMode = "backend";
    return payload;
  } catch (error) {
    await ensurePrototypeDataLoaded();
    const localPayload = getPrototypeScenarioPayload(scenarioId, budget);
    if (localPayload) {
      state.dataMode = "prototype";
      return localPayload;
    }
    throw error;
  }
}

function configureBudgetSlider(defaultBudget) {
  const selectedScenario = state.catalog.find(
    (scenario) => scenario.id === state.scenarioId,
  );
  const options = selectedScenario?.available_budgets ?? [];

  state.budgetOptions = options;
  elements.budgetRange.min = "0";
  elements.budgetRange.max = String(Math.max(0, options.length - 1));

  if (!options.length) {
    elements.budgetRange.value = "0";
    elements.budgetValue.textContent = "-";
    return;
  }

  const target = defaultBudget ?? options[0];
  const nearestIndex = options.reduce(
    (best, optionBudget, index) => {
      const bestGap = Math.abs(options[best] - target);
      const currentGap = Math.abs(optionBudget - target);
      return currentGap < bestGap ? index : best;
    },
    0,
  );

  elements.budgetRange.value = String(nearestIndex);
  elements.budgetValue.textContent = `${options[nearestIndex]} budget units`;
}

function populateScenarioSelect() {
  elements.scenarioSelect.innerHTML = state.catalog
    .map(
      (scenario) =>
        `<option value="${escapeHtml(scenario.id)}">${escapeHtml(scenario.title)}</option>`,
    )
    .join("");

  if (state.scenarioId) {
    elements.scenarioSelect.value = state.scenarioId;
  }
}

function setStatus(message, isError = false) {
  elements.customStatus.textContent = message;
  elements.customStatus.style.color = isError ? "var(--bad)" : "var(--muted)";
}

function maskApiKey(apiKey) {
  if (!apiKey) {
    return "";
  }
  if (apiKey.length <= 8) {
    return `${apiKey.slice(0, 2)}***`;
  }
  return `${apiKey.slice(0, 4)}...${apiKey.slice(-4)}`;
}

async function buildCustomRunsViaBackend(sample, sharedPrompt, modelConfig) {
  const scenarioId = "custom_1";
  const preview = String(sample.question).slice(0, 64);
  const title = `Single Example · sample 1 · ${preview}${sample.question.length > 64 ? "..." : ""}`;
  const payloads = { [scenarioId]: {} };

  for (const budget of CUSTOM_BUDGETS) {
    const payload = await postJson("/v1/debugger/demo/run-single", {
      question: sample.question,
      gold_answer: sample.gold_answer,
      shared_prompt: sharedPrompt,
      budget,
      provider: modelConfig.provider,
      model_id: modelConfig.model_id,
      api_key: modelConfig.api_key_raw || "",
    });

    payload.scenario = payload.scenario || {};
    payload.scenario.id = scenarioId;
    payload.scenario.title = title;
    payloads[scenarioId][String(budget)] = payload;
  }

  return {
    catalog: [
      {
        id: scenarioId,
        title,
        description: "Custom question loaded by user",
        available_budgets: CUSTOM_BUDGETS,
        default_budget: CUSTOM_BUDGETS[1],
      },
    ],
    payloads,
  };
}

async function runCustomInput() {
  const sharedPrompt = elements.globalPromptInput.value.trim();
  const provider = elements.providerSelect.value.trim();
  const modelId = elements.modelIdInput.value.trim();
  const apiKey = elements.modelApiKeyInput.value.trim();
  const question = elements.singleQuestionInput.value.trim();
  const goldAnswer = elements.singleGoldInput.value.trim();

  if (!provider) {
    setStatus("Please select a provider.", true);
    return;
  }

  if (!modelId) {
    setStatus("Please input a model ID.", true);
    return;
  }

  if (!apiKey) {
    setStatus("Please input an API key.", true);
    return;
  }

  if (!question) {
    setStatus("Please input a question.", true);
    return;
  }

  if (!goldAnswer) {
    setStatus("Please input a gold answer.", true);
    return;
  }

  const modelConfig = {
    provider,
    model_id: modelId,
    api_key_raw: apiKey,
    api_key_masked: maskApiKey(apiKey),
  };

  let customRuns;
  try {
    customRuns = await buildCustomRunsViaBackend(
      { question, gold_answer: goldAnswer },
      sharedPrompt,
      modelConfig,
    );
  } catch (error) {
    setStatus(
      `Custom run requires backend endpoint /v1/debugger/demo/run-single (${error.message}).`,
      true,
    );
    return;
  }

  state.customPayloads = customRuns.payloads;
  state.catalog = customRuns.catalog;
  state.dataMode = "custom";
  state.scenarioId = state.catalog[0]?.id || null;
  state.selectedStrategyId = null;
  state.selectedEventIndex = 0;
  state.selectedCandidateId = null;

  populateScenarioSelect();
  configureBudgetSlider(CUSTOM_BUDGETS[1]);
  await loadScenarioPayload();

  setStatus(
    `Ran backend strategy-scorer matrix for ${provider}:${modelId}. API key input remains placeholder until full backend credential wiring is implemented.`,
    false,
  );
}

async function restoreDemoData() {
  try {
    state.customPayloads = {};
    state.catalog = await loadCatalog();
    state.scenarioId = state.catalog[0]?.id || null;
    state.selectedStrategyId = null;
    state.selectedEventIndex = 0;
    state.selectedCandidateId = null;

    populateScenarioSelect();

    if (!state.catalog.length) {
      elements.strategyGrid.innerHTML =
        '<p class="tree-empty">No debugger scenarios are available.</p>';
      return;
    }

    configureBudgetSlider(state.catalog[0].default_budget);
    await loadScenarioPayload();
    setStatus("Restored demo data.", false);
  } catch (error) {
    setStatus(`Failed to restore demo data: ${error.message}`, true);
  }
}

async function loadScenarioPayload() {
  const budget = getCurrentBudget();
  if (!state.scenarioId || budget == null) {
    return;
  }

  const payload = await loadPayloadForScenario(state.scenarioId, budget);
  state.payload = payload;

  const currentStrategyExists = payload.strategies.some(
    (strategy) => strategy.id === state.selectedStrategyId,
  );

  if (!currentStrategyExists) {
    const best = [...payload.strategies].sort(
      (left, right) => left.comparison_rank - right.comparison_rank,
    )[0];
    state.selectedStrategyId = best?.id ?? null;
    state.selectedEventIndex = 0;
    state.selectedCandidateId = null;
  }

  render();
}

function selectFirstCandidate(eventItem) {
  if (!eventItem?.candidates?.length) {
    state.selectedCandidateId = null;
    return;
  }

  const selectedCandidate = eventItem.candidates.find(
    (candidate) => candidate.selected,
  );
  state.selectedCandidateId =
    selectedCandidate?.id ?? eventItem.candidates[0].id ?? null;
}

function getSelectedStrategy() {
  return state.payload?.strategies?.find(
    (strategy) => strategy.id === state.selectedStrategyId,
  );
}

function renderPrompt() {
  const scenario = state.payload?.scenario;
  elements.promptText.textContent = scenario?.prompt ?? "";
  elements.groundTruth.textContent = scenario?.ground_truth ?? "-";

  const modelConfig = scenario?.model_config || {};
  const metadataParts = [];
  if (modelConfig.provider && modelConfig.model_id) {
    metadataParts.push(`model=${modelConfig.provider}:${modelConfig.model_id}`);
  }
  if (modelConfig.api_key_masked) {
    metadataParts.push(`api_key=${modelConfig.api_key_masked}`);
  }
  if (scenario?.shared_prompt) {
    metadataParts.push(`shared_prompt=${scenario.shared_prompt}`);
  }
  if (scenario?.input_source) {
    metadataParts.push(`source=${scenario.input_source}`);
  }
  if (scenario?.strategy_count && scenario?.scorer_count) {
    metadataParts.push(
      `matrix=${scenario.strategy_count}x${scenario.scorer_count} (${scenario.run_count || scenario.strategy_count * scenario.scorer_count} runs)`,
    );
  }
  elements.promptMeta.textContent = metadataParts.join(" | ");
}

function renderStrategyCards() {
  const strategies = state.payload?.strategies ?? [];
  elements.strategyGrid.innerHTML = "";

  strategies.forEach((strategy, index) => {
    const run = strategy.run || {};
    const finalResult = run.final || {};
    const strategyLabel =
      run.strategy?.name || strategy.name || strategy.strategy_id || "Strategy";
    const scorerLabel = run.scorer?.name || strategy.scorer_id || "scorer";
    const isActive = strategy.id === state.selectedStrategyId;
    const card = document.createElement("article");
    card.className = `strategy-card${isActive ? " active" : ""}`;
    card.style.animationDelay = `${index * 50}ms`;

    const outcomeClass = finalResult.is_correct ? "outcome-ok" : "outcome-bad";
    const outcomeText = finalResult.is_correct ? "correct" : "incorrect";

    card.innerHTML = `
      <div class="strategy-title">
        <h3>${escapeHtml(strategyLabel)}</h3>
        <span class="rank-pill">rank #${strategy.comparison_rank}</span>
      </div>
      <p class="timeline-step">scorer · ${escapeHtml(scorerLabel)}</p>
      <p class="timeline-decision">${escapeHtml(strategy.summary || "")}</p>
      <div class="strategy-meta">
        <div><span class="timeline-step">answer</span><br /><span class="meta-value ${outcomeClass}">${escapeHtml(finalResult.answer ?? "-")} (${outcomeText})</span></div>
        <div><span class="timeline-step">confidence</span><br /><span class="meta-value">${formatMetric(finalResult.confidence ?? 0)}</span></div>
        <div><span class="timeline-step">quality</span><br /><span class="meta-value">${formatMetric(finalResult.quality_score ?? 0)}</span></div>
        <div><span class="timeline-step">tokens</span><br /><span class="meta-value">${formatMetric(run.tokens_used ?? 0)}</span></div>
      </div>
    `;

    card.addEventListener("click", () => {
      state.selectedStrategyId = strategy.id;
      state.selectedEventIndex = 0;
      selectFirstCandidate(strategy.run?.events?.[0]);
      render();
    });

    elements.strategyGrid.appendChild(card);
  });
}

function renderTimeline() {
  const selectedStrategy = getSelectedStrategy();
  const strategyName =
    selectedStrategy?.run?.strategy?.name || selectedStrategy?.name || "";
  const scorerName =
    selectedStrategy?.run?.scorer?.name || selectedStrategy?.scorer_id || "";
  const scorerLabel = scorerName ? ` | scorer=${scorerName}` : "";
  const events = selectedStrategy?.run?.events ?? [];
  const modeLabel =
    state.dataMode === "prototype"
      ? " | prototype mode (cached json)"
      : state.dataMode === "custom"
        ? " | custom mode (backend run)"
        : "";

  elements.timelineHint.textContent = selectedStrategy
    ? `${strategyName}${scorerLabel} | ${selectedStrategy.family}${modeLabel}`
    : "Select a strategy to inspect each decision point.";

  elements.timeline.innerHTML = "";

  if (!events.length) {
    elements.timeline.innerHTML =
      '<p class="tree-empty">No timeline events available.</p>';
    return;
  }

  if (state.selectedEventIndex >= events.length) {
    state.selectedEventIndex = 0;
  }

  events.forEach((eventItem, index) => {
    const active = index === state.selectedEventIndex;
    const node = document.createElement("article");
    node.className = `timeline-item${active ? " active" : ""}`;
    node.innerHTML = `
      <p class="timeline-step">step ${eventItem.step} · ${escapeHtml(eventItem.stage || "")}</p>
      <p class="timeline-title">${escapeHtml(eventItem.title || "")}</p>
      <p class="timeline-decision"><strong>${escapeHtml(eventItem.decision?.action || "")}</strong> · ${escapeHtml(eventItem.decision?.reason || "")}</p>
    `;

    node.addEventListener("click", () => {
      state.selectedEventIndex = index;
      selectFirstCandidate(eventItem);
      renderStepInspector();
      renderTimeline();
    });

    elements.timeline.appendChild(node);
  });

  const activeEvent = events[state.selectedEventIndex];
  if (!state.selectedCandidateId) {
    selectFirstCandidate(activeEvent);
  }
}

function renderSignals(eventItem) {
  const signals = eventItem?.signals ?? [];

  if (!signals.length) {
    elements.signals.innerHTML =
      '<p class="tree-empty">No signal telemetry for this step.</p>';
    return;
  }

  elements.signals.innerHTML = signals
    .map((signal) => {
      const percent = metricToPercent(signal);
      const threshold =
        signal.threshold != null
          ? `threshold ${formatMetric(signal.threshold)}`
          : "";
      return `
        <div class="signal-row">
          <header>
            <span>${escapeHtml(signal.name)}</span>
            <span>${formatMetric(signal.value)} ${threshold}</span>
          </header>
          <div class="signal-meter"><span style="width:${percent}%"></span></div>
        </div>
      `;
    })
    .join("");
}

function renderCandidates(eventItem) {
  const candidates = eventItem?.candidates ?? [];

  if (!candidates.length) {
    elements.candidates.innerHTML =
      '<p class="tree-empty">No candidates attached to this event.</p>';
    elements.candidateDetail.innerHTML =
      '<p class="tree-empty">Candidate detail appears when a candidate is selected.</p>';
    return;
  }

  elements.candidates.innerHTML = "";

  candidates.forEach((candidate) => {
    const selected = candidate.id === state.selectedCandidateId;
    const card = document.createElement("article");
    card.className = `candidate-card${selected ? " selected" : ""}`;
    const badgeClass = `badge-${candidate.status || "kept"}`;

    card.innerHTML = `
      <div class="candidate-header">
        <strong>${escapeHtml(candidate.label || candidate.id)}</strong>
        <span class="badge ${badgeClass}">${escapeHtml(candidate.status || "kept")}</span>
      </div>
      <p class="candidate-snippet">${escapeHtml(candidate.text || "")}</p>
    `;

    card.addEventListener("click", () => {
      state.selectedCandidateId = candidate.id;
      renderCandidates(eventItem);
      renderCandidateDetail(eventItem);
    });

    elements.candidates.appendChild(card);
  });

  renderCandidateDetail(eventItem);
}

function renderCandidateDetail(eventItem) {
  const candidates = eventItem?.candidates ?? [];
  const candidate =
    candidates.find((item) => item.id === state.selectedCandidateId) ||
    candidates.find((item) => item.selected) ||
    candidates[0];

  if (!candidate) {
    elements.candidateDetail.innerHTML =
      '<p class="tree-empty">Candidate detail appears when a candidate is selected.</p>';
    return;
  }

  const metrics = Object.entries(candidate.signals || {})
    .map(
      ([key, value]) =>
        `<span>${escapeHtml(key)}: <strong>${formatMetric(value)}</strong></span>`,
    )
    .join("");

  elements.candidateDetail.innerHTML = `
    <pre>${escapeHtml(candidate.text || "")}</pre>
    <div class="candidate-detail-metrics">${metrics || "<span>No candidate metrics.</span>"}</div>
  `;
}

function renderTree() {
  const selectedStrategy = getSelectedStrategy();
  const tree = selectedStrategy?.run?.tree;

  if (!tree?.nodes?.length) {
    elements.treeContainer.innerHTML =
      '<p class="tree-empty">No tree structure for this strategy at this budget.</p>';
    return;
  }

  const width = 620;
  const height = 230;
  const nodeMap = new Map(tree.nodes.map((node) => [node.id, node]));
  const selectedPath = tree.selected_path || [];

  const selectedEdgeSet = new Set();
  for (let index = 0; index < selectedPath.length - 1; index += 1) {
    selectedEdgeSet.add(`${selectedPath[index]}->${selectedPath[index + 1]}`);
  }

  const edges = (tree.edges || [])
    .map((edge) => {
      const source = nodeMap.get(edge.source);
      const target = nodeMap.get(edge.target);
      if (!source || !target) {
        return "";
      }

      const x1 = source.x * width;
      const y1 = source.y * height;
      const x2 = target.x * width;
      const y2 = target.y * height;
      const active = selectedEdgeSet.has(`${edge.source}->${edge.target}`);
      const edgeClass = active ? "tree-edge selected" : "tree-edge";

      return `<line class="${edgeClass}" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"></line>`;
    })
    .join("");

  const nodes = (tree.nodes || [])
    .map((node) => {
      const x = node.x * width;
      const y = node.y * height;
      const isSelected = selectedPath.includes(node.id);
      const radius = 8 + Number(node.value || 0) * 5;
      const nodeClass = isSelected ? "tree-node selected" : "tree-node";
      const label = `${node.label} (${formatMetric(node.value)})`;

      return `
        <g>
          <circle class="${nodeClass}" cx="${x}" cy="${y}" r="${radius}"></circle>
          <text class="tree-label" x="${x + 10}" y="${y - 8}">${escapeHtml(label)}</text>
        </g>
      `;
    })
    .join("");

  elements.treeContainer.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" class="tree-svg" preserveAspectRatio="xMidYMid meet">
      ${edges}
      ${nodes}
    </svg>
  `;
}

function renderStepInspector() {
  const selectedStrategy = getSelectedStrategy();
  const eventItem = selectedStrategy?.run?.events?.[state.selectedEventIndex];

  if (!eventItem) {
    elements.stepTitle.textContent = "Pick a timeline event.";
    elements.decisionBox.innerHTML =
      '<p class="tree-empty">Decision details appear for the selected event.</p>';
    elements.signals.innerHTML = "";
    elements.candidates.innerHTML = "";
    elements.candidateDetail.innerHTML =
      '<p class="tree-empty">Candidate detail appears when a candidate is selected.</p>';
    renderTree();
    return;
  }

  elements.stepTitle.textContent = eventItem.title || "Step";
  elements.decisionBox.innerHTML = `
    <p><strong>${escapeHtml(eventItem.decision?.action || "decision")}</strong></p>
    <p>${escapeHtml(eventItem.decision?.reason || "No decision rationale")}</p>
  `;

  renderSignals(eventItem);
  renderCandidates(eventItem);
  renderTree();
}

function render() {
  if (!state.payload) {
    return;
  }

  renderPrompt();
  renderStrategyCards();
  renderTimeline();
  renderStepInspector();
}

function bindHandlers() {
  elements.scenarioSelect.addEventListener("change", async (event) => {
    state.scenarioId = event.target.value;
    const selectedScenario = state.catalog.find(
      (scenario) => scenario.id === state.scenarioId,
    );
    configureBudgetSlider(selectedScenario?.default_budget);
    state.selectedStrategyId = null;
    state.selectedEventIndex = 0;
    state.selectedCandidateId = null;
    await loadScenarioPayload();
  });

  elements.budgetRange.addEventListener("input", () => {
    const budget = getCurrentBudget();
    elements.budgetValue.textContent = `${budget ?? "-"} budget units`;
  });

  elements.budgetRange.addEventListener("change", async () => {
    await loadScenarioPayload();
  });

  elements.refreshButton.addEventListener("click", () => {
    state.selectedEventIndex = 0;
    const selectedStrategy = getSelectedStrategy();
    selectFirstCandidate(selectedStrategy?.run?.events?.[0]);
    render();
  });

  elements.runCustomButton.addEventListener("click", async () => {
    await runCustomInput();
  });

  elements.resetDemoButton.addEventListener("click", async () => {
    await restoreDemoData();
  });
}

async function init() {
  bindHandlers();

  state.catalog = await loadCatalog();

  if (!state.catalog.length) {
    elements.strategyGrid.innerHTML =
      '<p class="tree-empty">No debugger scenarios are available.</p>';
    return;
  }

  state.scenarioId = state.catalog[0].id;
  populateScenarioSelect();
  configureBudgetSlider(state.catalog[0].default_budget);
  await loadScenarioPayload();
}

init().catch((error) => {
  elements.strategyGrid.innerHTML = `<p class="tree-empty">Failed to load debugger data: ${escapeHtml(error.message)}</p>`;
});
