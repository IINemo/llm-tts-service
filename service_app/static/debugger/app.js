const CACHED_EXAMPLES_PATHS =
  window.location.protocol === "file:"
    ? ["./cached_examples.json", "/static/debugger/cached_examples.json"]
    : ["/static/debugger/cached_examples.json", "./cached_examples.json"];
const DEFAULT_SYSTEM_PROMPT = "Reason step-by-step carefully";

const state = {
  catalog: [],
  payload: null,
  cachedSourcePayload: null,
  scenarioId: null,
  budgetOptions: [],
  dataMode: "backend",
  selectedStrategyId: null,
  selectedEventIndex: 0,
  selectedCandidateId: null,
  selectedTreeNodeId: null,
  customPayloads: {},
  prototypeCatalog: [],
  prototypePayloads: {},
  prototypeLoaded: false,
  modelValidation: null,
  validatedModelFingerprint: null,
  useCachedExample: false,
  cachedScenarioPrompt: "",
  prototypeAdvancedTemplates: {},
  advancedConfigExpanded: false,
  advancedConfigTemplateKey: null,
  advancedConfigDirty: false,
};

const elements = {
  cachedExplorerControls: document.getElementById("cachedExplorerControls"),
  cachedExplorerPrompt: document.getElementById("cachedExplorerPrompt"),
  scenarioSelect: document.getElementById("scenarioSelect"),
  caseSelect: document.getElementById("caseSelect"),
  useCachedToggle: document.getElementById("useCachedToggle"),
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
  providerSelect: document.getElementById("providerSelect"),
  modelIdInput: document.getElementById("modelIdInput"),
  modelApiKeyInput: document.getElementById("modelApiKeyInput"),
  validateModelButton: document.getElementById("validateModelButton"),
  modelCapabilityStatus: document.getElementById("modelCapabilityStatus"),
  strategySelect: document.getElementById("strategySelect"),
  scorerSelect: document.getElementById("scorerSelect"),
  advancedConfigToggle: document.getElementById("advancedConfigToggle"),
  advancedConfigPanel: document.getElementById("advancedConfigPanel"),
  advancedPromptInput: document.getElementById("advancedPromptInput"),
  advancedConfigHighlight: document.getElementById("advancedConfigHighlight"),
  advancedConfigYamlInput: document.getElementById("advancedConfigYamlInput"),
  resetAdvancedConfigButton: document.getElementById("resetAdvancedConfigButton"),
  advancedConfigStatus: document.getElementById("advancedConfigStatus"),
  singleQuestionInput: document.getElementById("singleQuestionInput"),
  runCustomButton: document.getElementById("runCustomButton"),
  resetDemoButton: document.getElementById("resetDemoButton"),
  customStatus: document.getElementById("customStatus"),
};

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Request failed: ${response.status} - ${errorText}`);
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
    const errorText = await response.text();
    throw new Error(`Request failed: ${response.status} - ${errorText}`);
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
  const selected = Number(elements.caseSelect.value || "");
  if (Number.isFinite(selected)) {
    return selected;
  }
  return state.budgetOptions[0] ?? null;
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
  const advancedConfigTemplates =
    rawBundle?.advanced_config_templates &&
    typeof rawBundle.advanced_config_templates === "object"
      ? rawBundle.advanced_config_templates
      : {};

  if (Array.isArray(rawBundle?.examples)) {
    const payloads = {};
    const scenarios = rawBundle.examples
      .map((example) => {
        if (!example || typeof example !== "object") {
          return null;
        }

        const scenarioId = String(example.id || "").trim();
        if (!scenarioId || typeof example.payloads !== "object") {
          return null;
        }

        payloads[scenarioId] = example.payloads;
        const availableBudgets = Array.from(
          new Set(
            (Array.isArray(example.available_budgets)
              ? example.available_budgets
              : Object.keys(example.payloads)
            )
              .map((value) => Number(value))
              .filter((value) => Number.isFinite(value) && value > 0),
          ),
        ).sort((left, right) => left - right);

        if (!availableBudgets.length) {
          return null;
        }

        return {
          id: scenarioId,
          title: String(example.title || scenarioId),
          description: String(example.description || ""),
          available_budgets: availableBudgets,
          default_budget: pickNearestBudget(example.default_budget, availableBudgets),
        };
      })
      .filter((item) => item != null);

    return {
      scenarios,
      payloads,
      advancedConfigTemplates,
    };
  }

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

      return {
        id: String(scenarioId),
        title: String(scenario.title || scenarioId),
        description: String(scenario.description || ""),
        available_budgets: availableBudgets,
        default_budget: pickNearestBudget(scenario.default_budget, availableBudgets),
      };
    })
    .filter((item) => item != null);

  return {
    scenarios: normalizedScenarios,
    payloads,
    advancedConfigTemplates,
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
  state.prototypeAdvancedTemplates = normalized.advancedConfigTemplates || {};
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
    const nearestBudget = pickNearestBudget(
      budget,
      Object.keys(customRuns)
        .map((value) => Number(value))
        .filter((value) => Number.isFinite(value) && value > 0),
    );
    const payload = customRuns[String(nearestBudget)];
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

function configureCaseSelect(defaultBudget) {
  const selectedScenario = state.catalog.find(
    (scenario) => scenario.id === state.scenarioId,
  );
  const options = selectedScenario?.available_budgets ?? [];

  state.budgetOptions = options;

  if (!options.length) {
    elements.caseSelect.innerHTML = '<option value="">No cases</option>';
    elements.caseSelect.value = "";
    elements.caseSelect.disabled = true;
    return;
  }

  const target = defaultBudget ?? options[0];
  const nearestBudget = options.reduce(
    (best, optionBudget) => {
      const bestGap = Math.abs(best - target);
      const currentGap = Math.abs(optionBudget - target);
      return currentGap < bestGap ? optionBudget : best;
    },
    options[0],
  );

  elements.caseSelect.innerHTML = options
    .map(
      (value, index) =>
        `<option value="${value}">Case ${index + 1}</option>`,
    )
    .join("");
  elements.caseSelect.disabled = false;
  elements.caseSelect.value = String(nearestBudget);
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

function setCapabilityStatus(message, isError = false) {
  elements.modelCapabilityStatus.textContent = message;
  elements.modelCapabilityStatus.style.color = isError
    ? "var(--bad)"
    : "var(--muted)";
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

function setAdvancedConfigStatus(message, isError = false) {
  if (!elements.advancedConfigStatus) {
    return;
  }
  elements.advancedConfigStatus.textContent = message;
  elements.advancedConfigStatus.style.color = isError
    ? "var(--bad)"
    : "var(--muted)";
}

function splitYamlInlineComment(text) {
  let inSingle = false;
  let inDouble = false;

  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    if (char === "'" && !inDouble) {
      inSingle = !inSingle;
      continue;
    }
    if (char === '"' && !inSingle && text[index - 1] !== "\\") {
      inDouble = !inDouble;
      continue;
    }
    if (char === "#" && !inSingle && !inDouble) {
      const prev = text[index - 1];
      if (index === 0 || /\s/.test(prev || "")) {
        return [text.slice(0, index), text.slice(index)];
      }
    }
  }

  return [text, ""];
}

function highlightYamlValueToken(token) {
  if (!token) {
    return "";
  }
  const leading = token.match(/^\s*/)?.[0] || "";
  const trailing = token.match(/\s*$/)?.[0] || "";
  const core = token.slice(leading.length, token.length - trailing.length);
  const normalized = core.trim();
  if (!normalized) {
    return escapeHtml(token);
  }

  let className = "";
  if (/^(true|false)$/i.test(normalized)) {
    className = "yaml-bool";
  } else if (/^(null|~)$/i.test(normalized)) {
    className = "yaml-null";
  } else if (/^-?\d+(\.\d+)?$/.test(normalized)) {
    className = "yaml-number";
  } else if (
    (normalized.startsWith('"') && normalized.endsWith('"')) ||
    (normalized.startsWith("'") && normalized.endsWith("'"))
  ) {
    className = "yaml-string";
  }

  if (!className) {
    return escapeHtml(token);
  }

  return `${escapeHtml(leading)}<span class="${className}">${escapeHtml(core)}</span>${escapeHtml(trailing)}`;
}

function highlightYamlValue(text) {
  const [valuePart, commentPart] = splitYamlInlineComment(text);
  const valueHtml = highlightYamlValueToken(valuePart);
  if (!commentPart) {
    return valueHtml;
  }
  return `${valueHtml}<span class="yaml-comment">${escapeHtml(commentPart)}</span>`;
}

function highlightYamlLine(line) {
  if (!line) {
    return "";
  }

  const fullCommentMatch = line.match(/^(\s*)(#.*)$/);
  if (fullCommentMatch) {
    return `${escapeHtml(fullCommentMatch[1])}<span class="yaml-comment">${escapeHtml(fullCommentMatch[2])}</span>`;
  }

  const keyMatch = line.match(/^(\s*)([^:#][^:\n]*?)(\s*:\s*)(.*)$/);
  if (keyMatch) {
    return `${escapeHtml(keyMatch[1])}<span class="yaml-key">${escapeHtml(keyMatch[2])}</span><span class="yaml-punc">${escapeHtml(keyMatch[3])}</span>${highlightYamlValue(keyMatch[4])}`;
  }

  const listMatch = line.match(/^(\s*)(-\s+)(.*)$/);
  if (listMatch) {
    return `${escapeHtml(listMatch[1])}<span class="yaml-punc">${escapeHtml(listMatch[2])}</span>${highlightYamlValue(listMatch[3])}`;
  }

  return escapeHtml(line);
}

function renderAdvancedConfigHighlight() {
  if (!elements.advancedConfigHighlight || !elements.advancedConfigYamlInput) {
    return;
  }
  const lines = elements.advancedConfigYamlInput.value.split("\n");
  const highlighted = lines.map((line) => highlightYamlLine(line)).join("\n");
  elements.advancedConfigHighlight.innerHTML = highlighted || " ";
  elements.advancedConfigHighlight.scrollTop =
    elements.advancedConfigYamlInput.scrollTop;
  elements.advancedConfigHighlight.scrollLeft =
    elements.advancedConfigYamlInput.scrollLeft;
}

function setAdvancedConfigYamlValue(value) {
  if (!elements.advancedConfigYamlInput) {
    return;
  }
  elements.advancedConfigYamlInput.value = value || "";
  renderAdvancedConfigHighlight();
}

function upsertPromptInAdvancedYaml(yamlText, prompt) {
  const normalizedPrompt = String(prompt || "").trim();
  const promptLine = `prompt: ${JSON.stringify(normalizedPrompt)}`;
  const hasPromptLine = /^prompt\s*:/m.test(yamlText);
  if (hasPromptLine) {
    return yamlText.replace(/^prompt\s*:.*$/m, promptLine);
  }
  if (!yamlText.trim()) {
    return `${promptLine}\n`;
  }
  return `${promptLine}\n${yamlText}`;
}

function getPreferredSystemPrompt(templatePayload) {
  const cachedPrompt = String(state.cachedScenarioPrompt || "").trim();
  if (cachedPrompt) {
    return cachedPrompt;
  }

  const inputPrompt = String(elements.advancedPromptInput?.value || "").trim();
  if (inputPrompt) {
    return inputPrompt;
  }

  const templatePrompt = String(templatePayload?.config?.prompt || "").trim();
  if (templatePrompt) {
    return templatePrompt;
  }
  return DEFAULT_SYSTEM_PROMPT;
}

function setAdvancedConfigPanelExpanded(expanded) {
  state.advancedConfigExpanded = Boolean(expanded);
  elements.advancedConfigPanel?.classList.toggle("hidden", !state.advancedConfigExpanded);
  if (state.advancedConfigExpanded) {
    renderAdvancedConfigHighlight();
  }
  if (elements.advancedConfigToggle) {
    elements.advancedConfigToggle.textContent = state.advancedConfigExpanded
      ? "Hide Advanced config"
      : "Show Advanced config";
    elements.advancedConfigToggle.setAttribute(
      "aria-expanded",
      state.advancedConfigExpanded ? "true" : "false",
    );
  }
}

function setAdvancedConfigEditorEnabled(enabled) {
  if (elements.advancedPromptInput) {
    elements.advancedPromptInput.disabled = !enabled;
  }
  if (elements.advancedConfigYamlInput) {
    elements.advancedConfigYamlInput.disabled = !enabled;
  }
  if (elements.resetAdvancedConfigButton) {
    elements.resetAdvancedConfigButton.disabled = !enabled;
  }
}

function getSelectedScorerIdForStrategy(strategy) {
  if (!strategy || strategy.requires_scorer === false) {
    return null;
  }
  const scorerId = elements.scorerSelect.value.trim();
  return scorerId || null;
}

function yamlScalar(value) {
  if (value == null) {
    return "null";
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  const text = String(value);
  const needsQuote =
    text === "" ||
    /^\s|\s$/.test(text) ||
    /[:{}\[\],&*#?|\-<>=!%@`]/.test(text) ||
    /\n/.test(text);
  return needsQuote ? JSON.stringify(text) : text;
}

function objectToYaml(value, indent = 0) {
  const prefix = " ".repeat(indent);
  if (Array.isArray(value)) {
    if (!value.length) {
      return `${prefix}[]`;
    }
    return value
      .map((item) => {
        if (item && typeof item === "object") {
          return `${prefix}-\n${objectToYaml(item, indent + 2)}`;
        }
        return `${prefix}- ${yamlScalar(item)}`;
      })
      .join("\n");
  }

  if (value && typeof value === "object") {
    const entries = Object.entries(value);
    if (!entries.length) {
      return `${prefix}{}`;
    }
    return entries
      .map(([key, item]) => {
        if (item && typeof item === "object") {
          return `${prefix}${key}:\n${objectToYaml(item, indent + 2)}`;
        }
        return `${prefix}${key}: ${yamlScalar(item)}`;
      })
      .join("\n");
  }

  return `${prefix}${yamlScalar(value)}`;
}

function buildAdvancedTemplateFromPrototype(strategyId, scorerId) {
  const templateSource = state.prototypeAdvancedTemplates || {};
  const prompt = String(templateSource.prompt || "");
  const generation = deepClone(templateSource.generation || {});
  const strategyTemplates =
    templateSource.strategies && typeof templateSource.strategies === "object"
      ? templateSource.strategies
      : {};
  const scorerTemplates =
    templateSource.scorers && typeof templateSource.scorers === "object"
      ? templateSource.scorers
      : {};

  const strategyConfig = deepClone(strategyTemplates[strategyId] || { type: strategyId });
  const config = {
    prompt,
    generation,
    strategy: strategyConfig,
  };
  if (scorerId) {
    config.scorer = deepClone(scorerTemplates[scorerId] || { type: scorerId });
  }

  return {
    config,
    config_yaml: `${objectToYaml(config)}\n`,
  };
}

async function fetchAdvancedConfigTemplate(strategyId, scorerId) {
  const params = new URLSearchParams();
  params.set("strategy_id", strategyId);
  if (scorerId) {
    params.set("scorer_id", scorerId);
  }
  return fetchJson(`/v1/debugger/demo/advanced-config/template?${params.toString()}`);
}

async function refreshAdvancedConfigTemplate(force = false) {
  const strategy = getSelectedValidatedStrategy();
  if (!strategy) {
    state.advancedConfigTemplateKey = null;
    state.advancedConfigDirty = false;
    if (elements.advancedPromptInput) {
      elements.advancedPromptInput.value = DEFAULT_SYSTEM_PROMPT;
    }
    setAdvancedConfigYamlValue("");
    setAdvancedConfigEditorEnabled(false);
    setAdvancedConfigStatus("Select strategy first to load advanced config YAML.", false);
    return;
  }

  const scorerId = getSelectedScorerIdForStrategy(strategy);
  const templateKey = `${strategy.id}::${scorerId || "none"}`;
  if (!force && state.advancedConfigTemplateKey === templateKey) {
    return;
  }

  let templatePayload = null;
  try {
    templatePayload = await fetchAdvancedConfigTemplate(strategy.id, scorerId);
  } catch (error) {
    if (state.dataMode === "prototype" || window.location.protocol === "file:") {
      await ensurePrototypeDataLoaded();
      templatePayload = buildAdvancedTemplateFromPrototype(strategy.id, scorerId);
      setAdvancedConfigStatus(
        "Loaded advanced config template from cached examples.",
        false,
      );
    } else {
      setAdvancedConfigEditorEnabled(false);
      setAdvancedConfigStatus(
        `Failed to load advanced config template: ${error.message}`,
        true,
      );
      return;
    }
  }

  const promptValue = getPreferredSystemPrompt(templatePayload);
  if (elements.advancedPromptInput) {
    elements.advancedPromptInput.value = promptValue;
  }
  const withPrompt = upsertPromptInAdvancedYaml(
    templatePayload?.config_yaml || "",
    promptValue,
  );
  setAdvancedConfigYamlValue(withPrompt);
  state.advancedConfigTemplateKey = templateKey;
  state.advancedConfigDirty = false;
  setAdvancedConfigEditorEnabled(!state.useCachedExample);
  if (!templatePayload?.config_yaml) {
    setAdvancedConfigStatus("Template loaded with empty YAML content.", true);
    return;
  }
  if (state.dataMode !== "prototype" && window.location.protocol !== "file:") {
    setAdvancedConfigStatus("Advanced config template loaded from backend defaults.", false);
  }
}

function normalizeStrategyOption(strategy) {
  const strategyId = String(strategy?.id || "");
  return {
    ...strategy,
    id: strategyId,
    requires_scorer:
      strategy?.requires_scorer != null
        ? Boolean(strategy.requires_scorer)
        : strategyId !== "baseline",
  };
}

function deriveOptionsFromPayload(payload) {
  const catalogStrategies = Array.isArray(payload?.strategy_catalog)
    ? payload.strategy_catalog
    : [];
  const strategiesFromRuns = Array.isArray(payload?.strategies)
    ? payload.strategies.map((item) => ({
        id: item?.strategy_id || item?.run?.strategy?.id,
        name: item?.run?.strategy?.name || item?.strategy_id,
        family: item?.family || item?.run?.strategy?.family,
      }))
    : [];

  const strategyMap = new Map();
  [...catalogStrategies, ...strategiesFromRuns].forEach((strategy) => {
    const normalized = normalizeStrategyOption(strategy);
    if (!normalized.id) {
      return;
    }
    if (!strategyMap.has(normalized.id)) {
      strategyMap.set(normalized.id, normalized);
    }
  });

  const scorersFromCatalog = Array.isArray(payload?.scorer_catalog)
    ? payload.scorer_catalog
    : [];
  const scorersFromRuns = Array.isArray(payload?.strategies)
    ? payload.strategies
        .map((item) => item?.run?.scorer)
        .filter((item) => item && item.id)
    : [];

  const scorerMap = new Map();
  [...scorersFromCatalog, ...scorersFromRuns].forEach((scorer) => {
    if (!scorer?.id) {
      return;
    }
    if (!scorerMap.has(scorer.id)) {
      scorerMap.set(scorer.id, scorer);
    }
  });

  return {
    strategies: Array.from(strategyMap.values()),
    scorers: Array.from(scorerMap.values()),
  };
}

function clearRenderedResults() {
  state.payload = null;
  state.selectedStrategyId = null;
  state.selectedEventIndex = 0;
  state.selectedCandidateId = null;
  state.selectedTreeNodeId = null;
  elements.promptText.textContent = "";
  elements.promptMeta.textContent = "";
  elements.groundTruth.textContent = "-";
  elements.strategyGrid.innerHTML =
    '<p class="tree-empty">No result yet. Configure inputs and click Run.</p>';
  elements.timelineHint.textContent = "Select a step to inspect content and scores.";
  elements.timeline.innerHTML =
    '<p class="tree-empty">No timeline events available.</p>';
  elements.stepTitle.textContent = "Pick a timeline step.";
  elements.decisionBox.innerHTML =
    '<p class="tree-empty">Decision details appear for the selected step.</p>';
  elements.signals.innerHTML =
    '<p class="tree-empty">No signal telemetry for this step.</p>';
  elements.candidates.innerHTML =
    '<p class="tree-empty">No candidates attached to this event.</p>';
  elements.candidateDetail.innerHTML =
    '<p class="tree-empty">Candidate detail appears when a candidate is selected.</p>';
  elements.treeContainer.innerHTML =
    '<p class="tree-empty">No tree structure for this strategy.</p>';
}

function applyCachedModeUi() {
  const disabled = state.useCachedExample;
  const controls = [
    elements.providerSelect,
    elements.modelIdInput,
    elements.modelApiKeyInput,
    elements.validateModelButton,
    elements.singleQuestionInput,
    elements.advancedPromptInput,
    elements.advancedConfigToggle,
  ];
  controls.forEach((control) => {
    control.disabled = disabled;
  });
  elements.cachedExplorerControls?.classList.toggle("hidden", !state.useCachedExample);
  elements.cachedExplorerPrompt?.classList.add("hidden");
  elements.useCachedToggle.checked = state.useCachedExample;
  setAdvancedConfigEditorEnabled(
    !disabled && Boolean(getSelectedValidatedStrategy()),
  );
}

function extractQuestionFromScenario(scenario) {
  const directQuestion = [
    scenario?.question,
    scenario?.input_question,
    scenario?.user_question,
  ].find((value) => typeof value === "string" && value.trim());
  if (directQuestion) {
    return directQuestion.trim();
  }

  const prompt = typeof scenario?.prompt === "string" ? scenario.prompt : "";
  if (!prompt.trim()) {
    return "";
  }

  const questionMatch = prompt.match(/(?:^|\n)\s*Question:\s*([\s\S]*)$/i);
  if (questionMatch?.[1]) {
    return questionMatch[1].trim();
  }

  const sharedPrompt =
    typeof scenario?.shared_prompt === "string" ? scenario.shared_prompt : "";
  if (sharedPrompt && prompt.startsWith(sharedPrompt)) {
    return prompt.slice(sharedPrompt.length).trim();
  }

  return prompt.trim();
}

function loadCachedScenarioValuesIntoInputs(payload) {
  const scenario = payload?.scenario || {};
  const modelConfig = scenario?.model_config || {};

  if (modelConfig.provider) {
    const hasProviderOption = Array.from(elements.providerSelect.options).some(
      (option) => option.value === modelConfig.provider,
    );
    if (hasProviderOption) {
      elements.providerSelect.value = modelConfig.provider;
    }
  }

  if (modelConfig.model_id) {
    elements.modelIdInput.value = modelConfig.model_id;
  }

  state.cachedScenarioPrompt = scenario?.shared_prompt || "";
  if (elements.advancedPromptInput) {
    elements.advancedPromptInput.value = state.cachedScenarioPrompt;
  }
  elements.singleQuestionInput.value = extractQuestionFromScenario(scenario);
}

async function loadCachedOptionsForCurrentScenario() {
  if (!state.scenarioId) {
    return;
  }
  const budget = getCurrentBudget();
  if (budget == null) {
    return;
  }

  const payload = await loadPayloadForScenario(state.scenarioId, budget);
  state.cachedSourcePayload = payload;
  loadCachedScenarioValuesIntoInputs(payload);
  const options = deriveOptionsFromPayload(payload);
  state.modelValidation = {
    ...(state.modelValidation || {}),
    strategies: options.strategies,
    scorers: options.scorers,
    supports_logprobs: true,
    supports_prefill: true,
  };
  state.validatedModelFingerprint = null;
  renderValidationOptions(options.strategies, options.scorers);
}

function getModelFingerprint() {
  return [
    elements.providerSelect.value.trim(),
    elements.modelIdInput.value.trim(),
    elements.modelApiKeyInput.value.trim(),
  ].join("|");
}

function resetSelectionSelect(selectElement, placeholderText) {
  selectElement.innerHTML = `<option value="">${escapeHtml(placeholderText)}</option>`;
  selectElement.value = "";
  selectElement.disabled = true;
}

function getSelectedValidatedStrategy() {
  const selectedId = elements.strategySelect.value;
  if (!selectedId || !state.modelValidation?.strategies) {
    return null;
  }
  return (
    state.modelValidation.strategies.find((item) => item.id === selectedId) || null
  );
}

function refreshScorerOptionsForSelectedStrategy() {
  const strategy = getSelectedValidatedStrategy();
  const scorers = Array.isArray(state.modelValidation?.scorers)
    ? state.modelValidation.scorers
    : [];

  if (!strategy) {
    resetSelectionSelect(elements.scorerSelect, "Select strategy first");
    return;
  }

  if (strategy.requires_scorer === false) {
    resetSelectionSelect(elements.scorerSelect, "Not used for baseline");
    return;
  }

  elements.scorerSelect.innerHTML = scorers
    .map(
      (scorer) =>
        `<option value="${escapeHtml(scorer.id)}">${escapeHtml(scorer.name)}</option>`,
    )
    .join("");
  elements.scorerSelect.disabled = !scorers.length;
  if (scorers.length) {
    elements.scorerSelect.value = scorers[0].id;
  }
}

function updateRunButtonEnabled() {
  const selectedStrategy = getSelectedValidatedStrategy();
  const scorerRequired = selectedStrategy?.requires_scorer !== false;
  const hasScorer = !scorerRequired || Boolean(elements.scorerSelect.value);

  elements.runCustomButton.disabled =
    !state.modelValidation ||
    !elements.strategySelect.value ||
    !hasScorer;
}

function invalidateModelValidation(message = null) {
  state.modelValidation = null;
  state.validatedModelFingerprint = null;
  state.advancedConfigTemplateKey = null;
  state.advancedConfigDirty = false;
  resetSelectionSelect(elements.strategySelect, "Validate model first");
  resetSelectionSelect(elements.scorerSelect, "Validate model first");
  setAdvancedConfigYamlValue("");
  setAdvancedConfigEditorEnabled(false);
  setAdvancedConfigStatus("Validate model first to load advanced config YAML.", false);
  updateRunButtonEnabled();

  if (message) {
    setCapabilityStatus(message, false);
  }
}

function renderValidationOptions(strategies, scorers) {
  elements.strategySelect.innerHTML = strategies
    .map(
      (strategy) =>
        `<option value="${escapeHtml(strategy.id)}">${escapeHtml(strategy.name)}</option>`,
    )
    .join("");

  elements.strategySelect.disabled = !strategies.length;

  if (strategies.length) {
    elements.strategySelect.value = strategies[0].id;
  }
  state.modelValidation = {
    ...state.modelValidation,
    strategies,
    scorers,
  };

  refreshScorerOptionsForSelectedStrategy();
  refreshAdvancedConfigTemplate(true).catch((error) => {
    setAdvancedConfigStatus(
      `Failed to refresh advanced config template: ${error.message}`,
      true,
    );
  });
  updateRunButtonEnabled();
}

async function validateModelConfig() {
  if (state.useCachedExample) {
    setCapabilityStatus("Disable cached example mode to validate a model.", true);
    return;
  }
  state.cachedScenarioPrompt = "";

  const provider = elements.providerSelect.value.trim();
  const modelId = elements.modelIdInput.value.trim();
  const apiKey = elements.modelApiKeyInput.value.trim();

  if (!provider) {
    setCapabilityStatus("Please select a provider.", true);
    return;
  }
  if (!modelId) {
    setCapabilityStatus("Please input a model ID.", true);
    return;
  }
  if (!apiKey) {
    setCapabilityStatus("Please input an API key.", true);
    return;
  }

  elements.validateModelButton.disabled = true;
  setCapabilityStatus("Validating model capabilities...");

  try {
    const validation = await postJson("/v1/debugger/demo/validate-model", {
      provider,
      model_id: modelId,
      api_key: apiKey,
    });

    const strategies = Array.isArray(validation.strategies)
      ? validation.strategies
      : [];
    const scorers = Array.isArray(validation.scorers) ? validation.scorers : [];

    state.modelValidation = validation;
    state.validatedModelFingerprint = getModelFingerprint();
    renderValidationOptions(strategies, scorers);

    const logprobsText = validation.supports_logprobs
      ? "logprobs=yes"
      : "logprobs=no";
    const prefillText = validation.supports_prefill
      ? "prefill=yes"
      : "prefill=no";
    setCapabilityStatus(
      `Validated ${provider}:${modelId} (${logprobsText}, ${prefillText}, key=${maskApiKey(apiKey)}).`,
    );
    setStatus(
      "Model validated. Choose strategy (and scorer if required), then run one sample.",
      false,
    );
  } catch (error) {
    invalidateModelValidation();
    setCapabilityStatus(
      `Model validation failed: ${error.message}.`,
      true,
    );
  } finally {
    elements.validateModelButton.disabled = false;
  }
}

function pickStrategyEntryFromPayload(payload, strategyId, scorerId) {
  const runs = Array.isArray(payload?.strategies) ? payload.strategies : [];
  if (!runs.length) {
    return null;
  }

  const exact = runs.find(
    (item) => item.strategy_id === strategyId && item.scorer_id === scorerId,
  );
  if (exact) {
    return exact;
  }

  return runs.find((item) => item.strategy_id === strategyId) || null;
}

function buildRunPayloadFromCachedSource(basePayload, strategyId, scorerId) {
  const payload = deepClone(basePayload);
  const selected = pickStrategyEntryFromPayload(payload, strategyId, scorerId);
  if (!selected) {
    throw new Error("Selected strategy/scorer is not available in this cached example.");
  }

  payload.strategies = [selected];
  payload.strategy_catalog = (payload.strategy_catalog || []).filter(
    (item) => item.id === selected.strategy_id,
  );
  payload.scorer_catalog = selected.scorer_id
    ? (payload.scorer_catalog || []).filter((item) => item.id === selected.scorer_id)
    : [];

  payload.scenario = payload.scenario || {};
  payload.scenario.selected_strategy_id = selected.strategy_id;
  payload.scenario.selected_scorer_id = selected.scorer_id || null;
  payload.scenario.strategy_count = 1;
  payload.scenario.scorer_count = selected.scorer_id ? 1 : 0;
  payload.scenario.run_count = 1;

  return payload;
}

async function runCustomInput() {
  const strategyId = elements.strategySelect.value.trim();
  const selectedStrategy = getSelectedValidatedStrategy();
  const scorerRequired = selectedStrategy?.requires_scorer !== false;
  const scorerId = scorerRequired ? elements.scorerSelect.value.trim() : "";
  if (!strategyId || (scorerRequired && !scorerId)) {
    setStatus("Please validate model and finish required strategy/scorer selection.", true);
    return;
  }

  const budget = getCurrentBudget() ?? 8;

  if (state.useCachedExample) {
    try {
      if (!state.cachedSourcePayload) {
        await loadCachedOptionsForCurrentScenario();
      }
      const payload = buildRunPayloadFromCachedSource(
        state.cachedSourcePayload,
        strategyId,
        scorerId || null,
      );
      state.payload = payload;
      state.selectedStrategyId = payload?.strategies?.[0]?.id || null;
      state.selectedEventIndex = 0;
      state.selectedCandidateId = null;
      state.selectedTreeNodeId = null;
      render();
      setStatus("Loaded cached example run.", false);
    } catch (error) {
      setStatus(`Failed to load cached example run: ${error.message}`, true);
    }
    return;
  }

  const provider = elements.providerSelect.value.trim();
  const modelId = elements.modelIdInput.value.trim();
  const apiKey = elements.modelApiKeyInput.value.trim();
  const question = elements.singleQuestionInput.value.trim();
  const systemPrompt = String(elements.advancedPromptInput?.value || "").trim();
  const advancedConfigYaml = upsertPromptInAdvancedYaml(
    elements.advancedConfigYamlInput.value,
    systemPrompt,
  );
  setAdvancedConfigYamlValue(advancedConfigYaml);

  if (!question) {
    setStatus("Please input a question.", true);
    return;
  }
  if (!state.modelValidation || state.validatedModelFingerprint !== getModelFingerprint()) {
    setStatus("Model settings changed. Please validate model again.", true);
    return;
  }

  let payload;
  try {
    payload = await postJson("/v1/debugger/demo/run-single", {
      question,
      budget,
      provider,
      model_id: modelId,
      api_key: apiKey,
      strategy_id: strategyId,
      scorer_id: scorerRequired ? scorerId : null,
      advanced_config_yaml: advancedConfigYaml,
    });
  } catch (error) {
    setStatus(
      `Custom run requires backend endpoint /v1/debugger/demo/run-single (${error.message}).`,
      true,
    );
    return;
  }

  const scenarioId = payload?.scenario?.id || "custom_1";
  const selectedBudget = Number(payload?.selected_budget || budget);
  const scenarioTitle =
    payload?.scenario?.title ||
    (scorerRequired
      ? `Single Example · ${strategyId} · ${scorerId}`
      : `Single Example · ${strategyId}`);

  state.customPayloads = {
    [scenarioId]: {
      [String(selectedBudget)]: payload,
    },
  };

  state.catalog = [
    {
      id: scenarioId,
      title: scenarioTitle,
      description: "Custom question loaded by user",
      available_budgets: [selectedBudget],
      default_budget: selectedBudget,
    },
  ];
  state.dataMode = "custom";
  state.scenarioId = scenarioId;
  state.selectedStrategyId = null;
  state.selectedEventIndex = 0;
  state.selectedCandidateId = null;
  state.selectedTreeNodeId = null;

  populateScenarioSelect();
  configureCaseSelect(selectedBudget);
  await loadScenarioPayload();

  const strategyName =
    elements.strategySelect.options[elements.strategySelect.selectedIndex]?.text ||
    strategyId;
  const scorerName =
    elements.scorerSelect.options[elements.scorerSelect.selectedIndex]?.text ||
    scorerId;

  setStatus(
    scorerRequired
      ? `Ran ${strategyName} with ${scorerName} on selected case.`
      : `Ran ${strategyName} on selected case.`,
    false,
  );
}

async function restoreDemoData() {
  state.useCachedExample = false;
  state.cachedSourcePayload = null;
  state.customPayloads = {};
  state.payload = null;
  state.selectedStrategyId = null;
  state.selectedEventIndex = 0;
  state.selectedCandidateId = null;
  state.selectedTreeNodeId = null;
  state.modelValidation = null;
  state.validatedModelFingerprint = null;
  state.cachedScenarioPrompt = "";

  elements.providerSelect.value = "openai";
  elements.modelIdInput.value = "openai/gpt-4o-mini";
  elements.modelApiKeyInput.value = "";
  elements.singleQuestionInput.value = "";
  if (elements.advancedPromptInput) {
    elements.advancedPromptInput.value = DEFAULT_SYSTEM_PROMPT;
  }
  setAdvancedConfigYamlValue("");

  try {
    state.catalog = await loadCatalog();
    state.scenarioId = state.catalog[0]?.id || null;
    populateScenarioSelect();
    if (state.catalog.length) {
      configureCaseSelect(state.catalog[0].default_budget);
    } else {
      state.budgetOptions = [];
      elements.caseSelect.innerHTML = '<option value="">No cases</option>';
      elements.caseSelect.value = "";
      elements.caseSelect.disabled = true;
    }
  } catch (error) {
    setStatus(`Failed to reload demo scenarios: ${error.message}`, true);
  }

  invalidateModelValidation(
    "Validate a model first to unlock compatible strategy/scorer options.",
  );
  setAdvancedConfigPanelExpanded(false);
  applyCachedModeUi();
  clearRenderedResults();
  setStatus("Cleared all inputs and results.", false);
}

async function loadScenarioPayload() {
  const budget = getCurrentBudget();
  if (!state.scenarioId || budget == null) {
    return;
  }

  const payload = await loadPayloadForScenario(state.scenarioId, budget);
  state.payload = payload;

  const payloadStrategies = Array.isArray(payload.strategies) ? payload.strategies : [];
  const currentStrategyExists = payloadStrategies.some(
    (strategy) => strategy.id === state.selectedStrategyId,
  );

  if (!currentStrategyExists) {
    const best = [...payloadStrategies].sort(
      (left, right) =>
        (left.comparison_rank ?? Number.MAX_SAFE_INTEGER) -
        (right.comparison_rank ?? Number.MAX_SAFE_INTEGER),
    )[0];
    state.selectedStrategyId = best?.id ?? null;
    state.selectedEventIndex = 0;
    state.selectedCandidateId = null;
    state.selectedTreeNodeId = null;
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
  if (scenario?.selected_strategy_id && scenario?.selected_scorer_id) {
    metadataParts.push(
      `selected=${scenario.selected_strategy_id}/${scenario.selected_scorer_id}`,
    );
  } else if (scenario?.selected_strategy_id) {
    metadataParts.push(`selected=${scenario.selected_strategy_id}`);
  } else if (scenario?.run_count) {
    metadataParts.push(`runs=${scenario.run_count}`);
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
    const scorerLabel = run.scorer?.name || strategy.scorer_id || "";
    const isActive = strategy.id === state.selectedStrategyId;
    const card = document.createElement("article");
    card.className = `strategy-card${isActive ? " active" : ""}`;
    card.style.animationDelay = `${index * 50}ms`;

    const rank = strategy.comparison_rank || 1;
    const scorerMeta = scorerLabel
      ? `<p class="timeline-step">scorer · ${escapeHtml(scorerLabel)}</p>`
      : "";

    card.innerHTML = `
      <div class="strategy-title">
        <h3>${escapeHtml(strategyLabel)}</h3>
        <span class="rank-pill">rank #${rank}</span>
      </div>
      ${scorerMeta}
      <p class="timeline-decision">${escapeHtml(strategy.summary || "")}</p>
      <div class="strategy-meta">
        <div><span class="timeline-step">confidence</span><br /><span class="meta-value">${formatMetric(finalResult.confidence ?? 0)}</span></div>
        <div><span class="timeline-step">tokens</span><br /><span class="meta-value">${formatMetric(run.tokens_used ?? 0)}</span></div>
      </div>
    `;

    card.addEventListener("click", () => {
      state.selectedStrategyId = strategy.id;
      state.selectedEventIndex = 0;
      state.selectedTreeNodeId = null;
      selectFirstCandidate(strategy.run?.events?.[0]);
      render();
    });

    elements.strategyGrid.appendChild(card);
  });

  if (!strategies.length) {
    elements.strategyGrid.innerHTML =
      '<p class="tree-empty">No strategy runs available for this payload.</p>';
  }
}

function renderTimelineOptions(eventItem) {
  const candidates = eventItem?.candidates ?? [];
  if (!candidates.length) {
    return '<p class="timeline-options-empty">No options recorded for this step.</p>';
  }

  return `
    <div class="timeline-options">
      ${candidates
        .map((candidate) => {
          const status = candidate.status || "kept";
          const statusClass = status === "selected" ? " selected" : "";
          const signalEntry = Object.entries(candidate.signals || {})[0];
          const signalText = signalEntry
            ? `${signalEntry[0]}: ${formatMetric(signalEntry[1])}`
            : "";
          const meta = [status, signalText].filter(Boolean).join(" · ");
          return `
            <div class="timeline-option${statusClass}">
              <p class="timeline-option-label">${escapeHtml(candidate.label || candidate.id || "option")}</p>
              <p class="timeline-option-meta">${escapeHtml(meta)}</p>
            </div>
          `;
        })
        .join("")}
    </div>
  `;
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
    : "Select a step to inspect content and scores.";

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
    const optionsHtml = renderTimelineOptions(eventItem);
    node.innerHTML = `
      <p class="timeline-step">step ${eventItem.step} · ${escapeHtml(eventItem.stage || "")}</p>
      <p class="timeline-title">${escapeHtml(eventItem.title || "")}</p>
      <p class="timeline-decision"><strong>${escapeHtml(eventItem.decision?.action || "")}</strong> · ${escapeHtml(eventItem.decision?.reason || "")}</p>
      ${optionsHtml}
    `;

    node.addEventListener("click", () => {
      state.selectedEventIndex = index;
      state.selectedTreeNodeId = null;
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
      state.selectedTreeNodeId = null;
      state.selectedCandidateId = candidate.id;
      renderCandidates(eventItem);
      renderCandidateDetail(eventItem);
    });

    elements.candidates.appendChild(card);
  });

  renderCandidateDetail(eventItem);
}

function renderCandidateDetail(eventItem) {
  const selectedStrategy = getSelectedStrategy();
  const activeTreeNode = state.selectedTreeNodeId
    ? getActiveTreeNode(selectedStrategy)
    : null;
  if (activeTreeNode) {
    const treeContext = resolveTreeNodeInspectorContext(
      activeTreeNode,
      selectedStrategy?.run?.events || [],
    );
    const nodeMetrics = Object.entries(treeContext.scores || {})
      .map(
        ([key, value]) =>
          `<span>${escapeHtml(key)}: <strong>${formatMetric(value)}</strong></span>`,
      )
      .join("");

    elements.candidateDetail.innerHTML = `
      <pre>${escapeHtml(treeContext.text || "No reasoning text available.")}</pre>
      <div class="candidate-detail-metrics">${nodeMetrics || "<span>No node scores.</span>"}</div>
    `;
    return;
  }

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

function pickPrimaryNumericScore(signals) {
  const entries = Object.entries(signals || {});
  for (const [, value] of entries) {
    const numeric = Number(value);
    if (Number.isFinite(numeric)) {
      return numeric;
    }
  }
  return null;
}

function pickEventSignalValue(eventItem, signalName = "confidence") {
  const signals = Array.isArray(eventItem?.signals) ? eventItem.signals : [];
  const match = signals.find((signal) => signal?.name === signalName);
  const value = Number(match?.value);
  if (Number.isFinite(value)) {
    return value;
  }
  return null;
}

function buildTreeFromEvents(events) {
  const eventList = Array.isArray(events) ? events : [];
  if (!eventList.length) {
    return null;
  }

  const firstEvent = eventList[0] || {};
  const firstStep = Math.max(1, Number(firstEvent.step) || 1);
  const firstCandidates = Array.isArray(firstEvent.candidates)
    ? firstEvent.candidates
    : [];
  const rootScoreMap = {};
  (firstEvent.signals || []).forEach((signal) => {
    const name = String(signal?.name || "").trim();
    const value = Number(signal?.value);
    if (name && Number.isFinite(value)) {
      rootScoreMap[name] = value;
    }
  });

  const rootValue =
    pickEventSignalValue(firstEvent, "confidence") ??
    pickPrimaryNumericScore(rootScoreMap) ??
    0.5;
  const rootTextParts = [];
  if (typeof firstEvent.title === "string" && firstEvent.title.trim()) {
    rootTextParts.push(firstEvent.title.trim());
  }
  const firstReason = firstEvent?.decision?.reason;
  if (typeof firstReason === "string" && firstReason.trim()) {
    rootTextParts.push(firstReason.trim());
  }

  const rootId = `step_${firstStep}_root`;
  const nodes = [
    {
      id: rootId,
      label: "Root",
      value: rootValue,
      depth: 0,
      x: 0.5,
      y: 0.14,
      step: firstStep,
      text: rootTextParts.join(" "),
      scores: rootScoreMap,
    },
  ];
  const edges = [];
  const selectedPath = [rootId];
  const nodeIdSet = new Set([rootId]);

  let selectedFirstNodeId = rootId;
  let selectedFirstNodeX = 0.5;
  const firstSelectedCandidate =
    firstCandidates.find((candidate) => candidate?.selected) ||
    firstCandidates[0] ||
    null;

  firstCandidates.forEach((candidate, index) => {
    const total = firstCandidates.length;
    const x =
      total <= 1
        ? 0.5
        : 0.15 + (0.7 * index) / Math.max(total - 1, 1);
    const rawNodeId = String(candidate?.id || `${rootId}_c${index + 1}`);
    const nodeId = nodeIdSet.has(rawNodeId)
      ? `${rawNodeId}__${index + 1}`
      : rawNodeId;
    nodeIdSet.add(nodeId);

    const nodeValue =
      pickPrimaryNumericScore(candidate?.signals || {}) ?? rootValue;
    const node = {
      id: nodeId,
      label: String(candidate?.label || `Candidate ${index + 1}`),
      value: nodeValue,
      depth: 1,
      x,
      y: 0.48,
      step: firstStep,
      candidate_id: typeof candidate?.id === "string" ? candidate.id : null,
      text: String(candidate?.text || ""),
      status: String(candidate?.status || ""),
      scores:
        candidate?.signals && typeof candidate.signals === "object"
          ? candidate.signals
          : {},
    };
    nodes.push(node);
    edges.push({ source: rootId, target: nodeId });

    if (
      firstSelectedCandidate &&
      ((candidate?.id && candidate.id === firstSelectedCandidate.id) ||
        candidate === firstSelectedCandidate)
    ) {
      selectedFirstNodeId = nodeId;
      selectedFirstNodeX = x;
    }
  });

  if (selectedFirstNodeId !== rootId) {
    selectedPath.push(selectedFirstNodeId);
  }

  const secondEvent = eventList[1] || null;
  if (secondEvent) {
    const secondCandidates = Array.isArray(secondEvent.candidates)
      ? secondEvent.candidates
      : [];
    const secondSelected =
      secondCandidates.find((candidate) => candidate?.selected) ||
      secondCandidates[0] ||
      null;
    const secondStep = Math.max(1, Number(secondEvent.step) || firstStep + 1);

    if (secondSelected) {
      const rawSecondNodeId = String(
        secondSelected.id || `step_${secondStep}_selected`,
      );
      const secondNodeId = nodeIdSet.has(rawSecondNodeId)
        ? `${rawSecondNodeId}__step${secondStep}`
        : rawSecondNodeId;
      const secondValue =
        pickPrimaryNumericScore(secondSelected?.signals || {}) ??
        pickEventSignalValue(secondEvent, "confidence") ??
        rootValue;

      nodes.push({
        id: secondNodeId,
        label: String(secondSelected.label || "Selected"),
        value: secondValue,
        depth: 2,
        x: selectedFirstNodeX,
        y: 0.82,
        step: secondStep,
        candidate_id:
          typeof secondSelected.id === "string" ? secondSelected.id : null,
        text: String(secondSelected.text || ""),
        status: String(secondSelected.status || ""),
        scores:
          secondSelected?.signals && typeof secondSelected.signals === "object"
            ? secondSelected.signals
            : {},
      });
      edges.push({ source: selectedFirstNodeId, target: secondNodeId });
      selectedPath.push(secondNodeId);
    }
  }

  return { nodes, edges, selected_path: selectedPath };
}

function getStrategyTree(selectedStrategy) {
  const derivedTree = buildTreeFromEvents(selectedStrategy?.run?.events || []);
  if (derivedTree?.nodes?.length) {
    return derivedTree;
  }
  return selectedStrategy?.run?.tree || null;
}

function resolveTreeNodeInspectorContext(node, events) {
  const eventList = Array.isArray(events) ? events : [];
  if (!node || !eventList.length) {
    return { eventIndex: null, candidateId: null, text: "", scores: {} };
  }

  const explicitStep = Number(node.step);
  const derivedStep = Number.isFinite(explicitStep)
    ? Math.max(1, explicitStep)
    : Number.isFinite(Number(node.depth))
      ? Math.max(1, Number(node.depth) + 1)
      : 1;
  const eventIndex = Math.min(derivedStep - 1, eventList.length - 1);
  const eventItem = eventList[eventIndex];
  const candidates = eventItem?.candidates ?? [];
  const hasNodeText =
    typeof node.text === "string" && node.text.trim().length > 0;
  const hasNodeScores =
    node.scores &&
    typeof node.scores === "object" &&
    Object.keys(node.scores).length > 0;
  const hasExplicitCandidateId =
    typeof node.candidate_id === "string" && node.candidate_id.length > 0;

  let candidate = hasExplicitCandidateId
    ? candidates.find((item) => item.id === node.candidate_id) || null
    : null;
  if (!candidate && hasNodeText) {
    candidate =
      candidates.find((item) => item.text === node.text) ||
      null;
  }
  if (!candidate && !hasNodeText && !hasNodeScores && candidates.length) {
    const nodeValue = Number(node.value);
    if (Number.isFinite(nodeValue)) {
      candidate = candidates.reduce((best, current) => {
        const bestScore = pickPrimaryNumericScore(best?.signals || {});
        const currentScore = pickPrimaryNumericScore(current?.signals || {});
        const bestDistance =
          bestScore == null ? Number.POSITIVE_INFINITY : Math.abs(bestScore - nodeValue);
        const currentDistance =
          currentScore == null
            ? Number.POSITIVE_INFINITY
            : Math.abs(currentScore - nodeValue);
        return currentDistance < bestDistance ? current : best;
      }, null);
    }
  }
  if (!candidate && !hasNodeText && !hasNodeScores && candidates.length) {
    candidate = candidates.find((item) => item.selected) || candidates[0];
  }

  const scores =
    node.scores && typeof node.scores === "object"
      ? node.scores
      : candidate?.signals || {};
  const text =
    typeof node.text === "string" && node.text.trim()
      ? node.text
      : candidate?.text || "";

  return {
    eventIndex,
    candidateId: candidate?.id || null,
    text,
    scores,
  };
}

function getActiveTreeNode(selectedStrategy) {
  const tree = getStrategyTree(selectedStrategy);
  const nodes = tree?.nodes || [];
  if (!nodes.length) {
    return null;
  }
  return nodes.find((node) => node.id === state.selectedTreeNodeId) || null;
}

function applyTreeNodeSelection(nodeId) {
  const selectedStrategy = getSelectedStrategy();
  const tree = getStrategyTree(selectedStrategy);
  const nodes = tree?.nodes || [];
  const node = nodes.find((item) => item.id === nodeId);
  if (!node) {
    return;
  }

  state.selectedTreeNodeId = node.id;
  const events = selectedStrategy?.run?.events || [];
  const context = resolveTreeNodeInspectorContext(node, events);
  if (context.eventIndex != null && events[context.eventIndex]) {
    state.selectedEventIndex = context.eventIndex;
    if (context.candidateId) {
      state.selectedCandidateId = context.candidateId;
    } else {
      state.selectedCandidateId = null;
    }
  }

  renderTimeline();
  renderStepInspector();
}

function renderTree() {
  const selectedStrategy = getSelectedStrategy();
  const tree = getStrategyTree(selectedStrategy);

  if (!tree?.nodes?.length) {
    elements.treeContainer.innerHTML =
      '<p class="tree-empty">No tree structure for this strategy.</p>';
    return;
  }

  const width = 620;
  const height = 230;
  const nodeMap = new Map(tree.nodes.map((node) => [node.id, node]));
  const selectedPath = Array.isArray(tree.selected_path) ? tree.selected_path : [];
  const activeNodeId = nodeMap.has(state.selectedTreeNodeId)
    ? state.selectedTreeNodeId
    : null;

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

  const selectedNodeSet = new Set(selectedPath);
  const nodes = (tree.nodes || [])
    .map((node) => {
      const x = node.x * width;
      const y = node.y * height;
      const isSelected = selectedNodeSet.has(node.id);
      const isActive = node.id === activeNodeId;
      const radius = 8 + Number(node.value || 0) * 5;
      const nodeClass = `${isSelected ? "tree-node selected" : "tree-node"}${isActive ? " focused" : ""}`;
      const label = `${node.label || node.id} (${formatMetric(node.value)})`;
      const groupClass = `tree-node-group${isActive ? " active" : ""}`;

      return `
        <g class="${groupClass}" data-node-id="${escapeHtml(node.id)}">
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

  const clickableNodes = elements.treeContainer.querySelectorAll("[data-node-id]");
  clickableNodes.forEach((nodeElement) => {
    nodeElement.addEventListener("click", (event) => {
      event.stopPropagation();
      const nodeId = nodeElement.getAttribute("data-node-id");
      if (!nodeId) {
        return;
      }
      applyTreeNodeSelection(nodeId);
    });
  });
}

function renderStepInspector() {
  const selectedStrategy = getSelectedStrategy();
  const eventItem = selectedStrategy?.run?.events?.[state.selectedEventIndex];

  if (!eventItem) {
    elements.stepTitle.textContent = "Pick a timeline step.";
    elements.decisionBox.innerHTML =
      '<p class="tree-empty">Decision details appear for the selected step.</p>';
    elements.signals.innerHTML = "";
    elements.candidates.innerHTML = "";
    elements.candidateDetail.innerHTML =
      '<p class="tree-empty">Candidate detail appears when a candidate is selected.</p>';
    renderTree();
    return;
  }

  const candidates = eventItem?.candidates ?? [];
  const highlightedCandidate =
    candidates.find((item) => item.id === state.selectedCandidateId) ||
    candidates.find((item) => item.selected) ||
    candidates[0];
  const activeTreeNode = getActiveTreeNode(selectedStrategy);
  const treeContext = resolveTreeNodeInspectorContext(
    activeTreeNode,
    selectedStrategy?.run?.events || [],
  );
  const stepContent =
    treeContext.text ||
    highlightedCandidate?.text ||
    "No step content available.";
  const nodeScores = Object.entries(treeContext.scores || {})
    .map(
      ([key, value]) =>
        `<span>${escapeHtml(key)}: <strong>${formatMetric(value)}</strong></span>`,
    )
    .join("");
  const nodeLabel = activeTreeNode?.label
    ? ` · node ${activeTreeNode.label}`
    : "";

  elements.stepTitle.textContent = `${eventItem.title || "Step"}${nodeLabel}`;
  elements.decisionBox.innerHTML = `
    <p><strong>${escapeHtml(eventItem.decision?.action || "decision")}</strong></p>
    <p>${escapeHtml(eventItem.decision?.reason || "No decision rationale")}</p>
    <pre class="step-content">${escapeHtml(stepContent)}</pre>
    <div class="candidate-detail-metrics">${nodeScores || "<span>No node scores.</span>"}</div>
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
    configureCaseSelect(selectedScenario?.default_budget);
    state.cachedSourcePayload = null;
    clearRenderedResults();
    if (state.useCachedExample) {
      await loadCachedOptionsForCurrentScenario();
    }
  });

  elements.caseSelect.addEventListener("change", async () => {
    state.cachedSourcePayload = null;
    clearRenderedResults();
    if (state.useCachedExample) {
      await loadCachedOptionsForCurrentScenario();
    }
  });

  elements.validateModelButton.addEventListener("click", async () => {
    await validateModelConfig();
  });

  elements.useCachedToggle.addEventListener("change", async (event) => {
    state.useCachedExample = Boolean(event.target.checked);
    state.cachedSourcePayload = null;
    state.cachedScenarioPrompt = "";
    clearRenderedResults();

    if (state.useCachedExample) {
      invalidateModelValidation("Cached example mode enabled.");
      try {
        if (state.dataMode === "custom") {
          state.catalog = await loadCatalog();
          state.scenarioId = state.catalog[0]?.id || null;
          populateScenarioSelect();
          if (state.catalog.length) {
            configureCaseSelect(state.catalog[0].default_budget);
          }
        }
        await loadCachedOptionsForCurrentScenario();
        setCapabilityStatus(
          "Cached example mode: model/question fields are disabled.",
          false,
        );
        setStatus("Choose strategy/scorer and click Run.", false);
      } catch (error) {
        state.useCachedExample = false;
        invalidateModelValidation(
          "Validate a model first to unlock compatible strategy/scorer options.",
        );
        setStatus(`Failed to enable cached example mode: ${error.message}`, true);
      }
    } else {
      invalidateModelValidation(
        "Validate a model first to unlock compatible strategy/scorer options.",
      );
      setStatus("Cached example mode disabled.", false);
    }

    applyCachedModeUi();
    updateRunButtonEnabled();
  });

  [
    elements.providerSelect,
    elements.modelIdInput,
    elements.modelApiKeyInput,
  ].forEach((field) => {
    field.addEventListener("input", () => {
      if (
        state.validatedModelFingerprint &&
        state.validatedModelFingerprint !== getModelFingerprint()
      ) {
        invalidateModelValidation(
          "Model settings changed. Validate again to refresh supported options.",
        );
      }
    });
  });

  elements.strategySelect.addEventListener("change", () => {
    refreshScorerOptionsForSelectedStrategy();
    refreshAdvancedConfigTemplate(true).catch((error) => {
      setAdvancedConfigStatus(
        `Failed to refresh advanced config template: ${error.message}`,
        true,
      );
    });
    updateRunButtonEnabled();
  });

  elements.scorerSelect.addEventListener("change", () => {
    refreshAdvancedConfigTemplate(true).catch((error) => {
      setAdvancedConfigStatus(
        `Failed to refresh advanced config template: ${error.message}`,
        true,
      );
    });
    updateRunButtonEnabled();
  });

  elements.advancedConfigToggle.addEventListener("click", () => {
    setAdvancedConfigPanelExpanded(!state.advancedConfigExpanded);
  });

  elements.resetAdvancedConfigButton.addEventListener("click", () => {
    refreshAdvancedConfigTemplate(true).catch((error) => {
      setAdvancedConfigStatus(
        `Failed to reset advanced config template: ${error.message}`,
        true,
      );
    });
  });

  elements.advancedPromptInput?.addEventListener("input", () => {
    state.cachedScenarioPrompt = "";
    const syncedYaml = upsertPromptInAdvancedYaml(
      elements.advancedConfigYamlInput.value,
      elements.advancedPromptInput.value,
    );
    setAdvancedConfigYamlValue(syncedYaml);
    state.advancedConfigDirty = true;
  });

  elements.advancedConfigYamlInput.addEventListener("input", () => {
    state.advancedConfigDirty = true;
    renderAdvancedConfigHighlight();
  });

  elements.advancedConfigYamlInput.addEventListener("scroll", () => {
    renderAdvancedConfigHighlight();
  });

  elements.advancedConfigYamlInput.addEventListener("focus", () => {
    renderAdvancedConfigHighlight();
  });

  elements.advancedConfigYamlInput.addEventListener("keydown", (event) => {
    if (event.key !== "Tab") {
      return;
    }
    event.preventDefault();
    const input = elements.advancedConfigYamlInput;
    const start = input.selectionStart;
    const end = input.selectionEnd;
    input.setRangeText("  ", start, end, "end");
    state.advancedConfigDirty = true;
    renderAdvancedConfigHighlight();
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
  setAdvancedConfigPanelExpanded(false);
  renderAdvancedConfigHighlight();
  applyCachedModeUi();
  invalidateModelValidation(
    "Validate a model first to unlock compatible strategy/scorer options.",
  );

  state.catalog = await loadCatalog();

  if (!state.catalog.length) {
    elements.strategyGrid.innerHTML =
      '<p class="tree-empty">No debugger scenarios are available.</p>';
    return;
  }

  state.scenarioId = state.catalog[0].id;
  populateScenarioSelect();
  configureCaseSelect(state.catalog[0].default_budget);
  clearRenderedResults();
  setStatus("No result yet. Fill inputs or enable cached example mode, then click Run.", false);
}

init().catch((error) => {
  elements.strategyGrid.innerHTML = `<p class="tree-empty">Failed to load debugger data: ${escapeHtml(error.message)}</p>`;
});
