const CUSTOM_BUDGETS = [4, 8, 12];

const STRATEGY_LIBRARY = [
  {
    id: "baseline",
    name: "Baseline (Raw CoT)",
    family: "single_pass",
    summary: "Single-pass raw chain-of-thought without search or reranking.",
  },
  {
    id: "beam_search",
    name: "Beam Search (ToT)",
    family: "tree_search",
    summary: "Tree-of-thought expansion with beam pruning.",
  },
  {
    id: "online_best_of_n",
    name: "Online Best-of-N",
    family: "reranking",
    summary: "Iterative candidate generation with stepwise reranking.",
  },
  {
    id: "offline_best_of_n",
    name: "Offline Best-of-N",
    family: "reranking",
    summary: "Generate full trajectories first, then rerank at the end.",
  },
  {
    id: "self_consistency",
    name: "Self-Consistency",
    family: "sample_and_vote",
    summary: "Sample diverse trajectories and select by answer consensus.",
  },
];

const SCORER_LIBRARY = [
  {
    id: "prm",
    name: "PRM",
    direction: "higher_better",
    threshold: 0.72,
  },
  {
    id: "sequence_prob",
    name: "Sequence Prob",
    direction: "higher_better",
    threshold: 0.65,
  },
  {
    id: "perplexity",
    name: "Perplexity",
    direction: "lower_better",
    threshold: 0.36,
  },
  {
    id: "entropy",
    name: "Entropy",
    direction: "lower_better",
    threshold: 0.34,
  },
];

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

function clamp(value, min = 0, max = 1) {
  return Math.max(min, Math.min(max, value));
}

function hashString(text) {
  let hash = 2166136261;
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function createRng(seed) {
  let value = seed >>> 0;
  return () => {
    value = Math.imul(value + 0x6d2b79f5, 1);
    let result = Math.imul(value ^ (value >>> 15), 1 | value);
    result ^= result + Math.imul(result ^ (result >>> 7), 61 | result);
    return ((result ^ (result >>> 14)) >>> 0) / 4294967296;
  };
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

function getPrototypeScenarios() {
  return window.VISUAL_DEBUGGER_PROTOTYPE?.scenarios || [];
}

function getPrototypeScenarioPayload(scenarioId, budget) {
  const prototypePayloads = window.VISUAL_DEBUGGER_PROTOTYPE?.payloads || {};
  const scenarioPayloads = prototypePayloads[scenarioId];
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

  const targetBudget = budget ?? availableBudgets[0];
  const selectedBudget = availableBudgets.reduce((best, current) => {
    const bestGap = Math.abs(best - targetBudget);
    const currentGap = Math.abs(current - targetBudget);
    return currentGap < bestGap ? current : best;
  }, availableBudgets[0]);

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
    const scenarios = getPrototypeScenarios();
    if (scenarios.length) {
      state.dataMode = "prototype";
      return deepClone(scenarios);
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
    (best, budget, index) => {
      const bestGap = Math.abs(options[best] - target);
      const currentGap = Math.abs(budget - target);
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

function perturbAnswer(goldAnswer, rng) {
  const text = String(goldAnswer || "").trim();
  const numericValue = Number(text);
  if (text && !Number.isNaN(numericValue)) {
    const delta = Math.max(1, Math.floor(rng() * 5));
    const sign = rng() > 0.5 ? 1 : -1;
    return String(numericValue + sign * delta);
  }

  if (!text) {
    return "unknown";
  }

  return `${text}_alt`;
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

function getBudgetUnit(family) {
  if (family === "tree_search") {
    return "node_expansions";
  }
  if (family === "sample_and_vote") {
    return "paths";
  }
  if (family === "reranking") {
    return "candidate_rollouts";
  }
  return "steps";
}

function familyBaseQuality(family) {
  if (family === "tree_search") {
    return 0.67;
  }
  if (family === "sample_and_vote") {
    return 0.65;
  }
  if (family === "reranking") {
    return 0.64;
  }
  return 0.56;
}

function scorerQualityShift(scorerId) {
  if (scorerId === "prm") {
    return 0.05;
  }
  if (scorerId === "sequence_prob") {
    return 0.03;
  }
  if (scorerId === "perplexity") {
    return 0.01;
  }
  return 0;
}

function scorerValueFromConfidence(scorer, confidence) {
  if (scorer.direction === "higher_better") {
    return clamp(confidence);
  }
  return clamp(1 - confidence);
}

function scorerSignal(scorer, value) {
  return {
    name: scorer.id,
    value,
    direction: scorer.direction,
    threshold: scorer.threshold,
  };
}

function buildSinglePassEvents(
  strategy,
  scorer,
  context,
  finalAnswer,
  confidence,
  rng,
) {
  const scorerValue = scorerValueFromConfidence(scorer, confidence);
  return [
    {
      step: 1,
      title: "Single-pass reasoning generation",
      stage: "generation",
      decision: {
        action: "stop",
        reason:
          "Single-pass baseline emits one chain; scorer evaluates the complete trajectory.",
      },
      signals: [
        {
          name: "confidence",
          value: confidence,
          direction: "higher_better",
          threshold: 0.7,
        },
        scorerSignal(scorer, scorerValue),
      ],
      candidates: [
        {
          id: `${strategy.id}_${scorer.id}_c1`,
          label: "Generated chain",
          text: `${context.question} -> proposed answer ${finalAnswer}.`,
          answer: finalAnswer,
          status: "selected",
          selected: true,
          signals: {
            [scorer.id]: scorerValue,
            chain_score: clamp(confidence + (rng() - 0.5) * 0.05),
          },
        },
      ],
    },
  ];
}

function buildSampleVoteEvents(
  strategy,
  scorer,
  context,
  finalAnswer,
  confidence,
  rng,
  budget,
) {
  const warmupConsensus = clamp(0.42 + rng() * 0.2);
  const finalConsensus = clamp(confidence - 0.02);
  const wrongAnswer = perturbAnswer(context.goldAnswer, rng);

  const firstEvent = {
    step: 1,
    title: "Warmup trajectory sampling",
    stage: "sampling",
    decision: {
      action: budget > 4 ? "escalate" : "stop",
      reason:
        budget > 4
          ? "Consensus below threshold after initial samples."
          : "Sampling budget exhausted.",
    },
    signals: [
      {
        name: "consensus",
        value: warmupConsensus,
        direction: "higher_better",
        threshold: 0.65,
      },
      scorerSignal(
        scorer,
        scorerValueFromConfidence(
          scorer,
          clamp(confidence - 0.11 + (rng() - 0.5) * 0.05),
        ),
      ),
    ],
    candidates: [
      {
        id: `${strategy.id}_${scorer.id}_p1`,
        label: "Path 1",
        text: `Candidate reasoning path ends with ${finalAnswer}.`,
        answer: finalAnswer,
        status: "kept",
        selected: false,
        signals: {
          [scorer.id]: scorerValueFromConfidence(
            scorer,
            clamp(confidence - 0.08 + rng() * 0.08),
          ),
          path_score: clamp(confidence - 0.08 + rng() * 0.08),
          tokens: 170 + Math.floor(rng() * 80),
        },
      },
      {
        id: `${strategy.id}_${scorer.id}_p2`,
        label: "Path 2",
        text: `Alternative chain predicts ${wrongAnswer}.`,
        answer: wrongAnswer,
        status: "pruned",
        selected: false,
        signals: {
          [scorer.id]: scorerValueFromConfidence(
            scorer,
            clamp(confidence - 0.22 + rng() * 0.06),
          ),
          path_score: clamp(confidence - 0.22 + rng() * 0.06),
          tokens: 160 + Math.floor(rng() * 90),
        },
      },
    ],
  };

  if (budget <= 4) {
    firstEvent.candidates[0].status = "selected";
    firstEvent.candidates[0].selected = true;
    return [firstEvent];
  }

  return [
    firstEvent,
    {
      step: 2,
      title: "Escalated sampling and final vote",
      stage: "selection",
      decision: {
        action: "stop",
        reason: "Consensus crossed threshold after additional trajectories.",
      },
      signals: [
        {
          name: "consensus",
          value: finalConsensus,
          direction: "higher_better",
          threshold: 0.65,
        },
        scorerSignal(scorer, scorerValueFromConfidence(scorer, confidence)),
      ],
      candidates: [
        {
          id: `${strategy.id}_${scorer.id}_p3`,
          label: "Path 3",
          text: `Escalated path verifies answer ${finalAnswer}.`,
          answer: finalAnswer,
          status: "selected",
          selected: true,
          signals: {
            [scorer.id]: scorerValueFromConfidence(scorer, confidence),
            path_score: clamp(confidence + rng() * 0.06),
            tokens: 170 + Math.floor(rng() * 80),
          },
        },
      ],
    },
  ];
}

function buildRerankEvents(
  strategy,
  scorer,
  context,
  finalAnswer,
  confidence,
  rng,
  budget,
) {
  const wrongAnswer = perturbAnswer(context.goldAnswer, rng);
  const topGap = clamp(0.03 + rng() * 0.1, 0, 0.3);

  const firstEvent = {
    step: 1,
    title: "Candidate generation and scoring",
    stage: "candidate_generation",
    decision: {
      action: budget > 4 ? "escalate" : "rerank",
      reason:
        budget > 4
          ? "Top-2 scorer gap is narrow; request more candidates."
          : "Select best candidate under current budget.",
    },
    signals: [
      {
        name: "top2_gap",
        value: topGap,
        direction: "higher_better",
        threshold: 0.08,
      },
      scorerSignal(
        scorer,
        scorerValueFromConfidence(
          scorer,
          clamp(confidence - 0.1 + rng() * 0.08),
        ),
      ),
    ],
    candidates: [
      {
        id: `${strategy.id}_${scorer.id}_r1`,
        label: "Candidate 1",
        text: `Reasoning candidate with answer ${finalAnswer}.`,
        answer: finalAnswer,
        status: budget > 4 ? "kept" : "selected",
        selected: budget <= 4,
        signals: {
          [scorer.id]: scorerValueFromConfidence(
            scorer,
            clamp(confidence - 0.04 + rng() * 0.05),
          ),
          rerank_score: clamp(confidence - 0.06 + rng() * 0.07),
        },
      },
      {
        id: `${strategy.id}_${scorer.id}_r2`,
        label: "Candidate 2",
        text: `Competing candidate with answer ${wrongAnswer}.`,
        answer: wrongAnswer,
        status: "pruned",
        selected: false,
        signals: {
          [scorer.id]: scorerValueFromConfidence(
            scorer,
            clamp(confidence - 0.24 + rng() * 0.05),
          ),
          rerank_score: clamp(confidence - 0.2 + rng() * 0.06),
        },
      },
    ],
  };

  if (budget <= 4) {
    return [firstEvent];
  }

  return [
    firstEvent,
    {
      step: 2,
      title: "Escalated reranking",
      stage: "reranking",
      decision: {
        action: "stop",
        reason: "Rerank confidence stabilized after additional candidates.",
      },
      signals: [
        {
          name: "top2_gap",
          value: clamp(topGap + 0.08 + rng() * 0.07),
          direction: "higher_better",
          threshold: 0.08,
        },
        scorerSignal(scorer, scorerValueFromConfidence(scorer, confidence)),
      ],
      candidates: [
        {
          id: `${strategy.id}_${scorer.id}_r3`,
          label: "Candidate 3",
          text: `Escalated candidate selected with answer ${finalAnswer}.`,
          answer: finalAnswer,
          status: "selected",
          selected: true,
          signals: {
            [scorer.id]: scorerValueFromConfidence(
              scorer,
              clamp(confidence + rng() * 0.05),
            ),
            rerank_score: clamp(confidence + rng() * 0.04),
          },
        },
      ],
    },
  ];
}

function buildTree(strategy, scorer, confidence, rng) {
  const runKey = `${strategy.id}_${scorer.id}`;
  const root = 0.48 + rng() * 0.08;
  const left = clamp(root + 0.08 + rng() * 0.07);
  const mid = clamp(root + 0.12 + rng() * 0.06);
  const right = clamp(root - 0.14 + rng() * 0.08);
  return {
    nodes: [
      {
        id: `${runKey}_n1`,
        label: "Root",
        value: root,
        depth: 0,
        x: 0.5,
        y: 0.14,
      },
      {
        id: `${runKey}_n2`,
        label: "A",
        value: left,
        depth: 1,
        x: 0.24,
        y: 0.47,
      },
      {
        id: `${runKey}_n3`,
        label: "B",
        value: mid,
        depth: 1,
        x: 0.52,
        y: 0.47,
      },
      {
        id: `${runKey}_n4`,
        label: "C",
        value: right,
        depth: 1,
        x: 0.8,
        y: 0.47,
      },
      {
        id: `${runKey}_n5`,
        label: "B2",
        value: confidence,
        depth: 2,
        x: 0.52,
        y: 0.82,
      },
    ],
    edges: [
      { source: `${runKey}_n1`, target: `${runKey}_n2` },
      { source: `${runKey}_n1`, target: `${runKey}_n3` },
      { source: `${runKey}_n1`, target: `${runKey}_n4` },
      { source: `${runKey}_n3`, target: `${runKey}_n5` },
    ],
    selected_path: [`${runKey}_n1`, `${runKey}_n3`, `${runKey}_n5`],
  };
}

function buildTreeEvents(
  strategy,
  scorer,
  context,
  finalAnswer,
  confidence,
  rng,
  budget,
) {
  const firstBest = clamp(0.58 + rng() * 0.12);
  const firstEvent = {
    step: 1,
    title: "Depth-1 tree expansion",
    stage: "tree_expand",
    decision: {
      action: budget > 4 ? "continue" : "stop",
      reason:
        budget > 4
          ? "Need deeper expansion to disambiguate branches."
          : "Expansion budget exhausted.",
    },
    signals: [
      {
        name: "best_value",
        value: firstBest,
        direction: "higher_better",
        threshold: 0.75,
      },
      scorerSignal(
        scorer,
        scorerValueFromConfidence(
          scorer,
          clamp(confidence - 0.11 + (rng() - 0.5) * 0.05),
        ),
      ),
    ],
    candidates: [
      {
        id: `${strategy.id}_${scorer.id}_t1`,
        label: "Node A",
        text: `Candidate node explores partial reasoning toward ${finalAnswer}.`,
        answer: "",
        status: "kept",
        selected: true,
        signals: {
          [scorer.id]: scorerValueFromConfidence(
            scorer,
            clamp(confidence - 0.08 + rng() * 0.04),
          ),
          value: firstBest,
          depth: 1,
        },
      },
      {
        id: `${strategy.id}_${scorer.id}_t2`,
        label: "Node B",
        text: `Alternative node drifts to ${perturbAnswer(context.goldAnswer, rng)}.`,
        answer: "",
        status: "pruned",
        selected: false,
        signals: {
          [scorer.id]: scorerValueFromConfidence(
            scorer,
            clamp(confidence - 0.24 + rng() * 0.04),
          ),
          value: clamp(firstBest - 0.2 + rng() * 0.06),
          depth: 1,
        },
      },
    ],
  };

  const events = [firstEvent];
  if (budget > 4) {
    events.push({
      step: 2,
      title: "Depth-2 selection",
      stage: "selection",
      decision: {
        action: "stop",
        reason: "Highest-value branch reaches solve threshold.",
      },
      signals: [
        {
          name: "best_value",
          value: confidence,
          direction: "higher_better",
          threshold: 0.75,
        },
        scorerSignal(scorer, scorerValueFromConfidence(scorer, confidence)),
      ],
      candidates: [
        {
          id: `${strategy.id}_${scorer.id}_t3`,
          label: "Node A2",
          text: `Selected branch outputs answer ${finalAnswer}.`,
          answer: finalAnswer,
          status: "selected",
          selected: true,
          signals: {
            [scorer.id]: scorerValueFromConfidence(scorer, confidence),
            value: confidence,
            depth: 2,
          },
        },
      ],
    });
  }

  return {
    events,
    tree: buildTree(strategy, scorer, confidence, rng),
  };
}

function buildCustomStrategyScorerRun(
  strategy,
  scorer,
  context,
  budget,
  sharedPrompt,
  modelConfig,
) {
  const seed = hashString(
    `${strategy.id}|${scorer.id}|${context.question}|${context.goldAnswer}|${budget}|${sharedPrompt}|${modelConfig.provider}|${modelConfig.model_id}`,
  );
  const rng = createRng(seed);
  const difficulty = clamp(0.23 + (hashString(context.question) % 35) / 100);
  const budgetFactor = budget / CUSTOM_BUDGETS[CUSTOM_BUDGETS.length - 1];
  const baseQuality = familyBaseQuality(strategy.family);
  const scorerShift = scorerQualityShift(scorer.id);

  const successChance = clamp(
    baseQuality +
      scorerShift +
      budgetFactor * 0.16 -
      difficulty * 0.2 +
      (rng() - 0.5) * 0.08,
  );
  const isCorrect = rng() < successChance;
  const answer = isCorrect
    ? String(context.goldAnswer || "")
    : perturbAnswer(context.goldAnswer, rng);

  const qualityScore = clamp(
    baseQuality +
      scorerShift +
      budgetFactor * 0.13 +
      (isCorrect ? 0.08 : -0.12) +
      (rng() - 0.5) * 0.06,
  );
  const confidence = clamp(
    qualityScore + (isCorrect ? 0.07 : -0.07) + (rng() - 0.5) * 0.05,
    0.05,
    0.98,
  );

  const run = {
    budget,
    budget_unit: getBudgetUnit(strategy.family),
    used_budget:
      strategy.family === "single_pass"
        ? 1
        : Math.max(2, Math.min(budget, Math.round(budget * (0.7 + rng() * 0.25)))),
    tokens_used: Math.round(560 + budget * 90 + qualityScore * 220 + rng() * 140),
    latency_ms: Math.round(3800 + budget * 780 + rng() * 1800),
    provider: modelConfig.provider,
    model_id: modelConfig.model_id,
    strategy: {
      id: strategy.id,
      name: strategy.name,
      family: strategy.family,
    },
    scorer: {
      id: scorer.id,
      name: scorer.name,
      direction: scorer.direction,
    },
    final: {
      answer,
      is_correct: isCorrect,
      selected_trajectory: `${strategy.id}_${scorer.id}_selected`,
      confidence,
      uncertainty: clamp(1 - confidence),
      quality_score: qualityScore,
      selection_reason: `${strategy.name} selected this trajectory using ${scorer.name} signals.`,
    },
    events: [],
  };

  if (strategy.family === "single_pass") {
    run.events = buildSinglePassEvents(
      strategy,
      scorer,
      context,
      answer,
      confidence,
      rng,
    );
  } else if (strategy.family === "sample_and_vote") {
    run.events = buildSampleVoteEvents(
      strategy,
      scorer,
      context,
      answer,
      confidence,
      rng,
      budget,
    );
  } else if (strategy.family === "reranking") {
    run.events = buildRerankEvents(
      strategy,
      scorer,
      context,
      answer,
      confidence,
      rng,
      budget,
    );
  } else {
    const treePayload = buildTreeEvents(
      strategy,
      scorer,
      context,
      answer,
      confidence,
      rng,
      budget,
    );
    run.events = treePayload.events;
    run.tree = treePayload.tree;
  }

  return {
    id: `${strategy.id}__${scorer.id}`,
    strategy_id: strategy.id,
    scorer_id: scorer.id,
    name: `${strategy.name} · ${scorer.name}`,
    family: strategy.family,
    summary: `${strategy.summary} Evaluated with ${scorer.name}.`,
    run,
    comparison_rank: 1,
  };
}

function applyStrategyRanks(strategies) {
  const ranked = [...strategies].sort(
    (left, right) =>
      (right.run?.final?.quality_score || 0) - (left.run?.final?.quality_score || 0),
  );

  const ranking = new Map(ranked.map((strategy, index) => [strategy.id, index + 1]));

  strategies.forEach((strategy) => {
    strategy.comparison_rank = ranking.get(strategy.id) || strategies.length;
  });
}

function buildCustomScenarioPayload(
  sample,
  budget,
  datasetName,
  index,
  sharedPrompt,
  modelConfig,
) {
  const prompt = sharedPrompt
    ? `${sharedPrompt}

Question: ${sample.question}`
    : sample.question;

  const context = {
    question: sample.question,
    goldAnswer: sample.gold_answer,
  };

  const runs = [];
  STRATEGY_LIBRARY.forEach((strategy) => {
    SCORER_LIBRARY.forEach((scorer) => {
      runs.push(
        buildCustomStrategyScorerRun(
          strategy,
          scorer,
          context,
          budget,
          sharedPrompt,
          modelConfig,
        ),
      );
    });
  });

  applyStrategyRanks(runs);

  return {
    scenario: {
      id: `custom_${index + 1}`,
      title: `${datasetName} #${index + 1}`,
      description:
        "Custom run generated in-browser from your input; each strategy is evaluated by PRM, sequence_prob, perplexity, and entropy.",
      prompt,
      ground_truth: sample.gold_answer,
      shared_prompt: sharedPrompt || "",
      input_source: "custom_single",
      model_config: {
        provider: modelConfig.provider,
        model_id: modelConfig.model_id,
        api_key_masked: modelConfig.api_key_masked,
      },
      strategy_count: STRATEGY_LIBRARY.length,
      scorer_count: SCORER_LIBRARY.length,
      run_count: runs.length,
    },
    available_budgets: CUSTOM_BUDGETS,
    selected_budget: budget,
    strategy_catalog: deepClone(STRATEGY_LIBRARY),
    scorer_catalog: deepClone(SCORER_LIBRARY),
    strategies: runs,
  };
}

function buildCustomRuns(samples, datasetName, sharedPrompt, modelConfig) {
  const catalog = [];
  const payloads = {};

  samples.forEach((sample, index) => {
    const scenarioId = `custom_${index + 1}`;
    const preview = String(sample.question).slice(0, 64);
    const title = `${datasetName} · sample ${index + 1} · ${preview}${sample.question.length > 64 ? "..." : ""}`;

    catalog.push({
      id: scenarioId,
      title,
      description: "Custom question loaded by user",
      available_budgets: CUSTOM_BUDGETS,
      default_budget: CUSTOM_BUDGETS[1],
    });

    payloads[scenarioId] = {};
    CUSTOM_BUDGETS.forEach((budget) => {
      payloads[scenarioId][String(budget)] = buildCustomScenarioPayload(
        sample,
        budget,
        datasetName,
        index,
        sharedPrompt,
        modelConfig,
      );
      payloads[scenarioId][String(budget)].scenario.id = scenarioId;
      payloads[scenarioId][String(budget)].scenario.title = title;
    });
  });

  return { catalog, payloads };
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

  const samples = [{ question, gold_answer: goldAnswer }];
  const modelConfig = {
    provider,
    model_id: modelId,
    api_key_raw: apiKey,
    api_key_masked: maskApiKey(apiKey),
  };

  let customRuns;
  let usedBackendRunner = true;
  try {
    customRuns = await buildCustomRunsViaBackend(
      samples[0],
      sharedPrompt,
      modelConfig,
    );
  } catch (error) {
    usedBackendRunner = false;
    customRuns = buildCustomRuns(
      samples,
      "Single Example",
      sharedPrompt,
      modelConfig,
    );
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
    `Ran ${STRATEGY_LIBRARY.length * SCORER_LIBRARY.length} strategy-scorer runs for your question with ${provider}:${modelId}. ${usedBackendRunner ? "Used backend /v1/debugger/demo/run-single." : "Backend run endpoint unavailable, used local prototype runner."} API key input is still a placeholder for future full backend model wiring.`,
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
    setStatus("Restored built-in demo data.", false);
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
      ? " | prototype mode (local example)"
      : state.dataMode === "custom"
        ? " | custom mode (user input)"
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
