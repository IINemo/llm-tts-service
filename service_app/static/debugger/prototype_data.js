const PROTOTYPE_QUESTION =
  "A student buys 5 notebooks at $3 each and 2 pens at $2 each. What is the total cost?";
const PROTOTYPE_GOLD = "19";
const PROTOTYPE_SHARED_PROMPT =
  "Reason step-by-step. Return the final answer in \\boxed{}.";

const PROTOTYPE_MODEL_CONFIG = {
  provider: "openrouter",
  model_id: "openai/gpt-4o-mini",
  api_key_masked: "sk-or...demo",
};

const PROTOTYPE_STRATEGIES = [
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

const PROTOTYPE_SCORERS = [
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

function perturbAnswer(goldAnswer, rng) {
  const text = String(goldAnswer || "").trim();
  const numericValue = Number(text);
  if (text && !Number.isNaN(numericValue)) {
    const delta = Math.max(1, Math.floor(rng() * 4) + 1);
    const sign = rng() > 0.5 ? 1 : -1;
    return String(numericValue + sign * delta);
  }
  if (!text) {
    return "unknown";
  }
  return `${text}_alt`;
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

function buildTree(strategyId, scorerId, confidence) {
  const runKey = `${strategyId}_${scorerId}`;
  return {
    nodes: [
      {
        id: `${runKey}_n1`,
        label: "Root",
        value: 0.5,
        depth: 0,
        x: 0.5,
        y: 0.14,
      },
      {
        id: `${runKey}_n2`,
        label: "A",
        value: clamp(confidence - 0.17),
        depth: 1,
        x: 0.24,
        y: 0.47,
      },
      {
        id: `${runKey}_n3`,
        label: "B",
        value: clamp(confidence - 0.08),
        depth: 1,
        x: 0.52,
        y: 0.47,
      },
      {
        id: `${runKey}_n4`,
        label: "C",
        value: clamp(confidence - 0.24),
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

function buildEvents(strategy, scorer, budget, answer, confidence, rng) {
  const wrongAnswer = perturbAnswer(PROTOTYPE_GOLD, rng);

  if (strategy.family === "single_pass") {
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
          scorerSignal(scorer, scorerValueFromConfidence(scorer, confidence)),
        ],
        candidates: [
          {
            id: `${strategy.id}_${scorer.id}_c1`,
            label: "Generated chain",
            text: `Reasoning chain outputs \\boxed{${answer}}.`,
            answer,
            status: "selected",
            selected: true,
            signals: {
              [scorer.id]: scorerValueFromConfidence(scorer, confidence),
              chain_score: confidence,
            },
          },
        ],
      },
    ];
  }

  if (strategy.family === "sample_and_vote") {
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
          value: clamp(confidence - 0.2),
          direction: "higher_better",
          threshold: 0.65,
        },
        scorerSignal(
          scorer,
          scorerValueFromConfidence(scorer, clamp(confidence - 0.1)),
        ),
      ],
      candidates: [
        {
          id: `${strategy.id}_${scorer.id}_p1`,
          label: "Path 1",
          text: `Candidate path predicts ${answer}.`,
          answer,
          status: budget > 4 ? "kept" : "selected",
          selected: budget <= 4,
          signals: {
            [scorer.id]: scorerValueFromConfidence(
              scorer,
              clamp(confidence - 0.04),
            ),
            path_score: clamp(confidence - 0.06),
          },
        },
        {
          id: `${strategy.id}_${scorer.id}_p2`,
          label: "Path 2",
          text: `Competing path predicts ${wrongAnswer}.`,
          answer: wrongAnswer,
          status: "pruned",
          selected: false,
          signals: {
            [scorer.id]: scorerValueFromConfidence(
              scorer,
              clamp(confidence - 0.2),
            ),
            path_score: clamp(confidence - 0.2),
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
        title: "Escalated sampling and final vote",
        stage: "selection",
        decision: {
          action: "stop",
          reason: "Consensus crossed threshold after additional trajectories.",
        },
        signals: [
          {
            name: "consensus",
            value: clamp(confidence - 0.02),
            direction: "higher_better",
            threshold: 0.65,
          },
          scorerSignal(scorer, scorerValueFromConfidence(scorer, confidence)),
        ],
        candidates: [
          {
            id: `${strategy.id}_${scorer.id}_p3`,
            label: "Path 3",
            text: `Escalated path verifies ${answer}.`,
            answer,
            status: "selected",
            selected: true,
            signals: {
              [scorer.id]: scorerValueFromConfidence(scorer, confidence),
              path_score: confidence,
            },
          },
        ],
      },
    ];
  }

  if (strategy.family === "reranking") {
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
          value: clamp(confidence - 0.6),
          direction: "higher_better",
          threshold: 0.08,
        },
        scorerSignal(
          scorer,
          scorerValueFromConfidence(scorer, clamp(confidence - 0.08)),
        ),
      ],
      candidates: [
        {
          id: `${strategy.id}_${scorer.id}_r1`,
          label: "Candidate 1",
          text: `High-scoring candidate predicts ${answer}.`,
          answer,
          status: budget > 4 ? "kept" : "selected",
          selected: budget <= 4,
          signals: {
            [scorer.id]: scorerValueFromConfidence(
              scorer,
              clamp(confidence - 0.03),
            ),
            rerank_score: clamp(confidence - 0.04),
          },
        },
        {
          id: `${strategy.id}_${scorer.id}_r2`,
          label: "Candidate 2",
          text: `Lower-scoring candidate predicts ${wrongAnswer}.`,
          answer: wrongAnswer,
          status: "pruned",
          selected: false,
          signals: {
            [scorer.id]: scorerValueFromConfidence(
              scorer,
              clamp(confidence - 0.2),
            ),
            rerank_score: clamp(confidence - 0.18),
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
            value: clamp(confidence - 0.48),
            direction: "higher_better",
            threshold: 0.08,
          },
          scorerSignal(scorer, scorerValueFromConfidence(scorer, confidence)),
        ],
        candidates: [
          {
            id: `${strategy.id}_${scorer.id}_r3`,
            label: "Candidate 3",
            text: `Escalated selection predicts ${answer}.`,
            answer,
            status: "selected",
            selected: true,
            signals: {
              [scorer.id]: scorerValueFromConfidence(scorer, confidence),
              rerank_score: confidence,
            },
          },
        ],
      },
    ];
  }

  return [
    {
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
          value: clamp(confidence - 0.12),
          direction: "higher_better",
          threshold: 0.75,
        },
        scorerSignal(
          scorer,
          scorerValueFromConfidence(scorer, clamp(confidence - 0.12)),
        ),
      ],
      candidates: [
        {
          id: `${strategy.id}_${scorer.id}_t1`,
          label: "Node A",
          text: `Promising branch toward ${answer}.`,
          answer: "",
          status: "kept",
          selected: true,
          signals: {
            [scorer.id]: scorerValueFromConfidence(
              scorer,
              clamp(confidence - 0.12),
            ),
            value: clamp(confidence - 0.12),
            depth: 1,
          },
        },
      ],
    },
    {
      step: 2,
      title: "Depth-2 selection",
      stage: "selection",
      decision: {
        action: "stop",
        reason: "Best branch reached solve threshold.",
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
          text: `Selected branch outputs \\boxed{${answer}}.`,
          answer,
          status: "selected",
          selected: true,
          signals: {
            [scorer.id]: scorerValueFromConfidence(scorer, confidence),
            value: confidence,
            depth: 2,
          },
        },
      ],
    },
  ];
}

function buildRun(strategy, scorer, budget) {
  const seed = hashString(`${strategy.id}|${scorer.id}|${budget}|${PROTOTYPE_QUESTION}`);
  const rng = createRng(seed);
  const difficulty = clamp(0.24 + (hashString(PROTOTYPE_QUESTION) % 30) / 100);
  const baseQuality = familyBaseQuality(strategy.family);
  const scorerShift = scorerQualityShift(scorer.id);
  const budgetFactor = budget / 12;

  const successChance = clamp(baseQuality + scorerShift + budgetFactor * 0.16 - difficulty * 0.2);
  const isCorrect = rng() < successChance;
  const answer = isCorrect ? PROTOTYPE_GOLD : perturbAnswer(PROTOTYPE_GOLD, rng);

  const quality = clamp(
    baseQuality + scorerShift + budgetFactor * 0.12 + (isCorrect ? 0.08 : -0.12),
    0.35,
    0.95,
  );
  const confidence = clamp(quality + (isCorrect ? 0.06 : -0.06), 0.1, 0.98);

  const run = {
    budget,
    budget_unit: getBudgetUnit(strategy.family),
    used_budget: strategy.family === "single_pass" ? 1 : Math.max(2, budget - 1),
    tokens_used: 660 + budget * 64 + Math.floor(rng() * 90),
    latency_ms: 5100 + budget * 540 + Math.floor(rng() * 420),
    provider: PROTOTYPE_MODEL_CONFIG.provider,
    model_id: PROTOTYPE_MODEL_CONFIG.model_id,
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
      selected_trajectory: `${strategy.id}_${scorer.id}_final`,
      confidence,
      uncertainty: clamp(1 - confidence),
      quality_score: quality,
      selection_reason: `${strategy.name} selected this trajectory under ${scorer.name} scoring (${PROTOTYPE_MODEL_CONFIG.provider}:${PROTOTYPE_MODEL_CONFIG.model_id}).`,
    },
    events: buildEvents(strategy, scorer, budget, answer, confidence, rng),
  };

  if (strategy.family === "tree_search") {
    run.tree = buildTree(strategy.id, scorer.id, confidence);
  }

  return run;
}

function buildStrategiesForBudget(budget) {
  const runs = [];
  PROTOTYPE_STRATEGIES.forEach((strategy) => {
    PROTOTYPE_SCORERS.forEach((scorer) => {
      runs.push({
        id: `${strategy.id}__${scorer.id}`,
        strategy_id: strategy.id,
        scorer_id: scorer.id,
        name: `${strategy.name} · ${scorer.name}`,
        family: strategy.family,
        summary: `${strategy.summary} Evaluated with ${scorer.name}.`,
        run: buildRun(strategy, scorer, budget),
        comparison_rank: 1,
      });
    });
  });

  const ranked = [...runs].sort(
    (left, right) => right.run.final.quality_score - left.run.final.quality_score,
  );
  const rankById = new Map(ranked.map((item, index) => [item.id, index + 1]));
  runs.forEach((item) => {
    item.comparison_rank = rankById.get(item.id) || runs.length;
  });

  return runs;
}

function buildPayload(budget) {
  const runs = buildStrategiesForBudget(budget);
  return {
    scenario: {
      id: "prototype_local_demo",
      title: "Prototype: Single-Sample Matrix",
      description:
        "Standalone example data with baseline, beam search, online/offline best-of-n, and self-consistency, each evaluated by PRM, sequence_prob, perplexity, and entropy.",
      prompt: `${PROTOTYPE_SHARED_PROMPT}\n\nQuestion: ${PROTOTYPE_QUESTION}`,
      shared_prompt: PROTOTYPE_SHARED_PROMPT,
      ground_truth: PROTOTYPE_GOLD,
      input_source: "prototype_dataset",
      model_config: PROTOTYPE_MODEL_CONFIG,
      strategy_count: PROTOTYPE_STRATEGIES.length,
      scorer_count: PROTOTYPE_SCORERS.length,
      run_count: runs.length,
    },
    available_budgets: [4, 8],
    selected_budget: budget,
    strategy_catalog: PROTOTYPE_STRATEGIES,
    scorer_catalog: PROTOTYPE_SCORERS,
    strategies: runs,
  };
}

window.VISUAL_DEBUGGER_PROTOTYPE = {
  scenarios: [
    {
      id: "prototype_local_demo",
      title: "Prototype: Single-Sample Matrix",
      description:
        "Local fallback with a full strategy x scorer matrix and model/prompt metadata.",
      available_budgets: [4, 8],
      default_budget: 8,
    },
  ],
  payloads: {
    prototype_local_demo: {
      "4": buildPayload(4),
      "8": buildPayload(8),
    },
  },
};
