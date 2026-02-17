import fs from "node:fs/promises";
import path from "node:path";

const inputPath = path.resolve("data", "training_dataset.jsonl");
const outputPath = path.resolve("data", "labeled_training_dataset.jsonl");
const summaryPath = path.resolve("data", "label_summary.json");

const HORIZONS = [
  { key: "15m", ms: 15 * 60 * 1000 },
  { key: "1h", ms: 60 * 60 * 1000 },
  { key: "3h", ms: 3 * 60 * 60 * 1000 },
];
const MAX_MODEL_WINDOW_MS = 7 * 24 * 60 * 60 * 1000;
const LATE_WINDOW_HOURS = 24;
const NEAR_RESOLUTION_HOURS = 6;
const RESOLUTION_PHASES = [
  { key: "T_7d_3d", minHours: 72, maxHours: 168 },
  { key: "T_3d_1d", minHours: 24, maxHours: 72 },
  { key: "T_24h_6h", minHours: 6, maxHours: 24 },
  { key: "T_6h_1h", minHours: 1, maxHours: 6 },
  { key: "T_1h_close", minHours: 0, maxHours: 1 },
];
const POLICY_WINDOW_HOURS_BY_PHASE = {
  T_7d_3d: 24,
  T_3d_1d: 12,
  T_24h_6h: 6,
  T_6h_1h: 1,
  T_1h_close: 0.5,
};

function safeNum(v) {
  if (v === null || v === undefined || v === "") {
    return null;
  }
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function mean(values) {
  if (values.length === 0) {
    return null;
  }
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function lowerText(v) {
  return typeof v === "string" ? v.toLowerCase() : "";
}

function classifyTaxonomy(row) {
  const primaryCategories = [
    row.leg1MarketCategory,
    row.leg2MarketCategory,
  ]
    .filter(Boolean)
    .map((x) => lowerText(x));
  const subcategories = [row.leg1MarketSubcategory, row.leg2MarketSubcategory]
    .filter(Boolean)
    .map((x) => lowerText(x));
  const tags = [...(row.leg1MarketTags ?? []), ...(row.leg2MarketTags ?? [])]
    .filter(Boolean)
    .map((x) => lowerText(x));
  const corpus = lowerText(
    [
      row.leg1MarketTitle,
      row.leg2MarketTitle,
      row.leg1EventTitle,
      row.leg2EventTitle,
      ...primaryCategories,
      ...subcategories,
      ...tags,
    ].join(" ")
  );

  const hasAny = (terms) => terms.some((term) => corpus.includes(term));
  const hasCategory = (terms) =>
    primaryCategories.some((c) => terms.some((term) => c.includes(term)));
  const hasTag = (terms) => tags.some((t) => terms.some((term) => t.includes(term)));

  const cryptoTerms = ["crypto", "bitcoin", "btc", "ethereum", "eth", "solana", "sol"];
  const politicsTerms = ["politic", "election", "president", "senate", "house", "governor"];
  const sportsTerms = ["sports", "nba", "nfl", "mlb", "nhl", "soccer", "fifa", "tennis"];
  const macroTerms = ["cpi", "inflation", "fed", "fomc", "interest rate", "gdp", "unemployment"];
  const commoditiesTerms = ["gold", "silver", "oil", "crude", "natural gas"];
  const normalizedDomainFromCategory = () => {
    const joined = primaryCategories.join(" ");
    if (joined.includes("politic")) return "politics";
    if (joined.includes("sport")) return "sports";
    if (
      joined.includes("finance") ||
      joined.includes("econom") ||
      joined.includes("crypto") ||
      joined.includes("business")
    ) {
      return "finance";
    }
    if (
      joined.includes("culture") ||
      joined.includes("entertainment") ||
      joined.includes("awards")
    ) {
      return "culture";
    }
    if (
      joined.includes("technology") ||
      joined.includes("tech") ||
      joined.includes("ai") ||
      joined.includes("science")
    ) {
      return "technology";
    }
    return null;
  };

  let domain = "other";
  let subdomain = "other";
  let topic = "unknown";
  let source = "heuristic";

  if (hasAny(cryptoTerms) || hasCategory(["crypto"]) || hasTag(["crypto"])) {
    domain = "finance";
    subdomain = "crypto";
    source = "market_metadata";
    if (hasAny(["bitcoin", "btc"])) {
      topic = "btc";
    } else if (hasAny(["ethereum", "eth"])) {
      topic = "eth";
    } else if (hasAny(["solana", "sol"])) {
      topic = "sol";
    } else {
      topic = "crypto_other";
    }
  } else if (hasAny(commoditiesTerms)) {
    domain = "finance";
    subdomain = "commodities";
    source = "market_metadata";
    if (hasAny(["gold"])) {
      topic = "gold";
    } else if (hasAny(["silver"])) {
      topic = "silver";
    } else if (hasAny(["oil", "crude"])) {
      topic = "oil";
    } else {
      topic = "commodities_other";
    }
  } else if (hasAny(macroTerms) || hasCategory(["econom", "finance"])) {
    domain = "finance";
    subdomain = "macro";
    source = "market_metadata";
    topic = hasAny(["cpi", "inflation"]) ? "inflation" : "macro_other";
  } else if (hasAny(politicsTerms) || hasCategory(["politic"])) {
    domain = "politics";
    subdomain = "elections";
    source = "market_metadata";
    topic = hasAny(["president"]) ? "presidential" : "politics_other";
  } else if (hasAny(sportsTerms) || hasCategory(["sports"])) {
    domain = "sports";
    subdomain = "general";
    source = "market_metadata";
    if (hasAny(["nba"])) {
      topic = "nba";
    } else if (hasAny(["nfl"])) {
      topic = "nfl";
    } else if (hasAny(["mlb"])) {
      topic = "mlb";
    } else {
      topic = "sports_other";
    }
  } else if (primaryCategories.length > 0 || tags.length > 0 || subcategories.length > 0) {
    domain = normalizedDomainFromCategory() ?? "other";
    subdomain = subcategories[0] ?? tags[0] ?? "other";
    topic = "metadata_other";
    source = "market_metadata";
  }

  return { domain, subdomain, topic, source };
}

function resolveCapacity(row) {
  const candidates = [
    safeNum(row.targetContractsAtDecision),
    safeNum(row.minKernelContractsAtDecision),
  ].filter((x) => x !== null && x > 0);

  if (candidates.length === 0) {
    return null;
  }
  return Math.min(...candidates);
}

function parseResolutionTs(row) {
  const candidates = [
    row.leg1ExpiresAt,
    row.leg2ExpiresAt,
    row.oppLeg1ExpiresAt,
    row.oppLeg2ExpiresAt,
  ]
    .filter(Boolean)
    .map((value) => new Date(value).getTime())
    .filter((ts) => Number.isFinite(ts));
  if (candidates.length === 0) {
    return null;
  }
  return Math.min(...candidates);
}

function resolvePhaseKey(ttrHours) {
  if (ttrHours === null) {
    return null;
  }
  const phase = RESOLUTION_PHASES.find(
    (p) => ttrHours >= p.minHours && ttrHours < p.maxHours
  );
  return phase?.key ?? null;
}

function resolvePolicyWindowHours(phaseKey, ttrHoursNow) {
  if (phaseKey && POLICY_WINDOW_HOURS_BY_PHASE[phaseKey] !== undefined) {
    return POLICY_WINDOW_HOURS_BY_PHASE[phaseKey];
  }
  if (ttrHoursNow === null) {
    return 24;
  }
  return Math.max(0.5, Math.min(24, ttrHoursNow / 4));
}

function futureRowsWithinHorizon(seq, idx, horizonMs, resolutionTs) {
  const t0 = new Date(seq[idx].decisionTs).getTime();
  const absoluteMaxTs =
    resolutionTs === null
      ? t0 + Math.min(horizonMs, MAX_MODEL_WINDOW_MS)
      : Math.min(t0 + horizonMs, t0 + MAX_MODEL_WINDOW_MS, resolutionTs);
  const out = [];
  for (let j = idx + 1; j < seq.length; j++) {
    const tj = new Date(seq[j].decisionTs).getTime();
    if (tj > absoluteMaxTs) {
      break;
    }
    out.push(seq[j]);
  }
  return out;
}

function buildResolutionAnchoredLabels(seq, idx, resolutionTs) {
  const now = seq[idx];
  const nowEdge = safeNum(now.expectedEdgeAtDecision);
  const nowCapacity = resolveCapacity(now);
  const nowTsMs = new Date(now.decisionTs).getTime();
  const windowEndTs =
    resolutionTs === null
      ? nowTsMs + MAX_MODEL_WINDOW_MS
      : Math.min(resolutionTs, nowTsMs + MAX_MODEL_WINDOW_MS);
  const candidates = [];
  for (let j = idx + 1; j < seq.length; j++) {
    const tj = new Date(seq[j].decisionTs).getTime();
    if (tj > windowEndTs) {
      break;
    }
    candidates.push(seq[j]);
  }
  if (candidates.length === 0 || nowEdge === null) {
    return null;
  }

  const ttrHoursNow =
    resolutionTs === null ? null : (resolutionTs - nowTsMs) / (60 * 60 * 1000);
  const phaseNow = resolvePhaseKey(ttrHoursNow);
  const policyWindowHours = resolvePolicyWindowHours(phaseNow, ttrHoursNow);
  const policyWindowEndTs = Math.min(windowEndTs, nowTsMs + policyWindowHours * 60 * 60 * 1000);

  let bestEdge7d = null;
  let bestEdgeTs = null;
  let bestCapAdj7d = null;
  let bestCapAdjTs = null;
  let bestFillRatio7d = null;
  let bestFutureCapacity7d = null;

  let bestLateCapAdj = null;
  let bestLateFillRatio = null;
  let bestNearResCapAdj = null;
  let bestPolicyWindowCapAdj = null;
  let bestPolicyWindowFillRatio = null;
  let bestPolicyWindowTs = null;

  for (const c of candidates) {
    const cEdge = safeNum(c.expectedEdgeAtDecision);
    const cCap = resolveCapacity(c);
    const cTsMs = new Date(c.decisionTs).getTime();
    const cTtrHours =
      resolutionTs === null ? null : (resolutionTs - cTsMs) / (60 * 60 * 1000);

    if (cEdge === null) {
      continue;
    }
    if (bestEdge7d === null || cEdge > bestEdge7d) {
      bestEdge7d = cEdge;
      bestEdgeTs = c.decisionTs;
    }
    if (cCap !== null) {
      if (bestFutureCapacity7d === null || cCap > bestFutureCapacity7d) {
        bestFutureCapacity7d = cCap;
      }
    }

    if (nowCapacity !== null && nowCapacity > 0 && cCap !== null) {
      const fillable = Math.min(nowCapacity, cCap);
      const fillRatio = fillable / nowCapacity;
      const currentProfit = nowEdge * nowCapacity;
      const futureProfit = cEdge * fillable;
      const uplift = futureProfit - currentProfit;

      if (bestCapAdj7d === null || uplift > bestCapAdj7d) {
        bestCapAdj7d = uplift;
        bestCapAdjTs = c.decisionTs;
        bestFillRatio7d = fillRatio;
      }

      if (cTtrHours !== null && cTtrHours <= LATE_WINDOW_HOURS) {
        if (bestLateCapAdj === null || uplift > bestLateCapAdj) {
          bestLateCapAdj = uplift;
          bestLateFillRatio = fillRatio;
        }
      }

      if (cTtrHours !== null && cTtrHours <= NEAR_RESOLUTION_HOURS) {
        if (bestNearResCapAdj === null || uplift > bestNearResCapAdj) {
          bestNearResCapAdj = uplift;
        }
      }

      if (cTsMs <= policyWindowEndTs) {
        if (bestPolicyWindowCapAdj === null || uplift > bestPolicyWindowCapAdj) {
          bestPolicyWindowCapAdj = uplift;
          bestPolicyWindowFillRatio = fillRatio;
          bestPolicyWindowTs = c.decisionTs;
        }
      }
    }
  }

  if (bestEdge7d === null) {
    return null;
  }

  const edgeUplift7d = bestEdge7d - nowEdge;
  const minutesToBestEdge7d =
    bestEdgeTs === null ? null : (new Date(bestEdgeTs).getTime() - nowTsMs) / (60 * 1000);
  const minutesToBestCapAdj7d =
    bestCapAdjTs === null
      ? null
      : (new Date(bestCapAdjTs).getTime() - nowTsMs) / (60 * 1000);
  const capacityChange7d =
    nowCapacity !== null && bestFutureCapacity7d !== null
      ? bestFutureCapacity7d - nowCapacity
      : null;
  const minutesToBestPolicyWindowCapAdj =
    bestPolicyWindowTs === null
      ? null
      : (new Date(bestPolicyWindowTs).getTime() - nowTsMs) / (60 * 1000);

  return {
    timeToResolutionHoursNow: ttrHoursNow,
    phaseNow,
    edgeUplift7d,
    improvesEdge7d: edgeUplift7d > 0,
    bestFutureEdge7d: bestEdge7d,
    minutesToBestEdge7d,
    nowCapacityContracts: nowCapacity,
    bestFutureCapacityContracts7d: bestFutureCapacity7d,
    capacityChangeContracts7d: capacityChange7d,
    capacityAdjustedUplift7d: bestCapAdj7d,
    improvesCapacityAdjusted7d: bestCapAdj7d !== null ? bestCapAdj7d > 0 : null,
    minutesToBestCapacityAdjusted7d: minutesToBestCapAdj7d,
    bestFillRatioAtNowSize7d: bestFillRatio7d,
    bestLateWindowCapacityAdjustedUplift: bestLateCapAdj,
    bestLateWindowFillRatioAtNowSize: bestLateFillRatio,
    enterEarlyBetterThanLate:
      bestLateCapAdj === null ? null : bestLateCapAdj <= 0,
    nearResolutionCapacityAdjustedUplift: bestNearResCapAdj,
    policyWindowHours,
    deltaNetPnlPolicyWindowAtNowSize: bestPolicyWindowCapAdj,
    buyNowBeatsWaitWindow:
      bestPolicyWindowCapAdj === null ? null : bestPolicyWindowCapAdj <= 0,
    bestWaitFillRatioPolicyWindowAtNowSize: bestPolicyWindowFillRatio,
    minutesToBestPolicyWindow: minutesToBestPolicyWindowCapAdj,
    labelCensoredPolicyWindow: bestPolicyWindowCapAdj === null,
  };
}

function buildLabelsForRow(seq, idx) {
  const now = seq[idx];
  const taxonomy = classifyTaxonomy(now);
  const resolutionTs = parseResolutionTs(now);
  const nowEdge = safeNum(now.expectedEdgeAtDecision);
  const nowCapacity = resolveCapacity(now);
  const nowTsMs = new Date(now.decisionTs).getTime();
  const labels = {};

  for (const horizon of HORIZONS) {
    const candidates = futureRowsWithinHorizon(seq, idx, horizon.ms, resolutionTs);
    if (candidates.length === 0 || nowEdge === null) {
      labels[horizon.key] = null;
      continue;
    }

    let bestEdge = null;
    let bestEdgeTs = null;
    let bestCapacityAdjustedUplift = null;
    let bestCapacityAdjustedTs = null;
    let bestFillRatio = null;
    let bestFutureCapacity = null;

    for (const c of candidates) {
      const cEdge = safeNum(c.expectedEdgeAtDecision);
      const cCapacity = resolveCapacity(c);
      if (cEdge === null) {
        continue;
      }

      if (bestEdge === null || cEdge > bestEdge) {
        bestEdge = cEdge;
        bestEdgeTs = c.decisionTs;
      }

      if (cCapacity !== null) {
        if (bestFutureCapacity === null || cCapacity > bestFutureCapacity) {
          bestFutureCapacity = cCapacity;
        }
      }

      if (nowCapacity !== null && nowCapacity > 0 && cCapacity !== null) {
        const fillable = Math.min(nowCapacity, cCapacity);
        const fillRatio = fillable / nowCapacity;
        const currentProfitAtNowSize = nowEdge * nowCapacity;
        const futureProfitAtNowSize = cEdge * fillable;
        const uplift = futureProfitAtNowSize - currentProfitAtNowSize;

        if (
          bestCapacityAdjustedUplift === null ||
          uplift > bestCapacityAdjustedUplift
        ) {
          bestCapacityAdjustedUplift = uplift;
          bestCapacityAdjustedTs = c.decisionTs;
          bestFillRatio = fillRatio;
        }
      }
    }

    if (bestEdge === null) {
      labels[horizon.key] = null;
      continue;
    }

    const edgeUplift = bestEdge - nowEdge;
    const minutesToBestEdge =
      bestEdgeTs === null
        ? null
        : (new Date(bestEdgeTs).getTime() - nowTsMs) / (60 * 1000);

    const capacityChange =
      nowCapacity !== null && bestFutureCapacity !== null
        ? bestFutureCapacity - nowCapacity
        : null;

    labels[horizon.key] = {
      edgeUplift,
      improvesEdge: edgeUplift > 0,
      bestFutureEdge: bestEdge,
      minutesToBestEdge,
      nowCapacityContracts: nowCapacity,
      bestFutureCapacityContracts: bestFutureCapacity,
      capacityChangeContracts: capacityChange,
      capacityAdjustedUplift: bestCapacityAdjustedUplift,
      improvesCapacityAdjusted:
        bestCapacityAdjustedUplift !== null ? bestCapacityAdjustedUplift > 0 : null,
      bestFillRatioAtNowSize: bestFillRatio,
      minutesToBestCapacityAdjusted:
        bestCapacityAdjustedTs === null
          ? null
          : (new Date(bestCapacityAdjustedTs).getTime() - nowTsMs) / (60 * 1000),
    };
  }

  labels.resolutionAnchored = buildResolutionAnchoredLabels(seq, idx, resolutionTs);
  labels.taxonomy = taxonomy;
  return labels;
}

const raw = await fs.readFile(inputPath, "utf8");
const rows = raw
  .split("\n")
  .map((line) => line.trim())
  .filter(Boolean)
  .map((line) => JSON.parse(line));

const grouped = new Map();
for (const row of rows) {
  if (!row.dedupeKey || !row.decisionTs || row.expectedEdgeAtDecision === null) {
    continue;
  }
  if (!grouped.has(row.dedupeKey)) {
    grouped.set(row.dedupeKey, []);
  }
  grouped.get(row.dedupeKey).push(row);
}

for (const seq of grouped.values()) {
  seq.sort((a, b) => new Date(a.decisionTs) - new Date(b.decisionTs));
}

const labeledRows = [];
for (const seq of grouped.values()) {
  for (let i = 0; i < seq.length; i++) {
    labeledRows.push({
      ...seq[i],
      labels: buildLabelsForRow(seq, i),
    });
  }
}

const lines = labeledRows.map((r) => JSON.stringify(r)).join("\n");
await fs.writeFile(outputPath, `${lines}\n`, "utf8");

const summary = {
  generatedAt: new Date().toISOString(),
  totalInputRows: rows.length,
  totalGroupedRows: labeledRows.length,
  dedupeSeries: grouped.size,
  horizons: {},
  phases: {},
  taxonomy: {},
};

for (const horizon of HORIZONS) {
  const key = horizon.key;
  const horizonLabels = labeledRows
    .map((r) => r.labels?.[key])
    .filter((l) => l !== null);
  const edgeUplifts = horizonLabels
    .map((l) => l.edgeUplift)
    .filter((x) => x !== null && Number.isFinite(x));
  const capAdjUplifts = horizonLabels
    .map((l) => l.capacityAdjustedUplift)
    .filter((x) => x !== null && Number.isFinite(x));
  const edgeImprove = horizonLabels
    .map((l) => (l.improvesEdge ? 1 : 0));
  const capAdjImprove = horizonLabels
    .map((l) =>
      l.improvesCapacityAdjusted === null ? null : l.improvesCapacityAdjusted ? 1 : 0
    )
    .filter((x) => x !== null);
  const fillRatios = horizonLabels
    .map((l) => l.bestFillRatioAtNowSize)
    .filter((x) => x !== null && Number.isFinite(x));

  summary.horizons[key] = {
    labelCount: horizonLabels.length,
    meanEdgeUplift: mean(edgeUplifts),
    meanCapacityAdjustedUplift: mean(capAdjUplifts),
    probImprovesEdge: mean(edgeImprove),
    probImprovesCapacityAdjusted: mean(capAdjImprove),
    meanBestFillRatioAtNowSize: mean(fillRatios),
  };
}

const phaseLabels = labeledRows
  .map((r) => r.labels?.resolutionAnchored)
  .filter((l) => l !== null);

for (const phase of RESOLUTION_PHASES) {
  const rowsInPhase = phaseLabels.filter((l) => l.phaseNow === phase.key);
  const capAdj7d = rowsInPhase
    .map((l) => l.capacityAdjustedUplift7d)
    .filter((x) => x !== null && Number.isFinite(x));
  const earlyBetter = rowsInPhase
    .map((l) =>
      l.enterEarlyBetterThanLate === null ? null : l.enterEarlyBetterThanLate ? 1 : 0
    )
    .filter((x) => x !== null);

  summary.phases[phase.key] = {
    rowCount: rowsInPhase.length,
    meanCapacityAdjustedUplift7d: mean(capAdj7d),
    probEnterEarlyBetterThanLate: mean(earlyBetter),
    policyWindowLabelCoverage:
      rowsInPhase.length > 0
        ? rowsInPhase.filter((l) => l.deltaNetPnlPolicyWindowAtNowSize !== null).length /
          rowsInPhase.length
        : null,
  };
}

const taxonomyLabels = labeledRows
  .map((r) => r.labels?.taxonomy)
  .filter((t) => t && t.domain);

for (const domain of [...new Set(taxonomyLabels.map((t) => t.domain))]) {
  const inDomain = labeledRows.filter((r) => r.labels?.taxonomy?.domain === domain);
  const res = inDomain
    .map((r) => r.labels?.resolutionAnchored)
    .filter((l) => l !== null);
  const deltaPolicy = res
    .map((l) => l.deltaNetPnlPolicyWindowAtNowSize)
    .filter((x) => x !== null && Number.isFinite(x));
  const buyNowBeats = res
    .map((l) => (l.buyNowBeatsWaitWindow === null ? null : l.buyNowBeatsWaitWindow ? 1 : 0))
    .filter((x) => x !== null);

  summary.taxonomy[domain] = {
    rowCount: inDomain.length,
    resolutionLabelCount: res.length,
    meanDeltaNetPnlPolicyWindowAtNowSize: mean(deltaPolicy),
    probBuyNowBeatsWaitWindow: mean(buyNowBeats),
  };
}

await fs.writeFile(summaryPath, `${JSON.stringify(summary, null, 2)}\n`, "utf8");
console.log(`Wrote labeled dataset: ${outputPath}`);
console.log(`Wrote label summary: ${summaryPath}`);
console.log(JSON.stringify(summary, null, 2));
