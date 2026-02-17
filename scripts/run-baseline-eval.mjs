import fs from "node:fs/promises";
import path from "node:path";

const inputPath = path.resolve("data", "labeled_training_dataset.jsonl");
const reportPath = path.resolve("data", "baseline_report.json");
const PHASE_KEYS = ["T_7d_3d", "T_3d_1d", "T_24h_6h", "T_6h_1h", "T_1h_close"];

function mean(values) {
  if (values.length === 0) {
    return null;
  }
  return values.reduce((a, b) => a + b, 0) / values.length;
}

const raw = await fs.readFile(inputPath, "utf8");
const rows = raw
  .split("\n")
  .map((line) => line.trim())
  .filter(Boolean)
  .map((line) => JSON.parse(line));

const horizon15m = rows.map((r) => r.labels?.["15m"]).filter(Boolean);
const horizon1h = rows.map((r) => r.labels?.["1h"]).filter(Boolean);
const horizon3h = rows.map((r) => r.labels?.["3h"]).filter(Boolean);
const resolutionAnchored = rows
  .map((r) => r.labels?.resolutionAnchored)
  .filter(Boolean);

function summarizeHorizon(labels) {
  const capAdjUplift = labels
    .map((l) => l.capacityAdjustedUplift)
    .filter((x) => x !== null && Number.isFinite(x));
  const improveCapAdj = labels
    .map((l) =>
      l.improvesCapacityAdjusted === null ? null : l.improvesCapacityAdjusted ? 1 : 0
    )
    .filter((x) => x !== null);
  return {
    labelCount: labels.length,
    meanCapacityAdjustedUplift: mean(capAdjUplift),
    probImprovesCapacityAdjusted: mean(improveCapAdj),
  };
}

const byPhase = {};
for (const phase of PHASE_KEYS) {
  const labels = resolutionAnchored.filter((l) => l.phaseNow === phase);
  const capAdj7d = labels
    .map((l) => l.capacityAdjustedUplift7d)
    .filter((x) => x !== null && Number.isFinite(x));
  const earlyBetter = labels
    .map((l) =>
      l.enterEarlyBetterThanLate === null ? null : l.enterEarlyBetterThanLate ? 1 : 0
    )
    .filter((x) => x !== null);
  const policyDelta = labels
    .map((l) => l.deltaNetPnlPolicyWindowAtNowSize)
    .filter((x) => x !== null && Number.isFinite(x));
  const buyNowBeats = labels
    .map((l) => (l.buyNowBeatsWaitWindow === null ? null : l.buyNowBeatsWaitWindow ? 1 : 0))
    .filter((x) => x !== null);
  const censored = labels
    .map((l) => (l.labelCensoredPolicyWindow ? 1 : 0))
    .filter((x) => x !== null);
  byPhase[phase] = {
    labelCount: labels.length,
    meanCapacityAdjustedUplift7d: mean(capAdj7d),
    probEnterEarlyBetterThanLate: mean(earlyBetter),
    policyWindowLabelCount: policyDelta.length,
    meanDeltaNetPnlPolicyWindowAtNowSize: mean(policyDelta),
    probBuyNowBeatsWaitWindow: mean(buyNowBeats),
    censoringRatePolicyWindow: mean(censored),
  };
}

const taxonomyRows = rows
  .map((r) => ({
    taxonomy: r.labels?.taxonomy,
    resolutionAnchored: r.labels?.resolutionAnchored,
  }))
  .filter((r) => r.taxonomy?.domain);

const domains = [...new Set(taxonomyRows.map((r) => r.taxonomy.domain))];
const byTaxonomyDomain = {};
for (const domain of domains) {
  const inDomain = taxonomyRows.filter((r) => r.taxonomy.domain === domain);
  const res = inDomain.map((r) => r.resolutionAnchored).filter(Boolean);
  const policyDelta = res
    .map((l) => l.deltaNetPnlPolicyWindowAtNowSize)
    .filter((x) => x !== null && Number.isFinite(x));
  const buyNowBeats = res
    .map((l) => (l.buyNowBeatsWaitWindow === null ? null : l.buyNowBeatsWaitWindow ? 1 : 0))
    .filter((x) => x !== null);
  byTaxonomyDomain[domain] = {
    rowCount: inDomain.length,
    resolutionAnchoredCount: res.length,
    policyWindowLabelCount: policyDelta.length,
    meanDeltaNetPnlPolicyWindowAtNowSize: mean(policyDelta),
    probBuyNowBeatsWaitWindow: mean(buyNowBeats),
  };
}

const report = {
  generatedAt: new Date().toISOString(),
  sampleSize: rows.length,
  metrics: {
    horizon15m: summarizeHorizon(horizon15m),
    horizon1h: summarizeHorizon(horizon1h),
    horizon3h: summarizeHorizon(horizon3h),
    resolutionAnchoredCount: resolutionAnchored.length,
    byResolutionPhase: byPhase,
    byTaxonomyDomain,
  },
  notes: [
    "This baseline reads from labeled_training_dataset.jsonl.",
    "Resolution-anchored labels are capped to a 7-day lookahead window for the current 30-day dataset.",
    "Primary policy-aligned targets are deltaNetPnlPolicyWindowAtNowSize (regression) and buyNowBeatsWaitWindow (classification).",
  ],
};

await fs.writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
console.log(`Baseline report written: ${reportPath}`);
console.log(JSON.stringify(report, null, 2));
