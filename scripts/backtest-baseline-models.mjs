import fs from "node:fs/promises";
import path from "node:path";

const dataPath = path.resolve("data", "labeled_training_dataset.jsonl");
const artifactsPath = path.resolve("data", "model_baseline_artifacts.json");
const outputPath = path.resolve("data", "model_backtest_report.json");

function safeNum(v) {
  if (v === null || v === undefined || v === "") return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function mean(values) {
  if (values.length === 0) return null;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function sigmoid(z) {
  if (z >= 0) {
    const ez = Math.exp(-z);
    return 1 / (1 + ez);
  }
  const ez = Math.exp(z);
  return ez / (1 + ez);
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function oneHotValue(value, buckets) {
  const out = new Array(buckets.length).fill(0);
  const idx = buckets.indexOf(value);
  if (idx >= 0) out[idx] = 1;
  return out;
}

function prepareRows(rows) {
  const out = [];
  for (const row of rows) {
    const res = row.labels?.resolutionAnchored;
    const tax = row.labels?.taxonomy;
    if (!res || !tax) continue;
    const ts = new Date(row.decisionTs).getTime();
    if (!Number.isFinite(ts)) continue;
    const delta = safeNum(res.deltaNetPnlPolicyWindowAtNowSize);
    const cls =
      res.buyNowBeatsWaitWindow === null || res.buyNowBeatsWaitWindow === undefined
        ? null
        : res.buyNowBeatsWaitWindow
          ? 1
          : 0;
    out.push({
      ...row,
      _decisionTsMs: ts,
      _regTarget: delta,
      _clsTarget: cls,
      _censored: Boolean(res.labelCensoredPolicyWindow),
      _phaseNow: res.phaseNow ?? "unknown",
      _ttrHours: safeNum(res.timeToResolutionHoursNow),
      _policyWindowHours: safeNum(res.policyWindowHours),
      _taxonomyDomain: tax.domain ?? "other",
    });
  }
  out.sort((a, b) => a._decisionTsMs - b._decisionTsMs);
  return out;
}

function splitChronological(rows, trainFrac = 0.7, validFrac = 0.15) {
  const n = rows.length;
  const trainEnd = Math.max(1, Math.floor(n * trainFrac));
  const validEnd = Math.max(trainEnd + 1, Math.floor(n * (trainFrac + validFrac)));
  return {
    train: rows.slice(0, trainEnd),
    valid: rows.slice(trainEnd, validEnd),
    test: rows.slice(validEnd),
  };
}

function buildVector(row, schema, scaler) {
  const x = [1];
  for (const key of schema.numericKeys) {
    const v = safeNum(row[key]);
    const s = scaler[key];
    const z = v === null ? 0 : (v - s.mean) / s.std;
    x.push(z);
  }
  x.push(...oneHotValue(row._phaseNow, schema.phaseBuckets));
  x.push(...oneHotValue(row._taxonomyDomain, schema.domainBuckets));
  x.push(...oneHotValue(row.leg1Venue ?? "unknown", schema.leg1VenueBuckets));
  x.push(...oneHotValue(row.leg2Venue ?? "unknown", schema.leg2VenueBuckets));
  x.push(...oneHotValue(row.leg1OrderIntent ?? "unknown", schema.leg1IntentBuckets));
  x.push(...oneHotValue(row.leg2OrderIntent ?? "unknown", schema.leg2IntentBuckets));
  return x;
}

function evaluateDecision(rows, probs, threshold) {
  const decisions = rows.map((_, i) => (probs[i] >= threshold ? "buy_now" : "wait"));
  const rewards = rows.map((r, i) => {
    const delta = r._regTarget;
    if (delta === null) return null;
    // delta = best_wait_pnl - buy_now_pnl at now-size.
    // reward here is policy pnl relative to always-buy-now baseline.
    return decisions[i] === "buy_now" ? 0 : delta;
  });
  const oracleRewards = rows.map((r) => {
    if (r._regTarget === null) return null;
    return Math.max(0, r._regTarget);
  });
  const validRewards = rewards.filter((x) => x !== null);
  const validOracle = oracleRewards.filter((x) => x !== null);
  const buyRate = mean(decisions.map((d) => (d === "buy_now" ? 1 : 0)));
  const waitRate = mean(decisions.map((d) => (d === "wait" ? 1 : 0)));
  return {
    threshold,
    rows: rows.length,
    buyRate,
    waitRate,
    meanRelativePnlVsAlwaysBuyNow: mean(validRewards),
    totalRelativePnlVsAlwaysBuyNow: validRewards.reduce((a, b) => a + b, 0),
    meanOracleRelativePnlVsAlwaysBuyNow: mean(validOracle),
    totalOracleRelativePnlVsAlwaysBuyNow: validOracle.reduce((a, b) => a + b, 0),
  };
}

const raw = await fs.readFile(dataPath, "utf8");
const rows = raw
  .split("\n")
  .map((line) => line.trim())
  .filter(Boolean)
  .map((line) => JSON.parse(line));
const prepared = prepareRows(rows);
const split = splitChronological(prepared);

const artifacts = JSON.parse(await fs.readFile(artifactsPath, "utf8"));
const schema = artifacts.schema;
const scaler = artifacts.scaler;
const clsW = artifacts.classification.weights;

const testRows = split.test.filter((r) => !r._censored && r._regTarget !== null);
const testX = testRows.map((r) => buildVector(r, schema, scaler));
const testProbs = testX.map((x) => sigmoid(dot(clsW, x)));

const thresholds = [0.4, 0.5, 0.6, 0.7];
const results = thresholds.map((t) => evaluateDecision(testRows, testProbs, t));
const best = [...results].sort(
  (a, b) => b.meanRelativePnlVsAlwaysBuyNow - a.meanRelativePnlVsAlwaysBuyNow
)[0];

const report = {
  generatedAt: new Date().toISOString(),
  splitSizes: {
    allPreparedRows: prepared.length,
    testRows: split.test.length,
    backtestRowsUsed: testRows.length,
  },
  strategy: "classification-gated wait/buy-now decision",
  assumptions: [
    "Reward is measured relative to always-buy-now baseline.",
    "Waiting reward uses observed policy-window deltaNetPnlPolicyWindowAtNowSize label.",
    "No transaction-cost drift beyond what is embedded in labels.",
  ],
  thresholdSweep: results,
  bestThresholdByMeanRelativePnl: best,
};

await fs.writeFile(outputPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
console.log(`Wrote backtest report: ${outputPath}`);
console.log(JSON.stringify(report, null, 2));
