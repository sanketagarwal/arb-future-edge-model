import fs from "node:fs/promises";
import path from "node:path";

const dataPath = path.resolve("data", "labeled_training_dataset.jsonl");
const artifactsPath = path.resolve("data", "model_robust_artifacts.json");
const outputPath = path.resolve("data", "model_robust_backtest_report.json");

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
function clip(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
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
    out.push({
      ...row,
      _decisionTsMs: ts,
      _regTarget: safeNum(res.deltaNetPnlPolicyWindowAtNowSize),
      _censored: Boolean(res.labelCensoredPolicyWindow),
      _phaseNow: res.phaseNow ?? "unknown",
      _ttrHours: safeNum(res.timeToResolutionHoursNow),
      _policyWindowHours: safeNum(res.policyWindowHours),
      _taxonomyDomain: tax.domain ?? "other",
      _segmentKey: `${tax.domain ?? "other"}::${res.phaseNow ?? "unknown"}`,
    });
  }
  out.sort((a, b) => a._decisionTsMs - b._decisionTsMs);
  return out;
}
function splitChronological(rows, trainFrac = 0.7, validFrac = 0.15) {
  const n = rows.length;
  const trainEnd = Math.max(1, Math.floor(n * trainFrac));
  const validEnd = Math.max(trainEnd + 1, Math.floor(n * (trainFrac + validFrac)));
  return { train: rows.slice(0, trainEnd), valid: rows.slice(trainEnd, validEnd), test: rows.slice(validEnd) };
}
function buildVector(row, schema, scaler) {
  const x = [1];
  for (const key of schema.numericKeys) {
    const v = safeNum(row[key]);
    const s = scaler[key];
    x.push(v === null ? 0 : (v - s.mean) / s.std);
  }
  x.push(...oneHotValue(row._phaseNow, schema.phaseBuckets));
  x.push(...oneHotValue(row._taxonomyDomain, schema.domainBuckets));
  x.push(...oneHotValue(row.leg1Venue ?? "unknown", schema.leg1VenueBuckets));
  x.push(...oneHotValue(row.leg2Venue ?? "unknown", schema.leg2VenueBuckets));
  x.push(...oneHotValue(row.leg1OrderIntent ?? "unknown", schema.leg1IntentBuckets));
  x.push(...oneHotValue(row.leg2OrderIntent ?? "unknown", schema.leg2IntentBuckets));
  return x;
}

function evaluate(rows, probs, threshold) {
  const decisions = probs.map((p) => (p >= threshold ? "buy_now" : "wait"));
  const rewards = rows.map((r, i) => (decisions[i] === "buy_now" ? 0 : r._regTarget));
  const oracle = rows.map((r) => Math.max(0, r._regTarget));
  return {
    threshold,
    rows: rows.length,
    buyRate: mean(decisions.map((d) => (d === "buy_now" ? 1 : 0))),
    waitRate: mean(decisions.map((d) => (d === "wait" ? 1 : 0))),
    meanRelativePnlVsAlwaysBuyNow: mean(rewards),
    totalRelativePnlVsAlwaysBuyNow: rewards.reduce((a, b) => a + b, 0),
    meanOracleRelativePnlVsAlwaysBuyNow: mean(oracle),
    totalOracleRelativePnlVsAlwaysBuyNow: oracle.reduce((a, b) => a + b, 0),
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
const threshold = artifacts.classification.tunedThreshold;
const winsor = artifacts.winsorization;

function predictLogit(row) {
  const x = buildVector(row, schema, scaler);
  const seg = artifacts.classification.segments[row._segmentKey];
  const w = seg ? seg.weights : artifacts.classification.globalWeights;
  return dot(w, x);
}
function calibratedProb(logit) {
  const { a, b } = artifacts.classification.platt;
  return sigmoid(a * logit + b);
}
function predictReg(row) {
  const x = buildVector(row, schema, scaler);
  const seg = artifacts.regression.segments[row._segmentKey];
  if (seg) return clip(dot(seg.weights, x), seg.lo, seg.hi);
  return clip(dot(artifacts.regression.globalWeights, x), winsor.low, winsor.high);
}

const validRows = split.valid.filter((r) => !r._censored && r._regTarget !== null);
const testRows = split.test.filter((r) => !r._censored && r._regTarget !== null);

const validProbs = validRows.map((r) => calibratedProb(predictLogit(r)));
const testProbs = testRows.map((r) => calibratedProb(predictLogit(r)));
const validEval = evaluate(validRows, validProbs, threshold);
const testEval = evaluate(testRows, testProbs, threshold);

const regPredTest = testRows.map((r) => predictReg(r));
const regErrors = testRows.map((r, i) => regPredTest[i] - r._regTarget);
const regMAE = mean(regErrors.map((e) => Math.abs(e)));
const regRMSE = Math.sqrt(mean(regErrors.map((e) => e * e)));

const report = {
  generatedAt: new Date().toISOString(),
  splitSizes: {
    allPreparedRows: prepared.length,
    validationRowsUsed: validRows.length,
    testRowsUsed: testRows.length,
  },
  tunedThresholdFromValidation: threshold,
  validationAtTunedThreshold: validEval,
  testAtTunedThreshold: testEval,
  regressionSanityOnTest: {
    mae: regMAE,
    rmse: regRMSE,
  },
};

await fs.writeFile(outputPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
console.log(`Wrote robust backtest report: ${outputPath}`);
console.log(JSON.stringify(report, null, 2));
