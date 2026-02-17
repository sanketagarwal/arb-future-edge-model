import fs from "node:fs/promises";
import path from "node:path";

const inputPath = path.resolve("data", "labeled_training_dataset.jsonl");
const reportPath = path.resolve("data", "model_robust_report.json");
const artifactsPath = path.resolve("data", "model_robust_artifacts.json");

const DOMAIN_KEYS = ["politics", "sports", "finance", "technology", "culture", "other"];
const MIN_SEGMENT_ROWS = 120;

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
function quantile(values, q) {
  if (values.length === 0) return null;
  const s = [...values].sort((a, b) => a - b);
  const idx = (s.length - 1) * q;
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return s[lo];
  const frac = idx - lo;
  return s[lo] * (1 - frac) + s[hi] * frac;
}
function aucRoc(yTrue, yScore) {
  if (yTrue.length === 0 || yScore.length !== yTrue.length) return null;
  const pairs = yTrue.map((y, i) => ({ y, s: yScore[i] }));
  const pos = pairs.filter((p) => p.y === 1).length;
  const neg = pairs.filter((p) => p.y === 0).length;
  if (pos === 0 || neg === 0) return null;
  pairs.sort((a, b) => a.s - b.s);
  let rankSumPos = 0;
  for (let i = 0; i < pairs.length; i++) if (pairs[i].y === 1) rankSumPos += i + 1;
  return (rankSumPos - (pos * (pos + 1)) / 2) / (pos * neg);
}

function classificationMetrics(yTrue, yProb, threshold = 0.5) {
  const yPred = yProb.map((p) => (p >= threshold ? 1 : 0));
  let tp = 0;
  let fp = 0;
  let tn = 0;
  let fn = 0;
  for (let i = 0; i < yTrue.length; i++) {
    const t = yTrue[i];
    const p = yPred[i];
    if (t === 1 && p === 1) tp++;
    if (t === 0 && p === 1) fp++;
    if (t === 0 && p === 0) tn++;
    if (t === 1 && p === 0) fn++;
  }
  const precision = tp + fp === 0 ? null : tp / (tp + fp);
  const recall = tp + fn === 0 ? null : tp / (tp + fn);
  const f1 =
    precision === null || recall === null || precision + recall === 0
      ? null
      : (2 * precision * recall) / (precision + recall);
  return {
    count: yTrue.length,
    accuracy: yTrue.length ? (tp + tn) / yTrue.length : null,
    precision,
    recall,
    f1,
    auc: aucRoc(yTrue, yProb),
  };
}
function regressionMetrics(yTrue, yPred) {
  if (yTrue.length === 0 || yPred.length !== yTrue.length) {
    return { count: 0, mae: null, rmse: null, r2: null, signAccuracy: null };
  }
  const errors = yTrue.map((y, i) => yPred[i] - y);
  const mae = mean(errors.map((e) => Math.abs(e)));
  const rmse = Math.sqrt(mean(errors.map((e) => e * e)));
  const yMean = mean(yTrue);
  const sst = yTrue.reduce((acc, y) => acc + (y - yMean) ** 2, 0);
  const sse = errors.reduce((acc, e) => acc + e * e, 0);
  const signAccuracy = mean(yTrue.map((y, i) => (Math.sign(y) === Math.sign(yPred[i]) ? 1 : 0)));
  return { count: yTrue.length, mae, rmse, r2: sst === 0 ? null : 1 - sse / sst, signAccuracy };
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
    const cls =
      res.buyNowBeatsWaitWindow === null || res.buyNowBeatsWaitWindow === undefined
        ? null
        : res.buyNowBeatsWaitWindow
          ? 1
          : 0;
    out.push({
      ...row,
      _decisionTsMs: ts,
      _regTarget: safeNum(res.deltaNetPnlPolicyWindowAtNowSize),
      _clsTarget: cls,
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

function buildSchema(trainRows) {
  return {
    numericKeys: [
      "expectedEdgeAtDecision",
      "targetContractsAtDecision",
      "minKernelContractsAtDecision",
      "avgLegPriceAtDecision",
      "budgetUsdAtDecision",
      "requestUsd",
      "availableUsd",
      "legCount",
      "_ttrHours",
      "_policyWindowHours",
    ],
    phaseBuckets: [...new Set(trainRows.map((r) => r._phaseNow))].sort(),
    domainBuckets: [...new Set(trainRows.map((r) => r._taxonomyDomain))].sort(),
    leg1VenueBuckets: [...new Set(trainRows.map((r) => r.leg1Venue ?? "unknown"))].sort(),
    leg2VenueBuckets: [...new Set(trainRows.map((r) => r.leg2Venue ?? "unknown"))].sort(),
    leg1IntentBuckets: [...new Set(trainRows.map((r) => r.leg1OrderIntent ?? "unknown"))].sort(),
    leg2IntentBuckets: [...new Set(trainRows.map((r) => r.leg2OrderIntent ?? "unknown"))].sort(),
  };
}
function fitScaler(trainRows, numericKeys) {
  const stats = {};
  for (const key of numericKeys) {
    const values = trainRows.map((r) => safeNum(r[key])).filter((x) => x !== null);
    const mu = mean(values) ?? 0;
    const sigma = Math.sqrt(mean(values.map((v) => (v - mu) ** 2)) ?? 1);
    stats[key] = { mean: mu, std: sigma > 1e-8 ? sigma : 1 };
  }
  return stats;
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

function ridgeTrain(X, y, { lr = 0.01, epochs = 1500, lambda = 0.2 } = {}) {
  const n = X.length;
  const d = X[0].length;
  const w = new Array(d).fill(0);
  for (let e = 0; e < epochs; e++) {
    const g = new Array(d).fill(0);
    for (let i = 0; i < n; i++) {
      const err = dot(w, X[i]) - y[i];
      for (let j = 0; j < d; j++) g[j] += (2 / n) * err * X[i][j];
    }
    for (let j = 1; j < d; j++) g[j] += 2 * lambda * w[j];
    for (let j = 0; j < d; j++) w[j] -= lr * g[j];
  }
  return w;
}
function logisticTrain(X, y, { lr = 0.02, epochs = 1800, lambda = 0.1, posWeight = 1 } = {}) {
  const n = X.length;
  const d = X[0].length;
  const w = new Array(d).fill(0);
  for (let e = 0; e < epochs; e++) {
    const g = new Array(d).fill(0);
    for (let i = 0; i < n; i++) {
      const p = sigmoid(dot(w, X[i]));
      const weight = y[i] === 1 ? posWeight : 1;
      const err = (p - y[i]) * weight;
      for (let j = 0; j < d; j++) g[j] += err * X[i][j];
    }
    for (let j = 0; j < d; j++) g[j] /= n;
    for (let j = 1; j < d; j++) g[j] += 2 * lambda * w[j];
    for (let j = 0; j < d; j++) w[j] -= lr * g[j];
  }
  return w;
}
function fitPlatt(logits, labels, { lr = 0.01, epochs = 1200, l2 = 0.01 } = {}) {
  let a = 1;
  let b = 0;
  const n = logits.length;
  if (n === 0) return { a, b };
  for (let e = 0; e < epochs; e++) {
    let ga = 0;
    let gb = 0;
    for (let i = 0; i < n; i++) {
      const z = a * logits[i] + b;
      const p = sigmoid(z);
      const err = p - labels[i];
      ga += err * logits[i];
      gb += err;
    }
    ga = ga / n + 2 * l2 * a;
    gb = gb / n + 2 * l2 * b;
    a -= lr * ga;
    b -= lr * gb;
  }
  return { a, b };
}
function groupBySegment(rows) {
  const m = new Map();
  for (const r of rows) {
    const key = r._segmentKey;
    if (!m.has(key)) m.set(key, []);
    m.get(key).push(r);
  }
  return m;
}

const raw = await fs.readFile(inputPath, "utf8");
const rawRows = raw
  .split("\n")
  .map((line) => line.trim())
  .filter(Boolean)
  .map((line) => JSON.parse(line));

const rows = prepareRows(rawRows);
const split = splitChronological(rows);
const schema = buildSchema(split.train);
const scaler = fitScaler(split.train, schema.numericKeys);

const regTrainRows = split.train.filter((r) => !r._censored && r._regTarget !== null);
const regValidRows = split.valid.filter((r) => !r._censored && r._regTarget !== null);
const regTestRows = split.test.filter((r) => !r._censored && r._regTarget !== null);

const clsTrainRows = split.train.filter((r) => !r._censored && r._clsTarget !== null);
const clsValidRows = split.valid.filter((r) => !r._censored && r._clsTarget !== null);
const clsTestRows = split.test.filter((r) => !r._censored && r._clsTarget !== null);

const regTrainYRaw = regTrainRows.map((r) => r._regTarget);
const winsorLo = quantile(regTrainYRaw, 0.05);
const winsorHi = quantile(regTrainYRaw, 0.95);

const vectorize = (rowsIn) => rowsIn.map((r) => buildVector(r, schema, scaler));

const regGlobalX = vectorize(regTrainRows);
const regGlobalY = regTrainRows.map((r) => clip(r._regTarget, winsorLo, winsorHi));
const regGlobalW = ridgeTrain(regGlobalX, regGlobalY);

const pos = clsTrainRows.filter((r) => r._clsTarget === 1).length;
const neg = clsTrainRows.filter((r) => r._clsTarget === 0).length;
const posWeight = pos > 0 ? neg / pos : 1;
const clsGlobalX = vectorize(clsTrainRows);
const clsGlobalY = clsTrainRows.map((r) => r._clsTarget);
const clsGlobalW = logisticTrain(clsGlobalX, clsGlobalY, { posWeight });

const regSegments = {};
const clsSegments = {};
const regGroups = groupBySegment(regTrainRows);
const clsGroups = groupBySegment(clsTrainRows);

for (const [seg, segRows] of regGroups.entries()) {
  if (segRows.length < MIN_SEGMENT_ROWS) continue;
  const X = vectorize(segRows);
  const rawY = segRows.map((r) => r._regTarget);
  const lo = quantile(rawY, 0.05);
  const hi = quantile(rawY, 0.95);
  const y = rawY.map((v) => clip(v, lo, hi));
  regSegments[seg] = { weights: ridgeTrain(X, y), lo, hi, rows: segRows.length };
}
for (const [seg, segRows] of clsGroups.entries()) {
  if (segRows.length < MIN_SEGMENT_ROWS) continue;
  const X = vectorize(segRows);
  const y = segRows.map((r) => r._clsTarget);
  const p = y.filter((v) => v === 1).length;
  const n = y.filter((v) => v === 0).length;
  if (p === 0 || n === 0) continue;
  clsSegments[seg] = { weights: logisticTrain(X, y, { posWeight: n / p }), rows: segRows.length };
}

function predictRegRow(row) {
  const x = buildVector(row, schema, scaler);
  const seg = regSegments[row._segmentKey];
  if (seg) return clip(dot(seg.weights, x), seg.lo, seg.hi);
  return clip(dot(regGlobalW, x), winsorLo, winsorHi);
}
function predictClsLogitRow(row) {
  const x = buildVector(row, schema, scaler);
  const seg = clsSegments[row._segmentKey];
  const w = seg ? seg.weights : clsGlobalW;
  return dot(w, x);
}

const validLogits = clsValidRows.map(predictClsLogitRow);
const validY = clsValidRows.map((r) => r._clsTarget);
const platt = fitPlatt(validLogits, validY);

function calibratedProb(logit) {
  return sigmoid(platt.a * logit + platt.b);
}
const clsProbTrain = clsTrainRows.map((r) => calibratedProb(predictClsLogitRow(r)));
const clsProbValid = clsValidRows.map((r) => calibratedProb(predictClsLogitRow(r)));
const clsProbTest = clsTestRows.map((r) => calibratedProb(predictClsLogitRow(r)));

const regPredTrain = regTrainRows.map(predictRegRow);
const regPredValid = regValidRows.map(predictRegRow);
const regPredTest = regTestRows.map(predictRegRow);

function decisionPnL(rowsIn, probs, threshold) {
  const rewards = rowsIn.map((r, i) => (probs[i] >= threshold ? 0 : r._regTarget));
  return {
    meanRelativePnlVsAlwaysBuyNow: mean(rewards),
    totalRelativePnlVsAlwaysBuyNow: rewards.reduce((a, b) => a + b, 0),
    buyRate: mean(probs.map((p) => (p >= threshold ? 1 : 0))),
  };
}
const thresholdGrid = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65];
const validThresholds = thresholdGrid.map((t) => ({
  threshold: t,
  ...decisionPnL(clsValidRows, clsProbValid, t),
}));
validThresholds.sort((a, b) => b.meanRelativePnlVsAlwaysBuyNow - a.meanRelativePnlVsAlwaysBuyNow);
const tunedThreshold = validThresholds[0].threshold;

const regression = {
  target: "deltaNetPnlPolicyWindowAtNowSize",
  train: regressionMetrics(regTrainRows.map((r) => r._regTarget), regPredTrain),
  valid: regressionMetrics(regValidRows.map((r) => r._regTarget), regPredValid),
  test: regressionMetrics(regTestRows.map((r) => r._regTarget), regPredTest),
  byDomainOnTest: Object.fromEntries(
    DOMAIN_KEYS.map((d) => {
      const idx = regTestRows.map((r, i) => ({ r, i })).filter((x) => x.r._taxonomyDomain === d);
      const y = idx.map((x) => x.r._regTarget);
      const p = idx.map((x) => regPredTest[x.i]);
      return [d, regressionMetrics(y, p)];
    })
  ),
};

const classification = {
  target: "buyNowBeatsWaitWindow",
  train: classificationMetrics(clsTrainRows.map((r) => r._clsTarget), clsProbTrain, tunedThreshold),
  valid: classificationMetrics(clsValidRows.map((r) => r._clsTarget), clsProbValid, tunedThreshold),
  test: classificationMetrics(clsTestRows.map((r) => r._clsTarget), clsProbTest, tunedThreshold),
  tunedThreshold,
  thresholdTuningOnValidation: validThresholds,
  byDomainOnTest: Object.fromEntries(
    DOMAIN_KEYS.map((d) => {
      const idx = clsTestRows.map((r, i) => ({ r, i })).filter((x) => x.r._taxonomyDomain === d);
      const y = idx.map((x) => x.r._clsTarget);
      const p = idx.map((x) => clsProbTest[x.i]);
      return [d, classificationMetrics(y, p, tunedThreshold)];
    })
  ),
};

const report = {
  generatedAt: new Date().toISOString(),
  sample: {
    totalRows: rows.length,
    train: split.train.length,
    valid: split.valid.length,
    test: split.test.length,
    regressionRows: { train: regTrainRows.length, valid: regValidRows.length, test: regTestRows.length },
    classificationRows: { train: clsTrainRows.length, valid: clsValidRows.length, test: clsTestRows.length },
  },
  setup: {
    winsorization: { lowQuantile: 0.05, highQuantile: 0.95, lowValue: winsorLo, highValue: winsorHi },
    segmentation: {
      minSegmentRows: MIN_SEGMENT_ROWS,
      regressionSegmentModels: Object.keys(regSegments).length,
      classificationSegmentModels: Object.keys(clsSegments).length,
    },
    calibration: { type: "platt", a: platt.a, b: platt.b },
  },
  models: { regression, classification },
  policyBacktestSummary: {
    validationAtTunedThreshold: decisionPnL(clsValidRows, clsProbValid, tunedThreshold),
    testAtTunedThreshold: decisionPnL(clsTestRows, clsProbTest, tunedThreshold),
  },
};

const artifacts = {
  generatedAt: report.generatedAt,
  schema,
  scaler,
  winsorization: { low: winsorLo, high: winsorHi },
  regression: { globalWeights: regGlobalW, segments: regSegments },
  classification: {
    globalWeights: clsGlobalW,
    segments: clsSegments,
    platt,
    tunedThreshold,
  },
};

await fs.writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
await fs.writeFile(artifactsPath, `${JSON.stringify(artifacts, null, 2)}\n`, "utf8");
console.log(`Wrote robust model report: ${reportPath}`);
console.log(`Wrote robust model artifacts: ${artifactsPath}`);
console.log(JSON.stringify(report, null, 2));
