import fs from "node:fs/promises";
import path from "node:path";

const inputPath = path.resolve("data", "labeled_training_dataset.jsonl");
const outputPath = path.resolve("data", "walkforward_backtest_report.json");

const MIN_SEGMENT_ROWS = 80;

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
    const reg = safeNum(res.deltaNetPnlPolicyWindowAtNowSize);
    out.push({
      ...row,
      _decisionTsMs: ts,
      _regTarget: reg,
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
    const vals = trainRows.map((r) => safeNum(r[key])).filter((x) => x !== null);
    const mu = mean(vals) ?? 0;
    const sd = Math.sqrt(mean(vals.map((v) => (v - mu) ** 2)) ?? 1);
    stats[key] = { mean: mu, std: sd > 1e-8 ? sd : 1 };
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
function ridgeTrain(X, y, { lr = 0.01, epochs = 1200, lambda = 0.2 } = {}) {
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
function logisticTrain(X, y, { lr = 0.02, epochs = 1500, lambda = 0.1, posWeight = 1 } = {}) {
  const n = X.length;
  const d = X[0].length;
  const w = new Array(d).fill(0);
  for (let e = 0; e < epochs; e++) {
    const g = new Array(d).fill(0);
    for (let i = 0; i < n; i++) {
      const p = sigmoid(dot(w, X[i]));
      const wt = y[i] === 1 ? posWeight : 1;
      const err = (p - y[i]) * wt;
      for (let j = 0; j < d; j++) g[j] += err * X[i][j];
    }
    for (let j = 0; j < d; j++) g[j] /= n;
    for (let j = 1; j < d; j++) g[j] += 2 * lambda * w[j];
    for (let j = 0; j < d; j++) w[j] -= lr * g[j];
  }
  return w;
}
function fitPlatt(logits, labels, { lr = 0.01, epochs = 800, l2 = 0.01 } = {}) {
  let a = 1;
  let b = 0;
  const n = logits.length;
  if (n === 0) return { a, b };
  for (let e = 0; e < epochs; e++) {
    let ga = 0;
    let gb = 0;
    for (let i = 0; i < n; i++) {
      const p = sigmoid(a * logits[i] + b);
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
    if (!m.has(r._segmentKey)) m.set(r._segmentKey, []);
    m.get(r._segmentKey).push(r);
  }
  return m;
}
function decisionPnl(rows, probs, threshold) {
  const rewards = rows.map((r, i) => (probs[i] >= threshold ? 0 : r._regTarget));
  return {
    meanRelativePnl: mean(rewards),
    totalRelativePnl: rewards.reduce((a, b) => a + b, 0),
    buyRate: mean(probs.map((p) => (p >= threshold ? 1 : 0))),
  };
}

function runWindow(windowRows, trainCount, validCount, testCount) {
  const train = windowRows.slice(0, trainCount).filter((r) => !r._censored && r._regTarget !== null);
  const valid = windowRows
    .slice(trainCount, trainCount + validCount)
    .filter((r) => !r._censored && r._regTarget !== null);
  const test = windowRows
    .slice(trainCount + validCount, trainCount + validCount + testCount)
    .filter((r) => !r._censored && r._regTarget !== null);
  if (train.length < 200 || valid.length < 50 || test.length < 50) {
    return null;
  }

  const schema = buildSchema(train);
  const scaler = fitScaler(train, schema.numericKeys);
  const vectorize = (rowsIn) => rowsIn.map((r) => buildVector(r, schema, scaler));

  const regTrainX = vectorize(train);
  const regTrainYRaw = train.map((r) => r._regTarget);
  const winsorLo = quantile(regTrainYRaw, 0.05);
  const winsorHi = quantile(regTrainYRaw, 0.95);
  const regTrainY = regTrainYRaw.map((v) => clip(v, winsorLo, winsorHi));
  const regGlobalW = ridgeTrain(regTrainX, regTrainY);

  const clsTrainY = train.map((r) => r._clsTarget ?? 0);
  const pos = clsTrainY.filter((x) => x === 1).length;
  const neg = clsTrainY.filter((x) => x === 0).length;
  const posWeight = pos > 0 ? neg / pos : 1;
  const clsGlobalW = logisticTrain(regTrainX, clsTrainY, { posWeight });

  const regSegments = {};
  const clsSegments = {};
  const segGroups = groupBySegment(train);
  for (const [seg, segRows] of segGroups.entries()) {
    if (segRows.length < MIN_SEGMENT_ROWS) continue;
    const X = vectorize(segRows);
    const regYRaw = segRows.map((r) => r._regTarget);
    const lo = quantile(regYRaw, 0.05);
    const hi = quantile(regYRaw, 0.95);
    regSegments[seg] = {
      lo,
      hi,
      w: ridgeTrain(X, regYRaw.map((v) => clip(v, lo, hi))),
    };
    const cy = segRows.map((r) => r._clsTarget ?? 0);
    const p = cy.filter((x) => x === 1).length;
    const n = cy.filter((x) => x === 0).length;
    if (p > 0 && n > 0) {
      clsSegments[seg] = {
        w: logisticTrain(X, cy, { posWeight: n / p }),
      };
    }
  }

  const predictLogit = (row) => {
    const x = buildVector(row, schema, scaler);
    const seg = clsSegments[row._segmentKey];
    const w = seg ? seg.w : clsGlobalW;
    return dot(w, x);
  };

  const validLogits = valid.map(predictLogit);
  const validLabels = valid.map((r) => r._clsTarget ?? 0);
  const platt = fitPlatt(validLogits, validLabels);

  const predictProb = (row) => sigmoid(platt.a * predictLogit(row) + platt.b);

  const grid = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65];
  const validProbs = valid.map(predictProb);
  const tuning = grid
    .map((t) => ({ threshold: t, ...decisionPnl(valid, validProbs, t) }))
    .sort((a, b) => b.meanRelativePnl - a.meanRelativePnl);
  const threshold = tuning[0].threshold;

  const testProbs = test.map(predictProb);
  const testDecision = decisionPnl(test, testProbs, threshold);
  return {
    threshold,
    trainRows: train.length,
    validRows: valid.length,
    testRows: test.length,
    validBestMeanRelativePnl: tuning[0].meanRelativePnl,
    ...testDecision,
  };
}

const raw = await fs.readFile(inputPath, "utf8");
const allRows = prepareRows(
  raw
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line))
);

const trainCount = 1400;
const validCount = 300;
const testCount = 250;
const stride = 150;
const required = trainCount + validCount + testCount;
const windows = [];

for (let start = 0; start + required <= allRows.length; start += stride) {
  const slice = allRows.slice(start, start + required);
  const result = runWindow(slice, trainCount, validCount, testCount);
  if (!result) continue;
  windows.push({
    windowIndex: windows.length + 1,
    startTs: new Date(slice[0]._decisionTsMs).toISOString(),
    endTs: new Date(slice[slice.length - 1]._decisionTsMs).toISOString(),
    ...result,
  });
}

const pnlList = windows.map((w) => w.totalRelativePnl);
const report = {
  generatedAt: new Date().toISOString(),
  config: { trainCount, validCount, testCount, stride, minSegmentRows: MIN_SEGMENT_ROWS },
  summary: {
    windowCount: windows.length,
    meanTotalRelativePnlPerWindow: mean(pnlList),
    medianTotalRelativePnlPerWindow: quantile(pnlList, 0.5),
    profitableWindowRate: mean(windows.map((w) => (w.totalRelativePnl > 0 ? 1 : 0))),
    meanBuyRate: mean(windows.map((w) => w.buyRate)),
  },
  windows,
};

await fs.writeFile(outputPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
console.log(`Wrote walk-forward backtest report: ${outputPath}`);
console.log(JSON.stringify(report, null, 2));
