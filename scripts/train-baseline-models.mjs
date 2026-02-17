import fs from "node:fs/promises";
import path from "node:path";

const inputPath = path.resolve("data", "labeled_training_dataset.jsonl");
const reportPath = path.resolve("data", "model_baseline_report.json");
const modelPath = path.resolve("data", "model_baseline_artifacts.json");

const DOMAIN_KEYS = ["politics", "sports", "finance", "technology", "culture", "other"];

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

function variance(values, mu) {
  if (values.length === 0 || mu === null) {
    return null;
  }
  return mean(values.map((v) => (v - mu) ** 2));
}

function std(values, mu) {
  const v = variance(values, mu);
  return v === null ? null : Math.sqrt(v);
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
  for (let i = 0; i < a.length; i++) {
    s += a[i] * b[i];
  }
  return s;
}

function clip(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function aucRoc(yTrue, yScore) {
  if (yTrue.length === 0 || yScore.length !== yTrue.length) {
    return null;
  }
  const pairs = yTrue.map((y, i) => ({ y, s: yScore[i] }));
  const pos = pairs.filter((p) => p.y === 1).length;
  const neg = pairs.filter((p) => p.y === 0).length;
  if (pos === 0 || neg === 0) {
    return null;
  }
  pairs.sort((a, b) => a.s - b.s);
  let rankSumPos = 0;
  for (let i = 0; i < pairs.length; i++) {
    if (pairs[i].y === 1) {
      rankSumPos += i + 1;
    }
  }
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
  const accuracy = yTrue.length === 0 ? null : (tp + tn) / yTrue.length;
  return {
    count: yTrue.length,
    accuracy,
    precision,
    recall,
    f1,
    auc: aucRoc(yTrue, yProb),
    confusion: { tp, fp, tn, fn },
  };
}

function regressionMetrics(yTrue, yPred) {
  if (yTrue.length === 0 || yPred.length !== yTrue.length) {
    return {
      count: 0,
      mae: null,
      rmse: null,
      r2: null,
      signAccuracy: null,
    };
  }
  const errors = yTrue.map((y, i) => yPred[i] - y);
  const absErrors = errors.map((e) => Math.abs(e));
  const sqErrors = errors.map((e) => e * e);
  const mu = mean(yTrue);
  const sst = yTrue.reduce((acc, y) => acc + (y - mu) ** 2, 0);
  const sse = sqErrors.reduce((a, b) => a + b, 0);
  const signMatches = yTrue.map((y, i) => (Math.sign(yPred[i]) === Math.sign(y) ? 1 : 0));
  return {
    count: yTrue.length,
    mae: mean(absErrors),
    rmse: Math.sqrt(mean(sqErrors)),
    r2: sst === 0 ? null : 1 - sse / sst,
    signAccuracy: mean(signMatches),
  };
}

function oneHotValue(value, buckets) {
  const out = new Array(buckets.length).fill(0);
  const idx = buckets.indexOf(value);
  if (idx >= 0) out[idx] = 1;
  return out;
}

function prepareRows(rows) {
  const enriched = [];
  for (const row of rows) {
    const labels = row.labels?.resolutionAnchored ?? null;
    const tax = row.labels?.taxonomy ?? null;
    if (!labels || !tax) {
      continue;
    }
    const decisionTsMs = new Date(row.decisionTs).getTime();
    if (!Number.isFinite(decisionTsMs)) {
      continue;
    }
    enriched.push({
      ...row,
      _decisionTsMs: decisionTsMs,
      _regTarget: safeNum(labels.deltaNetPnlPolicyWindowAtNowSize),
      _clsTarget:
        labels.buyNowBeatsWaitWindow === null || labels.buyNowBeatsWaitWindow === undefined
          ? null
          : labels.buyNowBeatsWaitWindow
            ? 1
            : 0,
      _censored: Boolean(labels.labelCensoredPolicyWindow),
      _phaseNow: labels.phaseNow ?? "unknown",
      _ttrHours: safeNum(labels.timeToResolutionHoursNow),
      _policyWindowHours: safeNum(labels.policyWindowHours),
      _taxonomyDomain: tax.domain ?? "other",
      _taxonomySubdomain: tax.subdomain ?? "other",
      _taxonomyTopic: tax.topic ?? "unknown",
    });
  }
  enriched.sort((a, b) => a._decisionTsMs - b._decisionTsMs);
  return enriched;
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

function buildFeatureSchema(trainRows) {
  const phaseBuckets = [...new Set(trainRows.map((r) => r._phaseNow))].sort();
  const domainBuckets = [...new Set(trainRows.map((r) => r._taxonomyDomain))].sort();
  const leg1VenueBuckets = [...new Set(trainRows.map((r) => r.leg1Venue ?? "unknown"))].sort();
  const leg2VenueBuckets = [...new Set(trainRows.map((r) => r.leg2Venue ?? "unknown"))].sort();
  const leg1IntentBuckets = [...new Set(trainRows.map((r) => r.leg1OrderIntent ?? "unknown"))].sort();
  const leg2IntentBuckets = [...new Set(trainRows.map((r) => r.leg2OrderIntent ?? "unknown"))].sort();

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
    phaseBuckets,
    domainBuckets,
    leg1VenueBuckets,
    leg2VenueBuckets,
    leg1IntentBuckets,
    leg2IntentBuckets,
  };
}

function fitScaler(trainRows, numericKeys) {
  const stats = {};
  for (const key of numericKeys) {
    const values = trainRows.map((r) => safeNum(r[key])).filter((x) => x !== null);
    const mu = mean(values);
    const sd = std(values, mu);
    stats[key] = {
      mean: mu ?? 0,
      std: sd && sd > 1e-8 ? sd : 1,
    };
  }
  return stats;
}

function buildVector(row, schema, scalerStats) {
  const x = [1];

  for (const key of schema.numericKeys) {
    const v = safeNum(row[key]);
    const s = scalerStats[key];
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

function ridgeTrain(X, y, { lr = 0.01, epochs = 1200, lambda = 0.1 } = {}) {
  const n = X.length;
  const d = X[0].length;
  const w = new Array(d).fill(0);

  for (let epoch = 0; epoch < epochs; epoch++) {
    const grad = new Array(d).fill(0);
    for (let i = 0; i < n; i++) {
      const pred = dot(w, X[i]);
      const err = pred - y[i];
      for (let j = 0; j < d; j++) {
        grad[j] += (2 / n) * err * X[i][j];
      }
    }
    for (let j = 1; j < d; j++) {
      grad[j] += 2 * lambda * w[j];
    }
    for (let j = 0; j < d; j++) {
      w[j] -= lr * grad[j];
    }
  }
  return w;
}

function logisticTrain(
  X,
  y,
  { lr = 0.02, epochs = 1500, lambda = 0.1, positiveClassWeight = 1.0 } = {}
) {
  const n = X.length;
  const d = X[0].length;
  const w = new Array(d).fill(0);

  for (let epoch = 0; epoch < epochs; epoch++) {
    const grad = new Array(d).fill(0);
    for (let i = 0; i < n; i++) {
      const z = dot(w, X[i]);
      const p = sigmoid(z);
      const weight = y[i] === 1 ? positiveClassWeight : 1;
      const err = (p - y[i]) * weight;
      for (let j = 0; j < d; j++) {
        grad[j] += err * X[i][j];
      }
    }
    for (let j = 0; j < d; j++) {
      grad[j] /= n;
    }
    for (let j = 1; j < d; j++) {
      grad[j] += 2 * lambda * w[j];
    }
    for (let j = 0; j < d; j++) {
      w[j] -= lr * grad[j];
    }
  }
  return w;
}

function predictLinear(X, w) {
  return X.map((x) => dot(w, x));
}

function predictProba(X, w) {
  return X.map((x) => sigmoid(dot(w, x)));
}

function subsetByDomain(rows, yTrue, yPred, domain) {
  const t = [];
  const p = [];
  for (let i = 0; i < rows.length; i++) {
    if (rows[i]._taxonomyDomain === domain) {
      t.push(yTrue[i]);
      p.push(yPred[i]);
    }
  }
  return { yTrue: t, yPred: p, count: t.length };
}

const raw = await fs.readFile(inputPath, "utf8");
const rawRows = raw
  .split("\n")
  .map((line) => line.trim())
  .filter(Boolean)
  .map((line) => JSON.parse(line));

const rows = prepareRows(rawRows);
const split = splitChronological(rows);
const schema = buildFeatureSchema(split.train);
const scaler = fitScaler(split.train, schema.numericKeys);

const buildDataset = (sourceRows, task) => {
  const filtered = sourceRows.filter((r) => {
    if (task === "regression") {
      return !r._censored && r._regTarget !== null;
    }
    return !r._censored && r._clsTarget !== null;
  });
  const X = filtered.map((r) => buildVector(r, schema, scaler));
  const y = filtered.map((r) => (task === "regression" ? r._regTarget : r._clsTarget));
  return { rows: filtered, X, y };
};

const regTrain = buildDataset(split.train, "regression");
const regValid = buildDataset(split.valid, "regression");
const regTest = buildDataset(split.test, "regression");

const regW = ridgeTrain(regTrain.X, regTrain.y, {
  lr: 0.01,
  epochs: 1400,
  lambda: 0.2,
});

const regPredTrain = predictLinear(regTrain.X, regW).map((v) => clip(v, -5000, 5000));
const regPredValid = predictLinear(regValid.X, regW).map((v) => clip(v, -5000, 5000));
const regPredTest = predictLinear(regTest.X, regW).map((v) => clip(v, -5000, 5000));

const clsTrain = buildDataset(split.train, "classification");
const clsValid = buildDataset(split.valid, "classification");
const clsTest = buildDataset(split.test, "classification");

const pos = clsTrain.y.filter((x) => x === 1).length;
const neg = clsTrain.y.filter((x) => x === 0).length;
const posWeight = pos > 0 ? neg / pos : 1;
const clsW = logisticTrain(clsTrain.X, clsTrain.y, {
  lr: 0.02,
  epochs: 1800,
  lambda: 0.1,
  positiveClassWeight: posWeight,
});

const clsProbTrain = predictProba(clsTrain.X, clsW);
const clsProbValid = predictProba(clsValid.X, clsW);
const clsProbTest = predictProba(clsTest.X, clsW);

const regression = {
  target: "deltaNetPnlPolicyWindowAtNowSize",
  train: regressionMetrics(regTrain.y, regPredTrain),
  valid: regressionMetrics(regValid.y, regPredValid),
  test: regressionMetrics(regTest.y, regPredTest),
  byDomainOnTest: Object.fromEntries(
    DOMAIN_KEYS.map((domain) => {
      const s = subsetByDomain(regTest.rows, regTest.y, regPredTest, domain);
      return [domain, regressionMetrics(s.yTrue, s.yPred)];
    })
  ),
};

const classification = {
  target: "buyNowBeatsWaitWindow",
  train: classificationMetrics(clsTrain.y, clsProbTrain),
  valid: classificationMetrics(clsValid.y, clsProbValid),
  test: classificationMetrics(clsTest.y, clsProbTest),
  byDomainOnTest: Object.fromEntries(
    DOMAIN_KEYS.map((domain) => {
      const s = subsetByDomain(clsTest.rows, clsTest.y, clsProbTest, domain);
      return [domain, classificationMetrics(s.yTrue, s.yPred)];
    })
  ),
};

const report = {
  generatedAt: new Date().toISOString(),
  sample: {
    totalRows: rows.length,
    splitRows: {
      train: split.train.length,
      valid: split.valid.length,
      test: split.test.length,
    },
    regressionRows: {
      train: regTrain.y.length,
      valid: regValid.y.length,
      test: regTest.y.length,
    },
    classificationRows: {
      train: clsTrain.y.length,
      valid: clsValid.y.length,
      test: clsTest.y.length,
      trainClassBalance: {
        positive: pos,
        negative: neg,
      },
    },
  },
  models: {
    regression,
    classification,
  },
  notes: [
    "Chronological split is used to avoid time leakage.",
    "Rows censored for policy window labels are excluded from supervised training.",
    "Baseline uses linear/logistic models with one-hot categorical features and standardized numeric features.",
  ],
};

const artifacts = {
  generatedAt: report.generatedAt,
  schema,
  scaler,
  regression: {
    algorithm: "ridge_linear",
    weights: regW,
  },
  classification: {
    algorithm: "logistic_l2",
    weights: clsW,
    positiveClassWeight: posWeight,
  },
};

await fs.writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
await fs.writeFile(modelPath, `${JSON.stringify(artifacts, null, 2)}\n`, "utf8");

console.log(`Wrote model baseline report: ${reportPath}`);
console.log(`Wrote model artifacts: ${modelPath}`);
console.log(JSON.stringify(report, null, 2));
