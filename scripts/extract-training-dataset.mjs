import "dotenv/config";
import fs from "node:fs/promises";
import path from "node:path";
import { Client } from "pg";

const databaseUrl = process.env.DATABASE_URL;

if (!databaseUrl) {
  console.error("Missing DATABASE_URL in .env");
  process.exit(1);
}

const outputPath = path.resolve("data", "training_dataset.jsonl");

const query = `
with sized_artifacts as (
  select
    a.opportunity_id,
    a.portfolio_id,
    a.created_at as decision_ts,
    o.dedupe_key,
    o.strategy_type,
    o.status as opportunity_status,
    o.created_at as opportunity_created_at,
    a.artifact->'sizing'->>'expectedEdge' as expected_edge_at_decision,
    a.artifact->'sizing'->>'targetContracts' as target_contracts_at_decision,
    (
      select min((kl->>'contracts')::numeric)
      from jsonb_array_elements(coalesce(a.artifact->'sizing'->'kernelLegs', '[]'::jsonb)) as kl
      where (kl->>'contracts') is not null
    ) as min_kernel_contracts_at_decision,
    (
      select avg((kl->>'price')::numeric)
      from jsonb_array_elements(coalesce(a.artifact->'sizing'->'kernelLegs', '[]'::jsonb)) as kl
      where (kl->>'price') is not null
    ) as avg_leg_price_at_decision,
    a.artifact->'sizing'->>'budgetUsd' as budget_usd_at_decision,
    a.artifact->>'requestUsd' as request_usd,
    a.artifact->>'availableUsd' as available_usd,
    a.artifact->'sizing'->'decisionProvenance'->>'capturedAt' as decision_captured_at,
    a.artifact->'sizing'->'legs'->0->>'venue' as leg1_venue,
    coalesce(
      a.artifact->'sizing'->'legs'->0->>'marketId',
      o.payload->'legs'->0->>'marketId'
    ) as leg1_market_id,
    coalesce(
      a.artifact->'sizing'->'legs'->0->>'symbolId',
      o.payload->'legs'->0->>'symbolId'
    ) as leg1_symbol_id,
    a.artifact->'sizing'->'legs'->0->>'side' as leg1_side,
    a.artifact->'sizing'->'legs'->0->>'orderIntent' as leg1_order_intent,
    a.artifact->'sizing'->'legs'->0->>'venueFeeClass' as leg1_fee_class,
    a.artifact->'sizing'->'legs'->0->>'expiresAt' as leg1_expires_at,
    o.payload->'legs'->0->>'expiresAt' as opp_leg1_expires_at,
    a.artifact->'sizing'->'legs'->1->>'venue' as leg2_venue,
    coalesce(
      a.artifact->'sizing'->'legs'->1->>'marketId',
      o.payload->'legs'->1->>'marketId'
    ) as leg2_market_id,
    coalesce(
      a.artifact->'sizing'->'legs'->1->>'symbolId',
      o.payload->'legs'->1->>'symbolId'
    ) as leg2_symbol_id,
    a.artifact->'sizing'->'legs'->1->>'side' as leg2_side,
    a.artifact->'sizing'->'legs'->1->>'orderIntent' as leg2_order_intent,
    a.artifact->'sizing'->'legs'->1->>'venueFeeClass' as leg2_fee_class,
    a.artifact->'sizing'->'legs'->1->>'expiresAt' as leg2_expires_at,
    o.payload->'legs'->1->>'expiresAt' as opp_leg2_expires_at,
    jsonb_array_length(coalesce(a.artifact->'sizing'->'legs', '[]'::jsonb)) as leg_count
  from arb_execution_decision_artifacts a
  join arb_opportunities o on o.id = a.opportunity_id
  where a.status = 'sized'
    and o.strategy_type = 'cross_venue_binary'
    and a.created_at >= now() - interval '30 days'
),
latest_execution as (
  select distinct on (e.opportunity_id)
    e.opportunity_id,
    e.state as execution_state,
    e.actual_edge,
    e.expected_edge,
    e.realized_pnl,
    e.target_contracts as execution_target_contracts,
    e.completed_at
  from arb_executions e
  where e.created_at >= now() - interval '30 days'
  order by e.opportunity_id, e.created_at desc
),
enriched_markets as (
  select
    s.*,
    coalesce(k1.category, p1.category) as leg1_market_category,
    coalesce(k1.title, p1.question) as leg1_market_title,
    coalesce(k1.series_ticker, p1.event_slug) as leg1_market_subcategory,
    coalesce(k1.event_ticker, p1.event_title) as leg1_event_title,
    p1.tags as leg1_market_tags,
    coalesce(k2.category, p2.category) as leg2_market_category,
    coalesce(k2.title, p2.question) as leg2_market_title,
    coalesce(k2.series_ticker, p2.event_slug) as leg2_market_subcategory,
    coalesce(k2.event_ticker, p2.event_title) as leg2_event_title,
    p2.tags as leg2_market_tags
  from sized_artifacts s
  left join kalshi_markets k1
    on s.leg1_venue = 'KALSHI' and k1.ticker = s.leg1_market_id
  left join polymarket_markets p1
    on s.leg1_venue = 'POLYMARKET' and p1.condition_id = s.leg1_market_id
  left join kalshi_markets k2
    on s.leg2_venue = 'KALSHI' and k2.ticker = s.leg2_market_id
  left join polymarket_markets p2
    on s.leg2_venue = 'POLYMARKET' and p2.condition_id = s.leg2_market_id
)
select
  s.*,
  e.execution_state,
  e.actual_edge,
  e.expected_edge as execution_expected_edge,
  e.realized_pnl,
  e.execution_target_contracts,
  e.completed_at
from enriched_markets s
left join latest_execution e on e.opportunity_id = s.opportunity_id
order by s.decision_ts asc;
`;

function safeNumber(value) {
  if (value === null || value === undefined || value === "") {
    return null;
  }
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function parseFeatures(row) {
  const parseTags = (value) => {
    if (value === null || value === undefined) {
      return [];
    }
    if (Array.isArray(value)) {
      return value.filter((t) => typeof t === "string");
    }
    return [];
  };

  return {
    opportunityId: row.opportunity_id,
    dedupeKey: row.dedupe_key ?? null,
    portfolioId: row.portfolio_id,
    strategyType: row.strategy_type,
    opportunityStatus: row.opportunity_status,
    decisionTs: row.decision_ts?.toISOString?.() ?? row.decision_ts,
    decisionCapturedAt: row.decision_captured_at ?? null,
    opportunityCreatedAt:
      row.opportunity_created_at?.toISOString?.() ?? row.opportunity_created_at,
    expectedEdgeAtDecision: safeNumber(row.expected_edge_at_decision),
    targetContractsAtDecision: safeNumber(row.target_contracts_at_decision),
    minKernelContractsAtDecision: safeNumber(row.min_kernel_contracts_at_decision),
    avgLegPriceAtDecision: safeNumber(row.avg_leg_price_at_decision),
    budgetUsdAtDecision: safeNumber(row.budget_usd_at_decision),
    requestUsd: safeNumber(row.request_usd),
    availableUsd: safeNumber(row.available_usd),
    legCount: safeNumber(row.leg_count),
    leg1Venue: row.leg1_venue ?? null,
    leg1MarketId: row.leg1_market_id ?? null,
    leg1SymbolId: row.leg1_symbol_id ?? null,
    leg1Side: row.leg1_side ?? null,
    leg1OrderIntent: row.leg1_order_intent ?? null,
    leg1FeeClass: row.leg1_fee_class ?? null,
    leg1ExpiresAt: row.leg1_expires_at ?? null,
    oppLeg1ExpiresAt: row.opp_leg1_expires_at ?? null,
    leg1MarketCategory: row.leg1_market_category ?? null,
    leg1MarketSubcategory: row.leg1_market_subcategory ?? null,
    leg1MarketTitle: row.leg1_market_title ?? null,
    leg1EventTitle: row.leg1_event_title ?? null,
    leg1MarketTags: parseTags(row.leg1_market_tags),
    leg2Venue: row.leg2_venue ?? null,
    leg2MarketId: row.leg2_market_id ?? null,
    leg2SymbolId: row.leg2_symbol_id ?? null,
    leg2Side: row.leg2_side ?? null,
    leg2OrderIntent: row.leg2_order_intent ?? null,
    leg2FeeClass: row.leg2_fee_class ?? null,
    leg2ExpiresAt: row.leg2_expires_at ?? null,
    oppLeg2ExpiresAt: row.opp_leg2_expires_at ?? null,
    leg2MarketCategory: row.leg2_market_category ?? null,
    leg2MarketSubcategory: row.leg2_market_subcategory ?? null,
    leg2MarketTitle: row.leg2_market_title ?? null,
    leg2EventTitle: row.leg2_event_title ?? null,
    leg2MarketTags: parseTags(row.leg2_market_tags),
    executionState: row.execution_state ?? null,
    actualEdgeAtExecution: safeNumber(row.actual_edge),
    expectedEdgeAtExecution: safeNumber(row.execution_expected_edge),
    realizedPnl: safeNumber(row.realized_pnl),
    executionTargetContracts: safeNumber(row.execution_target_contracts),
    executionCompletedAt: row.completed_at?.toISOString?.() ?? row.completed_at ?? null,
  };
}

const client = new Client({ connectionString: databaseUrl });
await client.connect();

try {
  const result = await client.query(query);
  const rows = result.rows.map(parseFeatures);

  const lines = rows.map((r) => JSON.stringify(r)).join("\n");
  await fs.writeFile(outputPath, `${lines}\n`, "utf8");

  const withExecution = rows.filter((r) => r.executionState !== null).length;

  console.log(`Wrote dataset rows: ${rows.length}`);
  console.log(`Rows with execution labels: ${withExecution}`);
  console.log(`Output: ${outputPath}`);
} finally {
  await client.end();
}
