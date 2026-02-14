# Local Compute Options

## Baseline
- machine: MacBook Air (M2, 2022)
- bottlenecks:
  - runtime: long multi-seed sweeps exceed practical local cycle time
  - memory: uncertainty-heavy profiles can exceed safe local budget
  - thermal: sustained sweeps trigger throttling risk
  - disk: manageable for smoke runs, but large result retention benefits external storage

## Purchase tiers

| tier | estimated_eur_cost_band | expected_runtime_impact | setup_complexity | recommendation_status |
| --- | --- | --- | --- | --- |
| low-cost improvement tier | EUR 250-700 | modest speedup from memory/storage headroom; fewer blocked debug sessions | low | now |
| mid-cost workstation tier | EUR 1800-3500 | major reduction in local sweep wall-clock, better multi-seed parallelism | medium | later |
| high-cost local acceleration tier | EUR 4500-9000 | high throughput for sustained qualification loads; can reduce cloud dependence | high | not_recommended |

## Tier details

### low-cost improvement tier
- examples: RAM-optimized refurb desktop mini, external NVMe scratch, thermal stand/fan kit.
- impact: improves local iteration smoothness, not a replacement for heavy remote sweeps.
- use when cloud spend is low but local friction is rising.

### mid-cost workstation tier
- examples: higher-memory workstation laptop/desktop with stronger sustained CPU/GPU.
- impact: enables more qualification preparation runs locally before cloud handoff.
- use when local blocking is frequent or cloud spend becomes consistently high.

### high-cost local acceleration tier
- examples: dedicated high-end workstation + local accelerator setup.
- impact: useful only for consistently heavy weekly qualification load.
- use only when sustained workload and spend thresholds justify ownership cost.

## Decision triggers
- keep cloud-first (`hold_cloud_only`) when qualification sweeps are sporadic and local smoke/debug is sufficient.
- recommend low-tier purchase when local friction is increasing and rolling 3-month cloud spend remains below `EUR 80/month`.
- recommend mid-tier purchase when either:
  - rolling 3-month cloud spend is `>= EUR 100/month`, or
  - blocked local sessions are `>2/week` for 3 consecutive weeks.
- recommend high-tier purchase only when both are true:
  - rolling 3-month cloud spend is `>= EUR 250/month` for 3 consecutive months,
  - active qualification workload is `>10 hours/week`.

## Hobby-mode default thresholds
- default: `hold_cloud_only`
- `upgrade_low` when local friction increases and rolling 3-month cloud spend is below `EUR 80/month`
- `upgrade_mid` when rolling 3-month cloud spend is `>= EUR 100/month` or blocked sessions are `>2/week` for 3 consecutive weeks
- `upgrade_high` only when rolling 3-month cloud spend is `>= EUR 250/month` for 3 months and workload is `>10 hours/week`
