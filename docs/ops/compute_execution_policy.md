# Compute Execution Policy

## Baseline machine
- profile: `macbook_air_m2_2022`
- hardware context: Apple MacBook Air (M2, 2022)
- purpose: local smoke/debug only for qualification workflows

## Placement rule
A run is placed on `remote` when any trigger is true:
1. estimated runtime per run `> 360` minutes (6 hours)
2. estimated batch runtime `> 360` minutes (6 hours)
3. seeds per condition `> 2`
4. projected memory footprint exceeds safe local budget (`8.0 GB`)
5. local thermal throttling detected
6. local OOM / repeated fatal runtime instability detected

If none of the above triggers are true, place run on `local`.

## Lane policy
- `ree-v2`: qualification lane (primary)
- `ree-experiments-lab`: stress/falsification lane (unchanged)
- `ree-v1-minimal`: temporary parity backstop during migration

## Local-allowed tasks
- schema validation
- hook coverage validation
- deterministic replay checks on smoke workloads
- single-seed quick debug runs

## Remote-required tasks
- qualification sweeps with >= 3 seeds per condition
- high-memory multi-condition comparison batches
- anything projected to violate local trigger thresholds

## Command entry points
- estimate: `python3 scripts/estimate_run_resources.py --profile all --machine macbook_air_m2_2022`
- job export: `python3 scripts/build_remote_job_spec.py --profile all`
- submit: `python3 scripts/submit_remote_job.py --job-spec-dir jobs/outgoing --dry-run`
- import: `python3 scripts/pull_remote_results.py --job-run-dir jobs/completed --runs-root evidence/experiments --dry-run`
