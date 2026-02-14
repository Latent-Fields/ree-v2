# Weekly Handoff - ree-v2 - 2026-02-09

## Metadata
- week_of_utc: `2026-02-09`
- producer_repo: `ree-v2`
- producer_commit: `4647b2e9c13572cd21d1cd3e6fb9a65b818b15da`
- generated_utc: `2026-02-14T16:11:30Z`

## Contract Sync
- ree_assembly_repo: `REE_assembly`
- ree_assembly_commit: `45b6613ed656098ae59bb76deae989e4f3378c5b`
- contract_lock_path: `contracts/ree_assembly_contract_lock.v1.json`
- contract_lock_hash: `421f756451519e768093161825bf8035231266260e11ee00b3e08caea3b3420e`
- schema_version_set: `experiment_pack/v1, experiment_pack_metrics/v1, hook_registry/v1, jepa_adapter_signals/v1`
- template_path: `/Users/dgolden/Documents/GitHub/REE_assembly/evidence/planning/WEEKLY_HANDOFF_TEMPLATE.md`
- template_sha256: `3995a20b39f04dc468c429d6d95d6f913b919c174beb40ccd3af04a081d5d0ad`

## CI Gates
| gate | status | evidence |
| --- | --- | --- |
| schema_validation | PASS | `python3 scripts/validate_experiment_pack.py --runs-root evidence/experiments` :: PASS: validated 4 run(s), schemas, adapter files, and contract lock hashes |
| seed_determinism | PASS | `python3 scripts/check_seed_determinism.py --profile all --max-abs-delta 1e-6` :: checked commit_dual_error_channels/single_error_stream seed=11 max_abs_delta=0 |
| hook_surface_coverage | PASS | `python3 scripts/validate_hook_surfaces.py --registry contracts/hook_registry.v1.json` :: PASS: hook surface contract verified for HK-001..HK-006 and HK-101..HK-104 |
| remote_export_import | PASS | `python3 scripts/estimate_run_resources.py --profile all --machine macbook_air_m2_2022` :: experiment_type	condition	estimated_runtime_minutes	execution_mode	offload_reason ; `python3 scripts/build_remote_job_spec.py --profile all --out-dir /var/folders/60/l2q_ptls0r76nzvtbqldjt4m0000gn/T/ree_v2_handoff_jobs_7me1tu2n/outgoing` :: wrote /var/folders/60/l2q_ptls0r76nzvtbqldjt4m0000gn/T/ree_v2_handoff_jobs_7me1tu2n/outgoing/commit_dual_error_channels__single_error_stream.json ; `python3 scripts/submit_remote_job.py --job-spec-dir /var/folders/60/l2q_ptls0r76nzvtbqldjt4m0000gn/T/ree_v2_handoff_jobs_7me1tu2n/outgoing --dry-run` :: dry-run submit OK: commit_dual_error_channels__pre_post_split_streams.json -> backend=<not configured> ; `python3 scripts/pull_remote_results.py --job-run-dir /var/folders/60/l2q_ptls0r76nzvtbqldjt4m0000gn/T/ree_v2_handoff_jobs_7me1tu2n/completed --runs-root evidence/experiments --dry-run` :: PASS: no completed remote result bundles in /var/folders/60/l2q_ptls0r76nzvtbqldjt4m0000gn/T/ree_v2_handoff_jobs_7me1tu2n/completed |

## Run-Pack Inventory
| experiment_type | run_id | seed | condition_or_scenario | status | evidence_direction | claim_ids_tested | failure_signatures | execution_mode | compute_backend | runtime_minutes | pack_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| commit_dual_error_channels | 2026-02-14T150300Z_commit-dual-error-channels_seed11_pre_post_split_streams | 11 | pre_post_split_streams | PASS | supports | MECH-060 | none | remote | cloud_gpu_a10g | 63.0 | evidence/experiments/commit_dual_error_channels/runs/2026-02-14T150300Z_commit-dual-error-channels_seed11_pre_post_split_streams |
| jepa_anchor_ablation | 2026-02-14T150100Z_jepa-anchor-ablation_seed11_ema_anchor_on | 11 | ema_anchor_on | PASS | supports | MECH-058 | none | remote | cloud_gpu_a10g | 68.0 | evidence/experiments/jepa_anchor_ablation/runs/2026-02-14T150100Z_jepa-anchor-ablation_seed11_ema_anchor_on |
| jepa_uncertainty_channels | 2026-02-14T150200Z_jepa-uncertainty-channels_seed11_explicit_uncertainty_head | 11 | explicit_uncertainty_head | PASS | supports | MECH-059 | none | remote | cloud_gpu_a10g | 88.0 | evidence/experiments/jepa_uncertainty_channels/runs/2026-02-14T150200Z_jepa-uncertainty-channels_seed11_explicit_uncertainty_head |
| trajectory_integrity | 2026-02-14T150000Z_trajectory-integrity_seed11_trajectory_first_enabled | 11 | trajectory_first_enabled | PASS | supports | MECH-056 | none | remote | cloud_gpu_a10g | 32.0 | evidence/experiments/trajectory_integrity/runs/2026-02-14T150000Z_trajectory-integrity_seed11_trajectory_first_enabled |

## Claim Summary
| claim_id | runs_added | supports | weakens | mixed | unknown | recurring_failure_signatures |
| --- | --- | --- | --- | --- | --- | --- |
| MECH-056 | 1 | 1 | 0 | 0 | 0 | none |
| MECH-058 | 1 | 1 | 0 | 0 | 0 | none |
| MECH-059 | 1 | 1 | 0 | 0 | 0 | none |
| MECH-060 | 1 | 1 | 0 | 0 | 0 | none |

## Open Blockers
- No blocking CI gate failures in this handoff cycle.
- Continue monitoring remote backend readiness for production (dry-run only in bootstrap).

## Local Compute Options Watch
- local_options_last_updated_utc: `2026-02-14T14:43:11Z`
- rolling_3mo_cloud_spend_eur: `0`
- local_blocked_sessions_this_week: `0`
- recommended_local_action: `hold_cloud_only`
- rationale: No spend/blocking pressure above hobby thresholds; keep cloud-first policy.
