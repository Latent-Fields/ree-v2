# Weekly Handoff - ree-v2 - 2026-02-09

## Metadata
- week_of_utc: `2026-02-09`
- producer_repo: `ree-v2`
- producer_commit: `7cf3b09feb088aa29d39e31ef98502236487c764`
- generated_utc: `2026-02-14T17:37:56Z`

## Contract Sync
- ree_assembly_repo: `REE_assembly`
- ree_assembly_commit: `1561369c79d4f2290e843b0a15407f91c72a8ea7`
- contract_lock_path: `contracts/ree_assembly_contract_lock.v1.json`
- contract_lock_hash: `421f756451519e768093161825bf8035231266260e11ee00b3e08caea3b3420e`
- schema_version_set: `experiment_pack/v1, experiment_pack_metrics/v1, hook_registry/v1, jepa_adapter_signals/v1`
- template_path: `/Users/dgolden/Documents/GitHub/REE_assembly/evidence/planning/WEEKLY_HANDOFF_TEMPLATE.md`
- template_sha256: `3995a20b39f04dc468c429d6d95d6f913b919c174beb40ccd3af04a081d5d0ad`

## CI Gates
| gate | status | evidence |
| --- | --- | --- |
| schema_validation | PASS | `python3 scripts/validate_experiment_pack.py --runs-root evidence/experiments` :: PASS: validated 22 run(s), schemas, adapter files, and contract lock hashes |
| seed_determinism | PASS | `python3 scripts/check_seed_determinism.py --profile all --max-abs-delta 1e-6` :: checked commit_dual_error_channels/single_error_stream seed=11 max_abs_delta=0 |
| hook_surface_coverage | PASS | `python3 scripts/validate_hook_surfaces.py --registry contracts/hook_registry.v1.json` :: PASS: hook surface contract verified for HK-001..HK-006 and HK-101..HK-104 |
| remote_export_import | PASS | `python3 scripts/estimate_run_resources.py --profile all --machine macbook_air_m2_2022` :: experiment_type	condition	estimated_runtime_minutes	execution_mode	offload_reason ; `python3 scripts/build_remote_job_spec.py --profile all --out-dir /var/folders/60/l2q_ptls0r76nzvtbqldjt4m0000gn/T/ree_v2_handoff_jobs_vo5nnjp3/outgoing` :: wrote /var/folders/60/l2q_ptls0r76nzvtbqldjt4m0000gn/T/ree_v2_handoff_jobs_vo5nnjp3/outgoing/commit_dual_error_channels__single_error_stream.json ; `python3 scripts/submit_remote_job.py --job-spec-dir /var/folders/60/l2q_ptls0r76nzvtbqldjt4m0000gn/T/ree_v2_handoff_jobs_vo5nnjp3/outgoing --dry-run` :: dry-run submit OK: commit_dual_error_channels__pre_post_split_streams.json -> backend=<not configured> ; `python3 scripts/pull_remote_results.py --job-run-dir /var/folders/60/l2q_ptls0r76nzvtbqldjt4m0000gn/T/ree_v2_handoff_jobs_vo5nnjp3/completed --runs-root evidence/experiments --dry-run` :: PASS: no completed remote result bundles in /var/folders/60/l2q_ptls0r76nzvtbqldjt4m0000gn/T/ree_v2_handoff_jobs_vo5nnjp3/completed |

## Run-Pack Inventory
| experiment_type | run_id | seed | condition_or_scenario | status | evidence_direction | claim_ids_tested | failure_signatures | execution_mode | compute_backend | runtime_minutes | pack_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| commit_dual_error_channels | 2026-02-14T150300Z_commit-dual-error-channels_seed11_pre_post_split_streams | 11 | pre_post_split_streams | PASS | supports | MECH-060 | none | remote | cloud_gpu_a10g | 63.0 | evidence/experiments/commit_dual_error_channels/runs/2026-02-14T150300Z_commit-dual-error-channels_seed11_pre_post_split_streams |
| commit_dual_error_channels | 2026-02-14T164820Z_commit-dual-error-channels_seed11_pre_post_split_streams_toyenv | 11 | pre_post_split_streams | PASS | supports | MECH-060 | mech060:postcommit_channel_contamination,mech060:attribution_reliability_break | remote | cloud_gpu_a10g | 63.0 | evidence/experiments/commit_dual_error_channels/runs/2026-02-14T164820Z_commit-dual-error-channels_seed11_pre_post_split_streams_toyenv |
| commit_dual_error_channels | 2026-02-14T164941Z_commit-dual-error-channels_seed11_pre_post_split_streams_toyenv | 11 | pre_post_split_streams | PASS | supports | MECH-060 | none | remote | cloud_gpu_a10g | 63.0 | evidence/experiments/commit_dual_error_channels/runs/2026-02-14T164941Z_commit-dual-error-channels_seed11_pre_post_split_streams_toyenv |
| commit_dual_error_channels | 2026-02-14T165000Z_commit-dual-error-channels_seed11_pre_post_split_streams_toyenv | 11 | pre_post_split_streams | PASS | supports | MECH-060 | none | remote | cloud_gpu_a10g | 63.0 | evidence/experiments/commit_dual_error_channels/runs/2026-02-14T165000Z_commit-dual-error-channels_seed11_pre_post_split_streams_toyenv |
| commit_dual_error_channels | 2026-02-14T165932Z_commit-dual-error-channels_seed11_pre_post_split_streams_toyenv_jepa_inference | 11 | pre_post_split_streams | PASS | supports | MECH-060 | none | remote | cloud_gpu_a10g | 63.0 | evidence/experiments/commit_dual_error_channels/runs/2026-02-14T165932Z_commit-dual-error-channels_seed11_pre_post_split_streams_toyenv_jepa_inference |
| jepa_anchor_ablation | 2026-02-14T150100Z_jepa-anchor-ablation_seed11_ema_anchor_on | 11 | ema_anchor_on | PASS | supports | MECH-058 | none | remote | cloud_gpu_a10g | 68.0 | evidence/experiments/jepa_anchor_ablation/runs/2026-02-14T150100Z_jepa-anchor-ablation_seed11_ema_anchor_on |
| jepa_anchor_ablation | 2026-02-14T164820Z_jepa-anchor-ablation_seed11_ema_anchor_on_toyenv | 11 | ema_anchor_on | PASS | supports | MECH-058 | mech058:ema_drift_under_shift | remote | cloud_gpu_a10g | 68.0 | evidence/experiments/jepa_anchor_ablation/runs/2026-02-14T164820Z_jepa-anchor-ablation_seed11_ema_anchor_on_toyenv |
| jepa_anchor_ablation | 2026-02-14T164941Z_jepa-anchor-ablation_seed11_ema_anchor_on_toyenv | 11 | ema_anchor_on | PASS | supports | MECH-058 | none | remote | cloud_gpu_a10g | 68.0 | evidence/experiments/jepa_anchor_ablation/runs/2026-02-14T164941Z_jepa-anchor-ablation_seed11_ema_anchor_on_toyenv |
| jepa_anchor_ablation | 2026-02-14T165000Z_jepa-anchor-ablation_seed11_ema_anchor_on_toyenv | 11 | ema_anchor_on | PASS | supports | MECH-058 | none | remote | cloud_gpu_a10g | 68.0 | evidence/experiments/jepa_anchor_ablation/runs/2026-02-14T165000Z_jepa-anchor-ablation_seed11_ema_anchor_on_toyenv |
| jepa_anchor_ablation | 2026-02-14T165932Z_jepa-anchor-ablation_seed11_ema_anchor_on_toyenv_jepa_inference | 11 | ema_anchor_on | PASS | supports | MECH-058 | none | remote | cloud_gpu_a10g | 68.0 | evidence/experiments/jepa_anchor_ablation/runs/2026-02-14T165932Z_jepa-anchor-ablation_seed11_ema_anchor_on_toyenv_jepa_inference |
| jepa_uncertainty_channels | 2026-02-14T150200Z_jepa-uncertainty-channels_seed11_explicit_uncertainty_head | 11 | explicit_uncertainty_head | PASS | supports | MECH-059 | none | remote | cloud_gpu_a10g | 88.0 | evidence/experiments/jepa_uncertainty_channels/runs/2026-02-14T150200Z_jepa-uncertainty-channels_seed11_explicit_uncertainty_head |
| jepa_uncertainty_channels | 2026-02-14T164820Z_jepa-uncertainty-channels_seed11_explicit_uncertainty_head_toyenv | 11 | explicit_uncertainty_head | PASS | supports | MECH-059 | mech059:uncertainty_metric_gaming_detected | remote | cloud_gpu_a10g | 88.0 | evidence/experiments/jepa_uncertainty_channels/runs/2026-02-14T164820Z_jepa-uncertainty-channels_seed11_explicit_uncertainty_head_toyenv |
| jepa_uncertainty_channels | 2026-02-14T164941Z_jepa-uncertainty-channels_seed11_explicit_uncertainty_head_toyenv | 11 | explicit_uncertainty_head | PASS | supports | MECH-059 | mech059:uncertainty_metric_gaming_detected | remote | cloud_gpu_a10g | 88.0 | evidence/experiments/jepa_uncertainty_channels/runs/2026-02-14T164941Z_jepa-uncertainty-channels_seed11_explicit_uncertainty_head_toyenv |
| jepa_uncertainty_channels | 2026-02-14T165000Z_jepa-uncertainty-channels_seed11_explicit_uncertainty_head_toyenv | 11 | explicit_uncertainty_head | PASS | supports | MECH-059 | none | remote | cloud_gpu_a10g | 88.0 | evidence/experiments/jepa_uncertainty_channels/runs/2026-02-14T165000Z_jepa-uncertainty-channels_seed11_explicit_uncertainty_head_toyenv |
| jepa_uncertainty_channels | 2026-02-14T165932Z_jepa-uncertainty-channels_seed11_explicit_uncertainty_head_toyenv_jepa_inference | 11 | explicit_uncertainty_head | PASS | supports | MECH-059 | none | remote | cloud_gpu_a10g | 88.0 | evidence/experiments/jepa_uncertainty_channels/runs/2026-02-14T165932Z_jepa-uncertainty-channels_seed11_explicit_uncertainty_head_toyenv_jepa_inference |
| trajectory_integrity | 2026-02-14T150000Z_trajectory-integrity_seed11_trajectory_first_enabled | 11 | trajectory_first_enabled | PASS | supports | MECH-056 | none | remote | cloud_gpu_a10g | 32.0 | evidence/experiments/trajectory_integrity/runs/2026-02-14T150000Z_trajectory-integrity_seed11_trajectory_first_enabled |
| trajectory_integrity | 2026-02-14T164820Z_trajectory-integrity_seed11_trajectory_first_enabled_toyenv | 11 | trajectory_first_enabled | PASS | supports | MECH-056 | none | remote | cloud_gpu_a10g | 32.0 | evidence/experiments/trajectory_integrity/runs/2026-02-14T164820Z_trajectory-integrity_seed11_trajectory_first_enabled_toyenv |
| trajectory_integrity | 2026-02-14T164941Z_trajectory-integrity_seed11_trajectory_first_enabled_toyenv | 11 | trajectory_first_enabled | PASS | supports | MECH-056 | none | remote | cloud_gpu_a10g | 32.0 | evidence/experiments/trajectory_integrity/runs/2026-02-14T164941Z_trajectory-integrity_seed11_trajectory_first_enabled_toyenv |
| trajectory_integrity | 2026-02-14T165000Z_trajectory-integrity_seed11_trajectory_first_enabled_toyenv | 11 | trajectory_first_enabled | PASS | supports | MECH-056 | none | remote | cloud_gpu_a10g | 32.0 | evidence/experiments/trajectory_integrity/runs/2026-02-14T165000Z_trajectory-integrity_seed11_trajectory_first_enabled_toyenv |
| trajectory_integrity | 2026-02-14T165932Z_trajectory-integrity_seed11_trajectory_first_enabled_toyenv_jepa_inference | 11 | trajectory_first_enabled | PASS | supports | MECH-056 | none | remote | cloud_gpu_a10g | 32.0 | evidence/experiments/trajectory_integrity/runs/2026-02-14T165932Z_trajectory-integrity_seed11_trajectory_first_enabled_toyenv_jepa_inference |
| trajectory_integrity | 2026-02-14T170702Z_trajectory-integrity_seed11_trajectory_first_enabled_toyenv_jepa_inference | 11 | trajectory_first_enabled | PASS | supports | MECH-056 | none | remote | cloud_gpu_a10g | 32.0 | evidence/experiments/trajectory_integrity/runs/2026-02-14T170702Z_trajectory-integrity_seed11_trajectory_first_enabled_toyenv_jepa_inference |
| trajectory_integrity | 2026-02-14T171300Z_trajectory-integrity_seed11_trajectory_first_enabled_toyenv_jepa_inference | 11 | trajectory_first_enabled | PASS | supports | MECH-056 | none | remote | cloud_gpu_a10g | 32.0 | evidence/experiments/trajectory_integrity/runs/2026-02-14T171300Z_trajectory-integrity_seed11_trajectory_first_enabled_toyenv_jepa_inference |

## Claim Summary
| claim_id | runs_added | supports | weakens | mixed | unknown | recurring_failure_signatures |
| --- | --- | --- | --- | --- | --- | --- |
| MECH-056 | 7 | 7 | 0 | 0 | 0 | none |
| MECH-058 | 5 | 5 | 0 | 0 | 0 | none |
| MECH-059 | 5 | 5 | 0 | 0 | 0 | mech059:uncertainty_metric_gaming_detected |
| MECH-060 | 5 | 5 | 0 | 0 | 0 | none |

## Open Blockers
- none

## Local Compute Options Watch
- local_options_last_updated_utc: `2026-02-14T14:43:11Z`
- rolling_3mo_cloud_spend_eur: `0`
- local_blocked_sessions_this_week: `0`
- recommended_local_action: `hold_cloud_only`
- rationale: No spend/blocking pressure above hobby thresholds; keep cloud-first policy.
- jepa_runs_total: `6`
- jepa_real_runs: `0`
- jepa_synthetic_fallback_runs: `6`
- jepa_real_verified_runs: `0`
- jepa_real_unverified_runs: `0`
