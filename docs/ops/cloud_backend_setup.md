# Cloud Backend Setup

## Goal
Provide provider-agnostic remote execution for qualification runs routed off local hardware.

## Required capabilities
- accept `ree_remote_job_spec/v1` JSON specs from `jobs/outgoing/`
- execute job payload pinned by:
  - source commit
  - `contracts/ree_assembly_contract_lock.v1.json` hash
  - `third_party/jepa_sources.lock.v1.json` hash
- return run packs without mutation to `jobs/completed/<experiment_type>__<run_id>/`

## Recommended bootstrap backend contract
1. queue worker reads JSON job specs.
2. worker checks environment image and source commit.
3. worker runs profile/condition seeds and writes run bundles.
4. worker uploads artifacts into `jobs/completed/`.
5. local operator runs pull script and re-validates packs.

## Environment variables
- `REE_REMOTE_BACKEND` (example: `local_queue`, `aws_batch`, `gcp_batch`, `modal`)
- `REE_REMOTE_PROJECT`
- `REE_REMOTE_REGION`
- `REE_REMOTE_QUEUE`
- `REE_REMOTE_CREDENTIAL_PROFILE`

## Dry-run behavior
`submit_remote_job.py --dry-run` validates job-spec schema and prints dispatch targets without sending jobs.

## Security and provenance
- do not execute remote jobs with mutable dependency tags
- pin container/image digests for deterministic replay audits
- keep credential scope limited to queue + artifact storage paths
