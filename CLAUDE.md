# ree-v2

## Git Workflow

Push directly to `main`: `git push origin HEAD:main`

Do NOT create feature branches or pull requests.

## Python

Use `/opt/local/bin/python3` (has torch 2.10.0).

## Multi-Session Coordination

See `REE_Working/CLAUDE.md` for session startup protocol.
Check `REE_Working/WORKSPACE_STATE.md` before editing `experiment_queue.json`.

## Governance

After experiments complete:
```
python scripts/sync_v2_results.py      # from REE_assembly root
python scripts/build_experiment_indexes.py
```
run_id must end `_v2`. architecture_epoch must be `"ree_hybrid_guardrails_v1"`.
