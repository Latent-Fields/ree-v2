# ree-v2

V2 implementation substrate for REE claim experimentation. Upgraded from synthetic scaffolding (2026-02-26 archive) to a real ree_core implementation as of 2026-03-06.

## V2 Architecture

- **HippocampalModule** (new in V2): terrain-navigated trajectory proposal. Resolves SD-001 — CEM-based trajectory search was misplaced in E2; it is now here, using E2 only as a forward rollout model.
- **E2**: pure fast transition model `f(z_t, a_t) → z_{t+1}`. No longer performs candidate generation.
- **CausalGridWorld**: real environment replacing synthetic data generation.
- **SD-002 resolved (2026-03-06)**: E1 prior wired into HippocampalModule's terrain search (E1→HippocampalModule mutual constitution).
- **SD-003**: `forward_counterfactual()` exposed via E2 for self-attribution. Substrate ready; experiments pending.
- **SD-004 (held for V3)**: Action objects as hippocampal map backbone — E2 produces compressed action-object representations that HippocampalModule maps over instead of raw state space, enabling much longer planning horizons. See `docs/architecture/design_decisions.md`.

## Experiment Status

V2 series complete: 15 experiments run (EXQ-014–EXQ-028) against real ree_core — 6 PASS, 7 FAIL (EXQ-027/028 were SD-003 scoping experiments).
All three V2 hard-stop criteria met; V3 transition formally triggered.
Governance sync complete: results indexed in `REE_assembly/evidence/experiments/` via `sync_v2_results.py` + `build_experiment_indexes.py`.
Full results in `evidence/experiments/`. Historical queue in `experiment_queue.json`.

## Cross-Repo Roundtrip

Run full `ree-v2` <-> `REE_assembly` flow (qualification, handoff sync/ingestion, dispatch emission, dispatch pullback):

```bash
scripts/cross_repo_roundtrip.sh
```

Preview commands without executing:

```bash
scripts/cross_repo_roundtrip.sh --dry-run
```

## License

Apache License 2.0 (see `LICENSE`).

## Citation

- Cite this repository using `CITATION.cff`.
- For canonical architectural attribution, cite Daniel Golden's REE specification in `https://github.com/Latent-Fields/REE_assembly/` (also captured as the preferred citation in `CITATION.cff`).
