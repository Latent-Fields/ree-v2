# ree-v2

> **ARCHIVED — SYNTHETIC SCAFFOLDING ONLY (2026-02-26)**
> This repo generates parametric synthetic data, not real experimental measurements.
> All PASS/FAIL results it produced are unreliable as evidence for REE claims.
> The active experimental substrate is `ree-v1-minimal`.
> See `REE_assembly/evidence/experiments/INDEX.md` for details.

Qualification harness for REE substrate claims with contract-compatible experiment packs for `REE_assembly` ingestion.

Primary scope in bootstrap:
- claim coverage: `MECH-056`, `MECH-058`, `MECH-059`, `MECH-060`
- interface compatibility: `experiment_pack/v1`, `hook_registry/v1`, `IMPL-022`
- qualification lane policy with explicit local-vs-remote compute placement
- active bridge hooks `HK-101..HK-104` plus cognifold trace artifacts (`TaskLoopObject` and `CouplingGraph`)

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
