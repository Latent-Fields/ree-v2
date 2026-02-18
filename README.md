# ree-v2

Qualification harness for REE substrate claims with contract-compatible experiment packs for `REE_assembly` ingestion.

Primary scope in bootstrap:
- claim coverage: `MECH-056`, `MECH-058`, `MECH-059`, `MECH-060`
- interface compatibility: `experiment_pack/v1`, `hook_registry/v1`, `IMPL-022`
- qualification lane policy with explicit local-vs-remote compute placement

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

Third-party JEPA licensing and attribution surfaces are tracked in:

- `third_party/THIRD_PARTY_NOTICES.md`
- `docs/third_party/JEPA_ATTRIBUTION_AND_CITATION.md`
- `third_party/jepa_sources.lock.v1.json`

## Citation

- Cite this repository using `CITATION.cff`.
- For canonical architectural attribution, cite Daniel Golden's REE specification in `https://github.com/Latent-Fields/REE_assembly/` (also captured as the preferred citation in `CITATION.cff`).
