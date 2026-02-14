# JEPA Attribution And Citation

This document tracks attribution and citation obligations for JEPA-related integration surfaces used in `ree-v2`.

## Scope

- Integration target: `IMPL-022` JEPA E1/E2 substrate contract
- Lock file: `third_party/jepa_sources.lock.v1.json`
- Notices file: `third_party/THIRD_PARTY_NOTICES.md`

## Upstream References

- `facebookresearch/vjepa2`: <https://github.com/facebookresearch/vjepa2>
- I-JEPA paper: `arXiv:2301.08243`
- V-JEPA2 paper/method reference: `arXiv:2506.09985`
- JEPA uncertainty extension reference: `arXiv:2412.10925`

## Citation Guidance

When reporting results that depend on JEPA-compatible substrate behavior, include:

1. the repository and commit/snapshot identifier from `third_party/jepa_sources.lock.v1.json`
2. the three arXiv references listed above
3. explicit statement that `ree-v2` toy qualification in this step is deterministic harness simulation, not full upstream training reproduction

## Compliance Notes

- Keep `license_id` in lock file synchronized with `third_party/THIRD_PARTY_NOTICES.md`.
- Keep run-pack provenance fields (`jepa_source_mode`, `jepa_source_commit`, `jepa_patch_set_hash`) aligned with lock values.
- Update this file when upstream references or licensing posture change.
