# Third-Party Notices

This repository includes and/or derives interface-compatible behavior from external JEPA research artifacts.

## JEPA Family References

- Upstream reference repository: `https://github.com/facebookresearch/vjepa2`
- Locked upstream commit/id: `internal_snapshot_sha256:0241895c20456d097606d9329fef27f50427cf42c208ac87eb983a52a545bdd5`
- Lock file: `third_party/jepa_sources.lock.v1.json`
- Compatibility target: `IMPL-022`

## License Notice

- Declared license id for referenced JEPA source mode: `CC-BY-NC-4.0`
- Repository operators must verify downstream usage is compatible with this license and intended non-commercial scope.

## Local Patch Set

- `bootstrap_internal_adapter_stub`

## Provenance Statement

`ree-v2` currently uses `source_mode=internal_minimal_impl` for toy qualification harness execution.
The immutable snapshot identifier above pins the internal JEPA-compatible adapter surface used by emitted run packs.
