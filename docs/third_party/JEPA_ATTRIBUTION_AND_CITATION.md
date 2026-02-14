# JEPA Attribution And Citation

This document defines attribution and citation requirements for JEPA-backed inference in `ree-v2`.

## Scope

- Integration target: `IMPL-022`
- Contract outputs preserved: `experiment_pack/v1`, `jepa_adapter_signals/v1`, hook registry v1 surfaces
- Lock file: `third_party/jepa_sources.lock.v1.json`
- Notices file: `third_party/THIRD_PARTY_NOTICES.md`

## Pinned Artifacts

- Code source: `facebookresearch/vjepa2`
- Code commit: `c2963a47433ecca0ad4f06ec28bcfa8cb5b5cefb`
- Primary checkpoint: `facebook/vjepa2-vitg-fpc64-256`
- Checkpoint revision: `f353acddbcb72f0e7f87e0e7f5bb8f8a1a8cee62`
- Primary filename: `model.safetensors`
- Primary SHA256: `f205e77aa2ade168db6b09d4bc420d156141f64ab964278a9c181a2bdf2a232b`
- Primary size bytes: `4138311608`
- Primary checkpoint license: `Apache-2.0`
- Alternate checkpoint documented: `facebook/vjepa2-vitl-fpc64-256` (`MIT`)

## Required Citations

- I-JEPA: `arXiv:2301.08243`
- V-JEPA 2: `arXiv:2506.09985`
- JEPA uncertainty extension: `arXiv:2412.10925`

## Citation Template

When publishing or handing off JEPA-backed `ree-v2` evidence, include:

1. repository + commit pin from `third_party/jepa_sources.lock.v1.json`
2. checkpoint repo + revision pin from `third_party/jepa_sources.lock.v1.json`
3. checkpoint hash + size pin (`checkpoint_filename`, `checkpoint_sha256`, `checkpoint_size_bytes`)
4. license IDs (`upstream_license_id`, `checkpoint_license_id`, `license_id`)
5. statement: "Inference-only backend (`torch.no_grad()`), no JEPA fine-tuning/training in `ree-v2`."
6. statement if fallback used: "Deterministic synthetic-frame fallback used for smoke due unavailable local decode/model dependencies."

## Compliance Rules

- Never emit JEPA-backed run packs without `manifest.scenario` provenance fields:
  - `jepa_source_mode`
  - `jepa_source_commit`
  - `jepa_patch_set_hash`
- For `backend=jepa_inference`, also include:
  - `jepa_checkpoint_repo_id`
  - `jepa_checkpoint_revision`
  - `jepa_checkpoint_license_id`
- For real (non-fallback) JEPA runs, require lock-verified checkpoint fields:
  - `jepa_checkpoint_filename`
  - `jepa_checkpoint_sha256`
  - `jepa_checkpoint_size_bytes`
  - `jepa_checkpoint_verified`
- Validate local checkpoint files before strict runs:
  - `python3 scripts/verify_jepa_checkpoint.py --checkpoint-path /absolute/path/to/model.safetensors --variant primary`
- Keep this doc and `third_party/THIRD_PARTY_NOTICES.md` synchronized with lock updates.
