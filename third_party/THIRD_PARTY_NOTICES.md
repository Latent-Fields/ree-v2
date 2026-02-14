# Third-Party Notices

This repository integrates an inference-only JEPA backend for qualification runs.
No JEPA training or fine-tuning is performed in `ree-v2`.

## Pinned Sources

- JEPA code repository: `https://github.com/facebookresearch/vjepa2`
- JEPA code commit pin: `c2963a47433ecca0ad4f06ec28bcfa8cb5b5cefb`
- Commit source URL: `https://github.com/facebookresearch/vjepa2/commit/c2963a47433ecca0ad4f06ec28bcfa8cb5b5cefb`

## Pinned Checkpoint Artifact

- Primary checkpoint: `facebook/vjepa2-vitg-fpc64-256`
- Primary checkpoint revision: `f353acddbcb72f0e7f87e0e7f5bb8f8a1a8cee62`
- Revision source URL: `https://huggingface.co/api/models/facebook/vjepa2-vitg-fpc64-256`
- Primary checkpoint filename: `model.safetensors`
- Primary checkpoint SHA256: `f205e77aa2ade168db6b09d4bc420d156141f64ab964278a9c181a2bdf2a232b`
- Primary checkpoint size bytes: `4138311608`
- Integrity source URL: `https://huggingface.co/facebook/vjepa2-vitg-fpc64-256/blob/main/model.safetensors`
- Primary checkpoint license: `Apache-2.0`

## Alternate Checkpoint (Documented)

- Alternate checkpoint: `facebook/vjepa2-vitl-fpc64-256`
- Alternate checkpoint filename: `model.safetensors`
- Alternate checkpoint SHA256: `25466aef85727d16546c6cf8c99f12fcfad9cbca8225d45f23685e2e025b786b`
- Alternate checkpoint size bytes: `1303947864`
- Integrity source URL: `https://huggingface.co/facebook/vjepa2-vitl-fpc64-256/blob/main/model.safetensors`
- Alternate checkpoint license: `MIT`

## License Notices

- Upstream code license (pinned repo commit): `MIT`
- Primary checkpoint license: `Apache-2.0`
- Effective lock `license_id` (primary artifact): `Apache-2.0`

## Compliance Surfaces

- Lock file: `third_party/jepa_sources.lock.v1.json`
- Attribution/citation: `docs/third_party/JEPA_ATTRIBUTION_AND_CITATION.md`
- Compliance checker: `scripts/check_third_party_compliance.py`
- Checkpoint verifier: `scripts/verify_jepa_checkpoint.py`

## Inference-Only Policy

- `inference_mode`: `inference_only_no_training`
- `optimizer_policy`: `forbidden`
- `gradient_policy`: `no_grad_only`

If required local video/model dependencies are unavailable on macOS smoke runs,
`ree-v2` uses deterministic synthetic-frame fallback while preserving provenance and contract outputs.
