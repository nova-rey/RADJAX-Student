# P3.12B - Hugging Face Descriptor Authority

P3.12B makes `HFCompatibilityDescriptor` the sole Hugging Face lifecycle
authority. Architectures construct the complete descriptor from their validated
configuration, catalog, layout, and initialized parameters. The derived
`HFPreservationReference` is a compact transport projection, not an independent
authority.

`learning_checkpoint.v3` persists both `hf_descriptor.json` and the derived
reference. Continuation restore requires a caller-provided matching descriptor.
Historical reference-only v3 checkpoints are inspection-only and fail modern
continuation restore with `checkpoint_hf_descriptor_missing`.

Reports retain a compact descriptor summary; replay compares descriptor-derived
identity. This checkpoint implements neither HF export nor Transformers,
safetensors, network access, a production architecture, or performance claims.

Current status: P3.12B is locally accepted only after its recorded receipt and
required local gates pass. P3.12C remains next and unstarted. Phase 4 remains
unstarted.

The recorded receipt is
[P3_12B_HF_DESCRIPTOR_AUTHORITY_RECEIPT.json](P3_12B_HF_DESCRIPTOR_AUTHORITY_RECEIPT.json)
under schema `radjax.p3_12b_hf_descriptor_authority.v1`. Its current evidence
digest is `56fafd1705acf611009acab2b2ff5e5a10482b6d4e6d08d23c4a5ac0b94fb919`.
