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

P3.12B.1 replaces the initial six-adversary receipt with the canonical
77-experiment literal descriptor-authority gate. Each experiment reconstructs
a valid baseline, changes a public input, invokes its public boundary twice,
and records actual observed failure evidence. The gate has 22 named positive
proofs and rejects incomplete inventory execution.

Current status: P3.12B is locally accepted only after the v2 recorded receipt
and required local gates pass. P3.12C remains next and unstarted. Phase 4
remains unstarted.

The recorded receipt is
[P3_12B_HF_DESCRIPTOR_AUTHORITY_RECEIPT.json](P3_12B_HF_DESCRIPTOR_AUTHORITY_RECEIPT.json)
under schema `radjax.p3_12b_hf_descriptor_authority.v2`. Its current evidence
digest is `bf80c4b0d325f414f40a9cd4a9329e11acd4fe50cf193f708e8c683b412ff0bb`.
