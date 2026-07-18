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

P3.12B.2 makes the source wiring itself recorded evidence. A JAX-free AST
audit binds the ordered 22-positive inventory and all 77 registered literal
adversaries to individual source digests, rejects expected-to-observed
translation helpers, and is included as a typed receipt field rather than a
decorative aggregate hash. Checkpoint, replay, and report descriptor drift are
observed at their respective owning validation boundaries.

P3.12B.3 closes the frozen anti-cheat contract: the typed audit proves the
exact ordered 22-positive and 77-adversarial inventories, named experiment
source evidence, observer independence, callable-derived observed boundaries,
receipt authority, and one-way production-import purity. There is no permissive
blocker-family translation and production code has no gate dependency.

Current status: P3.12B is locally accepted after the v2 recorded receipt and
required local gates pass. P3.12C is locally accepted with one production
lifecycle assembler; P3.12D remains next and unstarted. Phase 4 remains
unstarted.

The recorded receipt is
[P3_12B_HF_DESCRIPTOR_AUTHORITY_RECEIPT.json](P3_12B_HF_DESCRIPTOR_AUTHORITY_RECEIPT.json)
under schema `radjax.p3_12b_hf_descriptor_authority.v2`. Its current evidence
digest is `c2de2a9a5de8060cef347bb37203e01348a9d3e6174db8bcef60c7c8558944c4`.
