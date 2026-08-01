# P5.2 Validated Payload Access and Passport Preservation

P5.2 adds the artifact-owned decoding layer for the exact
`native_v3_student_v4` profile. It consumes an already admitted
`NativeV3StudentConsumptionView`; it never discovers paths, imports Tome, or
uses a filename as semantic authority.

Every resource open delegates to Contract's
`open_verified_student_resource_v4(artifact, resource_id, strict=...)`. That
operation re-admits the artifact, resolves the declared resource ID, and
rechecks raw SHA-256 and size immediately before yielding bytes. Student then
accepts only the Contract-declared JSON or NPZ encoding, with `allow_pickle`
disabled for NPZ. Parsed arrays are copied and read-only.

The immutable payload view preserves each resource's ID, role, instance,
semantic digest, raw digest/size, encoding, classification, and consumption
metadata. It exposes the Contract-declared target shard, example registry,
corridor mode table, packed assignments, selected passport index, selected
exemplar payload records, observed corridor statistics, and row-range
declaration. Passport and payload records remain complete frozen mappings, so
selected identity, source linkage, score/token facts, corridor mode,
fingerprint lineage, and source delivery path are never reconstructed from
array order or paths.

Contract owns rejection of malformed masks/positions/tokens, duplicate
resource/role identity, unsafe or mismatched inventory bindings, encoding
failures, passport/exemplar/corridor linkage failures, and invalid evidence.
Student does not duplicate or weaken that validation. Delivery receipt and
authority reference remain Contract-validated provenance/evidence, not loss
inputs; delivery path does not select target semantics.

This checkpoint does not collate, slice, shuffle, pad, convert to JAX,
construct a `LearningBatch`, choose a loss, run either pass, or execute a
checkpoint. Those remain P5.3 and later ownership.
