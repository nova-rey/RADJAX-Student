# P5.9 deterministic HF-shaped proof package

`radjax_student.behavior.proof_package` closes the Phase 5 evidence conveyor
with a finite, deterministic UTF-8 JSON package. The package is HF-shaped only
in the narrow sense that it includes `config.json` and the projected tokenizer,
vocabulary, and special-token identity objects. It is not a Transformers
export, model loader, or pretrained artifact.

The builder accepts only a strict-factory-attested P5.3 projection, a selected
neutral `ArchitectureConfig`, the P5.6 corridor checkpoint, P5.7 final
checkpoint, and P5.8 evaluation report. Before emitting bytes it requires the
complete checkpoint/report lineage to agree on Contract and Tome commits,
refreshed fixture receipt, language binding and `hf_language_projection_v1`,
behavioral source/authority/package/composition identities, architecture and
split identities, every pass policy, optimizer identity, materialization and
passport registry identity. Rebuilt or directly constructed P5.3 projections
are rejected through the admission factory's private attestation.

Every file is canonical JSON, inventory-listed with its SHA-256 and byte size.
Package identity derives from that sorted inventory. Replay requires exact
package identity and exact encoded bytes; loading requires the caller's
expected identity, so altered or mismatched lineage bytes fail closed. No
package model, report, or evidence embeds an artifact locator, temporary path,
archive member, or delivery path.

The explicit nonclaims are: general HF export, pretrained loading, Phase 6,
model quality/generalization, and artifact transport/locator behavior.
