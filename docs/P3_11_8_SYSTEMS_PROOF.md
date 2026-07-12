# P3.11.8 Stateful Systems Proof

The P3.11.8 systems proof carries the optimizer step invariant through the
complete JAX lifecycle. A continuation checkpoint is valid only when the
stable `OptimizerState.step` envelope and the optimizer-owned numerical step
identity describe the same completed-update count.

The proof must show, in the same run:

- initialization validates the optimizer-owned numerical state;
- every runtime step validates state before and after the update;
- checkpoint save validates the state before writing v3 payloads;
- restore validates the sidecar, descriptor, identity, schema, and step
  consistency before returning a checkpoint;
- uninterrupted and resumed execution produce identical final envelope and
  numerical step identities.

Generic checkpoint code never assumes a numerical path such as
`arrays["step"]`. The optimizer capability owns numerical-state structure,
dtypes, shapes, and step meaning. The v3 manifest records the optimizer ID,
capability version, numerical-state schema, envelope step, sidecar digest, and
descriptor digest. Any mismatch is fatal and is never repaired by choosing one
step value over another.

This is systems evidence only. It does not claim production architecture
quality, Tome training, Hugging Face export, accelerator-scale training, or
model performance.
