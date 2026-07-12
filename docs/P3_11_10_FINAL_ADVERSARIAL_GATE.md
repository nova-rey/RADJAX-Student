# P3.11.10 Final Adversarial Gate Addition

The final adversarial gate includes optimizer step identity as a mandatory
checkpoint condition. It exercises envelope-only tampering, numerical-step
tampering, sidecar-only edits, edits with recomputed sidecar and manifest
hashes, optimizer identity/schema changes, and a non-SGD optimizer with a
different numerical representation.

The accepted blocker vocabulary includes
`checkpoint_optimizer_step_mismatch`. Details contain only expected and
observed integer steps; raw array contents are never reported. The gate also
checks that v3 generic code delegates numerical step interpretation to the
registered optimizer capability and contains no SGD-specific keypath lookup.
