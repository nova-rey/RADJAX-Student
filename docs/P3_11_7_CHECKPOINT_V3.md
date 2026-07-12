# P3.11.7 Checkpoint v3

Checkpoint v3 stores parameters, architecture carry, and optimizer numerical
state in deterministic project-owned ZIP_STORED sidecars. Canonical JSON
descriptors define the pytree structure; NPZ member names are never used to
infer structure.

The optimizer capability owns numerical-state shape, dtype, and step meaning.
Before save and after restore, it must prove that the stable
`OptimizerState.step` envelope equals its own numerical completed-update
counter. A mismatch raises `checkpoint_optimizer_step_mismatch` with expected
and observed steps only. v3 records optimizer identity, capability version,
numerical-state schema, envelope step, sidecar digest, and descriptor digest.

The v2 scalar checkpoint remains read-compatible and is not treated as an HF
distribution checkpoint.
