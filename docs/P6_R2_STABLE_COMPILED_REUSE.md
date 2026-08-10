# P6.R2 — Stable compiled execution reuse

P6.R2 moves compilation ownership into the architecture-neutral generic JAX
execution seam. A loop executor builds one stable learning kernel for each
prepared lifecycle signature and supplies it to the runtime. The runtime keeps
a bounded in-process cache of prepared executable handles keyed by the
authoritative prepared-execution digest. Handles are opaque runtime state and
are never serialized into checkpoints.

The prepared identity includes the declared execution contract and the actual
argument shape/dtype/pytree signature. Consequently, compatible calls reuse the
same compiled execution, while shape, static-value, objective, placement,
backend, donation, or compilation-option changes resolve to a distinct
specialization or fail closed. Batch provenance is validated before a
computation-only view is passed to JAX, so evidence metadata cannot create
spurious specializations.

Focused evidence covers fake-backend compile-once/recompile-on-specialization
cases and a real RWKV lifecycle numerical comparison between eager and stable
JIT execution. The exact final-source T4 requalification is recorded in
[P6_R2_T4_REQUALIFICATION.json](P6_R2_T4_REQUALIFICATION.json) and the two
normal-writer raw receipts beneath `docs/P6_R2_RAW/`: corridor 2,048/2,048 and
exemplar 2,112/2,112, each with 256/264 checkpoints and one compilation event.
The frozen corpus, schedule, checkpoint cadence, and resource envelope are
unchanged. Corridor and exemplar host peaks were 17,263,288,320 and
17,694,294,016 bytes respectively, below the 23,284,565,606-byte ceiling;
device peaks remained 1,275,392 and 1,258,240 bytes against the frozen
11,727,028,224-byte limit.

No global cache clearing, process restart, eager burn, precision change,
architecture-specific branch, or checkpoint/evidence weakening is part of this
repair.
