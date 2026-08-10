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
JIT execution. The T4 reduced-burn requalification is recorded separately by
the normal `scripts/measure_p6_6_reduced_burn.py` writer after the final repair
commit; its frozen corpus, schedule, checkpoint cadence, and resource envelope
are unchanged.

No global cache clearing, process restart, eager burn, precision change,
architecture-specific branch, or checkpoint/evidence weakening is part of this
repair.
