# P3.7 Generic Learning Loop

P3.7 adds a bounded, deterministic loop over the P3.5 single-step seam.
`BatchSource` produces generic batches; the loop consumes them in order,
collects bounded metrics, applies exact max-step stopping, and can invoke a
checkpoint callback only after a successful step.

The loop contains no architecture math, objective semantics, optimizer logic,
Tome behavior, or checkpoint ownership. Gradient accumulation values other than
one are rejected until true accumulation exists. Source position is serializable
and must be restored for continuation.
