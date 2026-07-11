# P3.9 Synthetic End-to-End Learning Smoke

P3.9 proves the accepted generic learning stack on one deterministic offline
problem: `y = 2x + 1`, for `x = -2, -1, 0, 1, 2` and corresponding targets
`-3, -1, 1, 3, 5`. It is an integration proof, not a model-quality claim.

The reference `synthetic_linear_v1` architecture has exactly two scalar
parameters: `trunk.weight` and `head.bias`. Its forward calculation is
`trunk.weight * x + head.bias`; its named regions are `whole_student`, `trunk`,
and `head`; and the objective surface is `final_output`. The MSE objective and
its scalar gradients flow through the P3.5 objective seam. The accepted `sgd.v1`
backend uses a fixed positive learning rate and no momentum or schedule.

The smoke executes whole-student, trunk-only, and head-only runs. It proves the
whole run reduces loss by at least half, targeted runs preserve excluded values
and per-parameter optimizer state, and all successful runs attach a P3.8C run
report. Retained metrics include loss, gradient norm, parameter norm, learning
rate, and changed/unchanged parameter counts. A deterministic observer-only
hook records lifecycle observations and emits `synthetic.hook_observed`.

Checkpoint evidence uses the P3.6 layered checkpoint for learning state,
architecture parameters, and optimizer state, plus a validated source-state
sidecar owned by this smoke's batch-source continuation boundary. Restore first
validates every component and then constructs a new source, so a failed restore
does not mutate a destination. Resume matches an uninterrupted 12-step run
exactly. Hash corruption, incompatible architecture/optimizer identity, and a
missing source state are rejected.

The runner repeats the whole-student configuration from an identical initial
state and compares parameters, optimizer state, retained metrics, hook events,
checkpoint receipts, reports, and deterministic receipt serialization. The
top-level schema is `radjax.p3_9_synthetic_learning_smoke.v1`; it emits the
stable `p3_9_*` blockers defined by the implementation without tracebacks.

Run it with:

```bash
PYTHONPATH=src python3 -m radjax_student.learning.synthetic_smoke
PYTHONPATH=src python3 -m radjax_student.learning.synthetic_smoke --json
```

P3.9 makes no claim about model quality, real architecture support, Tome
training, language modeling, distributed training, accelerator performance,
production hyperparameters, evaluation, or generalization.
