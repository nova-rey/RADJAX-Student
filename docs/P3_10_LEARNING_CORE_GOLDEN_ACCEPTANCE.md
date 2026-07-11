# P3.10 Learning Core Golden Acceptance

P3.10 closes Phase 3 by independently auditing the accepted P3.1 through P3.9
public seams. It adds no learning behavior and does not create a second loop,
optimizer, checkpoint, hook, or reporting implementation.

The immutable receipt schema is
`radjax.p3_10_learning_core_acceptance.v1`. Its eleven validity sections cover
contract vocabulary, optimizer mechanics, single-step evidence, loop evidence,
the v2 checkpoint contract, source-state resume equivalence, P3.8
observability, P3.9 synthetic learning, deterministic replay, documentation,
and AST-based test inventory. P3.8 and P3.9 receipts are downstream evidence,
not substitutes for independent P3.2, P3.5, P3.6, or P3.7 audits. Each section
is dependency-injected so adversarial tests can corrupt an accepted seam or
raise an unexpected exception; the gate fails closed with structured blockers
and no tracebacks. The P3.10.1 scenario matrix contains 78 named tests with
real tamper coverage for optimizer state and schedules, scoped single-step
state, loop termination and checkpoint cadence, and checkpoint hashes, sizes,
ownership, schema, and source state.

Run the gate with:

```bash
PYTHONPATH=src python3 -m radjax_student.learning.p3_10_acceptance
PYTHONPATH=src python3 -m radjax_student.learning.p3_10_acceptance --json
```

Exit status is zero only for a passing receipt. The gate claims contract and
synthetic systems evidence, not model quality, real architecture support, Tome
training, language modeling, distributed or accelerator performance,
production hyperparameters, evaluation, or generalization.
