# P5.5 — Behavioral Objectives and Scoped SGD Qualification

P5.5 consumes only the neutral P5.4 `CorridorBatchV1` and `ExemplarBatchV1`
values plus Student logits. It receives no artifact, manifest, locator,
delivery name, archive member, Tome path, architecture parameter, or teacher
probability beyond the Contract-declared sparse outcomes.

`corridor_objective_v1` computes entropy, top-1 margin, top-8 mass, top-32
mass, and tail mass from Student softmax probabilities at the declared
coordinates. Each statistic uses a Contract-declared inclusive `[min, max]`
interval and contributes zero inside it or squared outside distance. The
versioned Student policy gives all five statistics weight one and normalizes
each by the same positive attention-mask × Contract-assignment-weight sum.
It reports weighted per-statistic and joint inside-corridor rates.

`exemplar_coarse_cross_entropy_v1` treats every declared active token ID as a
singleton outcome and every unlisted vocabulary token as exactly one aggregate
tail outcome. The Student tail log probability is a log-sum-exp over unlisted
log probabilities. The teacher is only the declared singleton probabilities
and aggregate `tail_mass`; bucket masses are not used numerically and no
unseen-token distribution is inferred.

The existing `SgdOptimizer` is production-qualified for this neutral path. It
remains the sole `sgd.v1` identity and supports JAX execution, layout-derived
scoped updates, resolved schedule values, finite-gradient rejection, stable
envelope serialization, and deterministic replay. Its public verified-update
gate rejects NaN/Inf gradients or an invalid schedule before candidate
parameters or optimizer state can leave the execution boundary. Independent
corridor and exemplar tests each run that complete qualification sequence;
neither uses a mixed synthetic substitute. P5.5 makes no learning
pass, model-quality, checkpoint-production, held-out evaluation, packaging,
or RWKV-specific claim.
