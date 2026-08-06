# P5.8 deterministic held-out evaluation

`radjax_student.behavior.evaluation` evaluates only the held-out neutral
`CorridorBatchV1` and `ExemplarBatchV1` against one identity-bound final P5.7
`ExemplarCheckpointV1`. It also requires the linked P5.6 checkpoint, both
training batches, and the complete `BehavioralBatchesV1` P5.4 materialization.
The evaluator binds the P5.4 `BehavioralMaterializationDescriptorV1`, which
derives the split policy/source, complete assignment mapping, and all four
canonical partition coordinate/passport sets at materialization. The immutable
`BehavioralBatchesV1` validates that descriptor on construction, so replacing a
held-out surface with a partial subset cannot produce a supported complete
materialization. Omission, substitution, duplication, or a changed expected
split fails closed before the caller-owned forward function runs.

P5.6 accepts the sealed `BehavioralBatchesV1` itself and requires its exact
training corridor object, validates source/split continuity at entry, and
records the verified descriptor identity in its checkpoint. P5.7 propagates
only that verified identity; P5.8 never accepts an arbitrary materialization
identity string.

The training batches prove no held-out example enters either training surface
and that each training cursor completed its own partition.

Every held-out corridor coordinate and exemplar passport must be unique and is
reported once in canonical UTF-8 order. Corridor interval metrics and exemplar
coarse-cross-entropy metrics are separately reported only after every scalar is
finite. The report binds the final checkpoint, predecessor, Contract/Tome,
receipt, language/HF, behavioral, architecture, split, and policy identities;
exact replay requires the report identity to match.

This boundary takes no optimizer and emits no model or optimizer state. It has
no artifact, archive, locator, tokenizer, architecture-specific, quality,
generalization, package, or readiness behavior.
