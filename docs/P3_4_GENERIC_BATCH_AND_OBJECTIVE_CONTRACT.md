# P3.4 Generic Batch and Objective Contract

P3.4 freezes the model-neutral vocabulary for what learning consumes and what
it is asked to optimize. The API remains in `radjax_student.learning`.

`LearningBatch` is a generic envelope of inputs, targets, weights, metadata,
and an independent objective scope. `BatchMetadata` records generic cardinality,
sequence, padding, mask, and source facts. Neither model names Tome corridors,
exemplars, teacher logits, or architecture tensors.

`ObjectiveRequest` describes an objective ID, scope, batch reference, required
outputs, and `WeightingPolicy`. `ObjectiveResult` records an evaluated loss,
components, metrics, warnings, and non-claims. They provide no evaluation
implementation.

Behavior adapters may later populate generic batches from a validated Tome, but
they must not redefine the batch contract. Architecture owns forward outputs,
runtime owns execution, and learning will own weighting interpretation and
objective invocation.

P3.4 does not load a Tome, compute teacher logits or gradients, execute an
objective, or run a learning step.
