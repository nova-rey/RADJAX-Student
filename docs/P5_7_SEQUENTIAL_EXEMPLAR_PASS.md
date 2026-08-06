# P5.7 sequential exemplar pass

`radjax_student.behavior.exemplar_pass` accepts only a P5.6
`CorridorCheckpointV1` and a training `ExemplarBatchV1`. It validates the exact
predecessor identity and every continued Contract/Tome/receipt, language/HF,
behavioral authority/source, architecture, split, corridor policy/reduction,
ordering, and batching identity before using the predecessor model and optimizer
state. It then executes the declared coarse exemplar objective once and emits
the final `exemplar_v1` checkpoint with an independent exemplar cursor.

Exemplar passports must remain in the P5.4 canonical UTF-8
example/position/fingerprint order. Reordered passports, held-out batches,
mixed objectives, changed continuity authority, foreign optimizer state, and
replay identity changes fail closed. Exact replay proves identical model state,
optimizer state, and final checkpoint identity.

This boundary has no artifact, archive, locator, tokenizer, or
architecture-specific behavior. It makes no held-out evaluation, package,
quality, distributed, or production-readiness claim.
