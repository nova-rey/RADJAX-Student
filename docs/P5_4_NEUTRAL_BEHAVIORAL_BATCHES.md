# P5.4 — Neutral Behavioral Batch Materialization

`radjax_student.behavior` is the terminal boundary for the immutable P5.3
authority projection. It accepts no artifact, Contract descriptor, resource
locator, physical delivery name, or producer path; it emits only neutral
tensors, stable coordinates, partition membership, declared sparse targets,
and immutable passports.

`BehaviorSplitPolicyV1` partitions stable `example_id` values in UTF-8 byte
order. It requires at least two unique exemplar-bearing IDs, reserves the
first for training and the last for held-out evaluation, then allocates every
remaining ID to the smaller partition (training on ties). Its SHA-256 identity
binds its ID, rule version, behavioral-source identity, and complete mapping.
Every coordinate, assignment, passport, and exemplar follows its example ID.

For Tome `8508b1351d0ed8d6a3a14049e4d6f8a849c33cf1`'s refreshed fixture, both
the directory and `student.tgz` materialized the same split identity:
`sha256:4cf616a35fe3af7e6fcb46975853a439b80f89bfc53991663cfd11d3264f8dad`.
The literal mapping is:

- training: `corpus_000000001`, `corpus_000000002`
- held-out: `corpus_000000003`, `corpus_000000004`

The training partition has one exemplar and the held-out partition has two;
both partitions have nonempty corridor assignments. Multipart tensors, mode
IDs, all five Contract-declared per-mode statistic intervals (`min`, `mean`,
`max`), and weights are checked before batching. The complete immutable mode
declarations and typed statistic bounds remain mapped to the batch mode IDs.
Selected payloads must join
one-for-one to passports, coordinates must lie inside the declared target
tensors, and active sparse distributions must be finite and nonnegative.
Arrays are copied read-only and record mappings are recursively frozen.

The integration test opens the actual committed Tome directory and archive
deliveries through the public P5.3 projection. It asserts the literal split and
that every corridor coordinate, selected passport, and materialized selected
payload stays in the corresponding partition in both transports.

This checkpoint adds neither an objective, optimizer, learning pass,
checkpoint, architecture assumption, nor held-out evaluation.
