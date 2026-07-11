# P3.8B Learning Loop Hook Integration

The generic loop now emits immutable hook contexts at loop start, batch receipt,
step start/end, successful checkpoint, and loop end. It uses only the P3.8A
dispatcher, merges returned metrics and warnings, and carries disabled-hook IDs
for the duration of the run.

Fail-fast hook blockers stop the loop before the next core action. Hooks still
receive no parameter, gradient, optimizer, architecture, runtime, or checkpoint
payloads, so they remain observer-only.

P3.8C run reporting and the P3.8 completion gate remain pending.

## Failure and Terminal Events

`failure` is emitted for learning-step and checkpoint exceptions; `checkpoint`
is emitted only after successful creation. Checkpoint and loop-end hook blockers
are respected, while core failure reasons remain distinct from hook failures.

Failure hooks preserve their own blockers and warnings without replacing the
core learning-step or checkpoint failure reason.
