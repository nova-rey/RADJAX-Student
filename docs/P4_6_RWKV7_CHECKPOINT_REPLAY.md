# P4.6 RWKV-7 Checkpoint, Restore, and Replay

P4.6 proves that the existing generic v3 checkpoint format persists a trained
RWKV-7 reference lifecycle without an RWKV format fork or a checkpoint branch.
The proof executes one real sparse-CE eager step through the P3.12C/P3.12D
assembly, saves `JaxLearningLifecycle.checkpoint()` using
`save_learning_checkpoint_v3`, restores it into a fresh identical assembly, and
continues both source and restored lifecycles with the same next finite-JSON
sequence.

## Preserved identity and state

The v3 manifest and typed checkpoint preserve the architecture ID, architecture
state ID, parameter-layout digest, parameter-catalog digest, configuration
digest, carry descriptor, HF descriptor, and descriptor-derived HF reference.
The RWKV plugin version is preserved in the existing HF descriptor. V3 stores
the architecture configuration by digest, not a second configuration payload.

The focused proof establishes equality of restored parameters, persistent carry,
optimizer numerical arrays/envelope, learning state, layout, state, carry
descriptor, HF descriptor/reference, and direct tiny-domain logits plus returned
carry. The source and restored generic loop then produce equal next-step loss,
parameters, carry, optimizer state, and learning state within the declared
float32 tolerance.

## Current-owner rejection and callable boundary

The existing checkpoint owner rejects foreign expected configuration, layout,
HF reference/descriptor, and carry descriptor identities. No rejection code was
changed or added. The generic runtime callable identity is compared at the
assembled summary and execution boundaries; it is not serialized as a v3
checkpoint component, so P4.6 does not claim checkpoint-persisted callable
identity.

No generic change was made. V3 already supports mapping pytrees for parameters,
carry, and optimizer state. P4.6 does not claim a new format, full BPTT,
training-recipe parity, cross-hardware replay, HF conversion/export, pretrained
weights, quality, performance, multi-device/TPU execution, teacher/Tome work,
a second architecture, or Phase 5 behavior.
