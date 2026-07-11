# P3.5 Single Learning Step

P3.5 composes the completed generic contracts into exactly one deterministic
scalar learning step. `radjax_student.steps.learning_step()` validates a generic
batch, calls the architecture forward boundary, evaluates a supplied scalar
objective, applies its stable-path gradients through the optimizer, reports
metrics, and advances learning and optimizer state once.

The proof uses the P3.2 fake architecture and P3.3 scalar SGD backend with a
tiny `y = 2x + 1` objective. Whole-student and named-region selection prove that
selected paths update while excluded parameter values and per-path optimizer
state remain unchanged.

This is the first proof of learning mechanics, not behavior quality. It does
not load Tome data, implement an epoch or loop, checkpoint, distribute work, or
provide an RWKV, Mamba, or transformer model.
