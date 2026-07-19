# P4.5 RWKV-7 Generic Learning Lifecycle

P4.5 integrates the existing executable RWKV-7 reference plugin with the
generic P3.12C lifecycle assembler and P3.12D runtime-bound callable. A test
fixture constructs an `ArchitectureRegistry`, explicitly registers the RWKV
plugin, supplies that registry to `JaxLearningAssemblyRegistries`, and executes
the learning-composed `radjax.learning.generic_jax_step` callable. No RWKV
registration or branch was added to `learning/composition.py`.

## Generic sparse categorical cross-entropy

`radjax.objective.sparse_cross_entropy`, version 1, is registered beside MSE.
It accepts an architecture-owned logits surface `[B,T,V]` and requires exactly
integer `targets.token_ids` `[B,T]` in `[0,V)`. It computes mean token negative
log likelihood and these finite metrics:

- `objective.sparse_cross_entropy`
- `objective.token_accuracy`

The objective deliberately has no masking, ignore-index, label smoothing,
token weighting, or architecture-specific mode. Its only configuration is
`{"reduction": "mean"}`. This is architecture-neutral: any JAX plugin that
owns a logits surface and token-ID target surface can select it.

## RWKV lifecycle binding

The RWKV plugin now validates the declared finite-JSON tiny-domain input:
`inputs.token_ids` is one `[1,4]` integer sequence in vocabulary `[0,16)`.
It exposes its existing final logits surface to the generic objective. Its
initialization result supplies the pre-existing generic `architecture_carry.v1`
identity with the static digest of its three persistent carry leaves; the
RWKV-owned source descriptor remains separate. This is result conformance to
the already-existing lifecycle contract, not a generic framework change.

Focused tests execute the real assembled eager and JIT steps, prove finite
loss/metrics/gradients, parameter movement, optimizer and learning-state
advance, runtime-key invocation advance, persistent-carry advance, and a later
step accepting the carry returned by the prior step. Eager and JIT parameter,
carry, loss, and callable structures agree within float32 tolerance.

The source-dependent P3.5 audit and P3.12C/D typed receipts were regenerated
from this source tree and byte-compared across fresh generation. Their frozen
positive/adversarial inventories remain unchanged; no foundation fixture was
made RWKV-specific.

## Frozen gradient boundary

- Gradients differentiate through multiple token positions within one fixture sequence.
- Carry returned from one learning step can seed a later learning step.
- Carry crossing separate learning-step boundaries is stop-gradient state.
- Full cross-step BPTT, truncated-BPTT scheduling, and long-context recurrent training are not proven.

The focused proof mutates token positions one and three independently and
observes different finite parameter gradients. It differentiates both the loss
and returned carry with respect to incoming carry and observes zero leaves at
the learning-step boundary. This does not claim a training recipe, full
cross-step BPTT, truncated-BPTT scheduling, long-context recurrent training,
initialization parity, HF conversion, pretrained weights, model quality,
performance, multi-device/TPU execution, teacher/Tome/distillation, a second
architecture, or Phase 5 behavior.

## Generic-change ledger

The Phase 4 generic ledger contains exactly these three approved changes:

1. Sparse categorical cross-entropy objective — token-logit architectures need
   integer targets; objectives owns the neutral `[B,T,V]` / `targets.token_ids`
   contract, which a future Transformer or Mamba-style JAX plugin can use.
2. Runtime-owned initialization-key materializer — runtime owns derivation of
   executable initialization entropy for any JAX architecture.
3. Architecture-neutral request seam for runtime-supplied initialization
   material — learning composes the runtime value and any JAX architecture
   consumes it without importing runtime or inventing seed authority.

No additional generic change was made in P4.5.
