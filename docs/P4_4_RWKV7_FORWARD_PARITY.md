# P4.4 RWKV-7 Recurrent Forward Parity

P4.4 adds the RWKV-7 reference plugin's lazy, pure-JAX recurrent execution.
The sole equation authority is the pinned
[BlinkDL NumPy inference source](https://github.com/BlinkDL/RWKV-LM/blob/442120a5b40f7d764328bebde94324bc8790806f/RWKV-v7/rwkv_v7_numpy.py): repository
`BlinkDL/RWKV-LM`, commit `442120a5b40f7d764328bebde94324bc8790806f`, path
`RWKV-v7/rwkv_v7_numpy.py`, SHA-256
`dd683466cf97880c82879afbc8abb27a9596b12344a825d8325a1a1753597ee6`.
`qrwkv-xla` remains advisory-only and is neither imported, copied wholesale,
nor a production or test dependency.

## Execution boundary

`RWKV7ReferencePlugin.apply_jax` accepts the existing JAX plugin boundary:
the declared parameter tree, declared persistent carry, a rank-2 `[1, 4]`
integer `token_ids` batch, and final-logits objective scope. It validates the
frozen vocabulary range and delegates only to lazily imported `kernels.py`.
The result has logits `[1, 4, 16]` plus exactly these persistent carry leaves:

- `last_x_time` `[2, 8]`
- `last_x_channel` `[2, 8]`
- `time_state_matrix` `[2, 2, 4, 4]`

The kernels implement embedding; `ln0`, `ln1`, `ln2`, and output layer norms;
time-mix projections; decay, key/value/receptance recurrence; group norm;
channel mix; residual blocks; final norm; and untied output head. The sequence
path uses `jax.lax.scan`; the step path is the same pure JAX body. The first
block creates token-local `v0`; later blocks consume it, and it is never a
carry leaf.

JAX and `jax.numpy` occur only inside executable functions in concrete plugin
implementation modules. The base architecture contracts, models, registry, and
package entrypoints remain importable without JAX. RWKV `__init__.py`, config,
schema, and registration stay JAX-free. Architecture code imports neither
`radjax_student.runtime` nor any runtime key/seed authority.

## Fixture provenance and verification

The checked-in test-only generator creates one deterministic fixture using an
independent NumPy oracle rather than plugin helpers. Normal execution needs no
network, and tests regenerate/byte-compare checked-in data without fetching the
authority source. The provenance file records all source identities and digests:

| Item | Path | SHA-256 |
| --- | --- | --- |
| Generator | `tests/support/generate_rwkv7_reference_fixture.py` | `8ec17400d8337e913f9c173c19a0b6705b1ce509fb356c1f49166d24f6e65e54` |
| Oracle | `tests/support/rwkv7_reference_oracle.py` | `7097174bb058a1b91ea474c4147bbaa9f48f92f158e3900fc98c6fd53689b2b6` |
| Fixture | `tests/fixtures/rwkv7_reference/parity_fixture.json` | `15ef0a22d9d3fd5fb69af84887c9436a8f6b3536a3dfb11fb73d36f87b8139a6` |

The frozen fixture is float32 with vocabulary 16, hidden width 8, two blocks,
head count 2, head size 4, FFN width 16, and four token positions `[1, 7, 3,
5]`. Focused tests regenerate and byte-compare the fixture/provenance, compare
all logits and all persistent carry leaves to the independent oracle with
`rtol=1e-5` and `atol=2e-5`, prove step/scan agreement, finiteness, carry
change, token-order sensitivity, parameter-perturbation sensitivity, malformed
token/carry rejection, JAX capability registration, and P3.12C import audit
purity.

## Claim boundary

The exact compatibility claim is only step/sequence equation parity with the
pinned NumPy inference source on this declared tiny float32 fixture domain.
P4.4 does not claim parity beyond it, upstream initialization parity, training
recipe parity, gradients or learning integration, checkpoint/replay behavior,
HF conversion or export, pretrained-weight or weight-file compatibility, model
quality, performance, multi-device or TPU execution, teacher/Tome/distillation
work, a second architecture, or Phase 5 behavior.

P4.4 makes no generic framework change. The Phase 4 ledger still contains
exactly the approved sparse categorical cross-entropy objective, runtime-owned
initialization-key materializer, and architecture-neutral runtime-supplied
initialization-material request seam. P4.5 is responsible for generic learning
integration and its separately stated carry-gradient boundaries.
