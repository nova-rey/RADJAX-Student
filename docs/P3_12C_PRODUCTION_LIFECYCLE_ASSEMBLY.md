# P3.12C - Production Lifecycle Assembly

P3.12C establishes
`radjax_student.learning.assemble_jax_learning_lifecycle` as the one public
production authority for creating an executable JAX learning lifecycle. A
caller supplies a frozen `JaxLearningAssemblyRequest` containing typed
identities, configurations, schedule values, and non-executable orchestration
inputs. `JaxLearningAssemblyRegistries` selects executable implementations;
callers cannot inject plugins, backends, contexts, keys, arrays, parameter
catalogs, layouts, HF descriptors, or optimizer state.

The result is a frozen `JaxLearningAssemblyResult` with the exact
`JaxLearningLifecycle`, its bound `JaxLoopExecutor`, selected component
evidence, a canonical identity summary, and a deterministic assembly digest.
The schemas are `radjax.jax_learning_assembly_request.v1`,
`radjax.jax_learning_assembly_result.v1`, and
`radjax.jax_learning_assembly.v1`.

## Ownership and construction

The assembly sequence is explicit and owner-preserving:

```text
typed request
  -> production registry selection
  -> architecture initialization and HF identity derivation
  -> architecture-owned objective-surface resolution
  -> objective-registry execution descriptor
  -> optimizer-owned state initialization
  -> runtime-owned context and key-stream binding
  -> learning-owned lifecycle and loop executor
  -> generic production step, checkpoint, and report
```

Architecture owns initialization, parameter catalog/layout, carry, HF
descriptor/reference, and objective-surface resolution. The ObjectiveRegistry
owns objective selection and implementation identity. The optimizer owns
optimizer-state initialization and updates. Runtime owns backend selection,
device/context construction, placement, dispatch, and key streams. Learning
coordinates those outputs but does not inspect parameter or optimizer leaves,
fabricate HF identity, choose a device, or execute component semantics.

## Frozen evidence

The receipt at
[`P3_12C_PRODUCTION_LIFECYCLE_ASSEMBLY_RECEIPT.json`](P3_12C_PRODUCTION_LIFECYCLE_ASSEMBLY_RECEIPT.json)
uses schema `radjax.p3_12c_production_lifecycle_assembly_receipt.v1`. It has
exactly 17 ordered positive proofs and 36 ordered adversarial cases. Every
adversary is invoked twice from fresh baseline inputs; acceptance requires
exact observed/expected blocker equality and an observed boundary derived from
the callable actually invoked. There is no permissive blocker-family matcher,
prefix matcher, or expected-to-observed translation.

The recorded assembly digest is
`add7080a34ed2c81d4f5ae93fe8ec74783f3ce1b9823c5cb32cb92da5661bda7`; the
receipt evidence digest is
`fffff62866a30f96afbeafc23e3b0d7aa9e641cb3c978acd2a8ba0a97f4eb98e`.

The JAX-free one-authority audit uses schema
`radjax.p3_12c_one_authority_audit.v1`. It rejects a second assembler,
validation-owned happy paths, production-to-validation imports, executable
request injection, assembler fabrication of component-owned identity, raw
device selection, parameter/optimizer leaf inspection, caller success flags,
and permissive observation. Its 24 synthetic source fixtures execute the real
audit independently. Production code imports no P3.12C validation gate code.
Its recorded implementation-audit digest is
`2886ef483837590bd66ed5882b3d1ecaa1de6e1f461b6fca881384f0a0c42b26`.

The product-path proof executes one real generic JAX step through the assembled
executor, verifies finite loss, metrics, and exposed gradients, parameter
movement, optimizer/learning-state advancement, and carry advancement, then
emits and validates a v3 checkpoint and compact report against the assembled
identities.

## Validation construction review

| Path | Prior role | Classification | Migrated to production assembler | Retained lower-level reason |
| --- | --- | --- | --- | --- |
| `src/radjax_student/validation/p3_11_8_systems_receipt.py` and `tests/test_p3_11_stateful_systems.py` | Stateful owner-contract proof | Focused lower-level | No | Test-only direct records isolate malformed architecture/optimizer/lifecycle contracts; they are not the accepted happy path. |
| `src/radjax_student/validation/p3_11_9_replay/runner_jax.py` | Successful stateful replay conveyor | Must migrate | Yes | `_new_assembly` calls the production assembler; `_new_lifecycle` is only a compatibility projection of that result. |
| `src/radjax_student/validation/p3_11_10_gate/runner_jax.py` | Final adversarial gate positive path | Historical consumer | Yes, transitively | It reuses P3.11.9 rather than constructing a lifecycle. |
| `src/radjax_student/validation/p3_11_10_gate/implementations/section_a_contracts.py` | Malformed boundary fixture | Focused lower-level | No | Its direct executor constructor deliberately receives invalid payloads to test its owner boundary. |
| `src/radjax_student/validation/p3_12a_objective_identity/runner_jax.py` | Objective identity/checkpoint/replay proof | Must migrate | Yes, transitively | It obtains lifecycle values from P3.11.9's production-backed compatibility projection. |
| `src/radjax_student/validation/p3_12b_hf_descriptor_authority/runner_jax.py` | HF descriptor checkpoint/replay/report proof | Must migrate | Yes, transitively | It obtains lifecycle values from the same production-backed projection. |
| `src/radjax_student/learning/p3_10_acceptance.py` | Historical Phase 3 acceptance reader | Historical reader | Not applicable | It audits historical seams and does not construct a JAX lifecycle. |
| `tests/test_p3_5_jax_learning.py` and `tests/test_p3_9_synthetic_learning_smoke.py` | Focused generic JAX lower-level contracts | Focused lower-level | No | They test individual step/smoke seams rather than a competing validation happy-path recipe. |
| `src/radjax_student/validation/p3_12c_production_lifecycle_assembly/runner_jax.py` | Canonical P3.12C product path | Must migrate | Yes | It calls the public assembler directly and never reproduces the construction sequence. |

P3.12C is locally accepted only from its typed executed evidence and local
gates. P3.12D is next and unstarted. Phase 4 remains unstarted. This checkpoint
does not claim RWKV, Tome consumption, distillation, HF export, model quality,
multi-device or TPU execution, performance, a production CLI, or resume
assembly.
