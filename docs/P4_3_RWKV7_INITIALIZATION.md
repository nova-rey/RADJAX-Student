# P4.3 RWKV-7 Parameter and Carry Initialization

P4.3 makes the fixed tiny float32 RWKV-7 reference schema initialize complete
parameter and persistent-carry trees. It does not add an RWKV forward kernel,
equation-parity claim, training step, checkpoint proof, HF conversion, or
weight-file compatibility.

## Initialization ownership

The initialization flow has one randomness authority:

```text
runtime initialization reference
  -> runtime validates and lazily materializes a JAX key
  -> learning constructs ArchitectureInitRequest
  -> concrete architecture consumes opaque initialization material
```

`runtime.jax_bridge.materialize_initialization_jax_key` accepts only a canonical
`runtime_keys.v1:initialization:<nonnegative decimal>` reference and derives the
runtime-owned initialization key. `learning.assembly` invokes that materializer
while assembling a JAX lifecycle. `ArchitectureInitRequest` carries the result
as opaque, nonserialized `runtime_initialization_material`; it is excluded from
request serialization, checkpoints, configuration, HF descriptors, reports,
and receipts.

`RWKV7ReferencePlugin` imports neither `radjax_student.runtime` nor a seed
parser. It rejects missing or invalid supplied material, splits only the
provided key across its fixed parameter slots, and does not hash, reconstruct,
or substitute a seed authority. Equal runtime references therefore provide
equal keys and equal initialized leaves; changed references change initialized
values.

## Bounded JAX import rule

The P3.12C source audit permits only function-local `jax` and `jax.numpy`
imports in concrete implementation modules below a
`radjax_student.architecture.<plugin>` package. It does not permit NumPy model
math, Torch, Transformers, or any other optional framework. Architecture base
contracts, models, registry, and package entrypoints remain JAX-free. The RWKV
package entrypoint, config, schema, and registration modules remain JAX-free;
JAX is loaded only when initialization or execution is requested.

## Materialized result

The plugin uses fixed catalog-path slots and JAX float32 normal draws with a
fixed architecture-owned scale to fill every declared parameter-layout leaf.
It validates the materialized layout, creates zeroed `last_x_time`,
`last_x_channel`, and `time_state_matrix` carry leaves from the declared carry
descriptor, and returns the existing descriptor-derived HF reference. The
executable JAX capability remains unadvertised until P4.4, and upstream
initialization parity is not claimed.

## Approved generic changes

The Phase 4 ledger contains exactly these approved architecture-neutral generic
changes:

1. Sparse categorical cross-entropy objective.
2. Runtime-owned initialization-key materializer.
3. `ArchitectureInitRequest` runtime-supplied initialization material.

The third seam is architecture-neutral because every JAX architecture needs
executable initialization entropy; runtime owns derivation, learning owns
composition, and architecture owns parameter initialization. A future Mamba,
Transformer, or other JAX plugin can consume the same seam. Allowing an
architecture-to-runtime import would invert those ownership boundaries and is
not an acceptable alternative.

## Verification and non-claims

Focused tests prove canonical-reference rejection at the runtime owner,
deterministic supplied-key and complete-tree equality, changed-reference value
difference, complete float32 layout/carry/HF evidence, nonserialization, no
plugin runtime import, base-package import isolation, and the bounded P3.12C
audit allowance. Non-RWKV test doubles remain ordinary architecture contracts;
they are not reclassified as production JAX plugins.

P4.3 makes no claim of pinned-equation parity, upstream initialization parity,
pretrained weights, HF conversion or export, model quality, training behavior,
performance, multi-device or TPU execution, teacher/Tome/distillation work, or
Phase 5 behavior.
