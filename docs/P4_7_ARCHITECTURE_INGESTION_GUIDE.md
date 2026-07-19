# P4.7 Architecture Ingestion Guide and Bounded Audit

This is the repeatable procedure for adding a future architecture plugin after
the first RWKV-7 reference. It records only the literal source boundaries
reviewed in P4.7; it is not a claim that arbitrary plugins are safe.

## Ingestion checklist

1. Start with the P4.1 contract: define one explicit package under
   `radjax_student.architecture.<plugin>`, static config/schema, a caller-owned
   registration function, and no discovery or default registration.
2. Record the mathematical authority, frozen compatibility domain, parameter
   mapping, carry/layout/HF identity, fixture provenance, and every non-claim
   before adding execution math.
3. Keep static modules JAX-free. A concrete JAX implementation may import only
   `jax`/`jax.numpy` lazily when initialization or execution is requested; it
   must not import runtime, learning, objectives, checkpoints, or validation.
4. Keep runtime key materialization and callable identity runtime-owned, and
   use the typed architecture request seam rather than parsing or deriving a
   runtime seed in the plugin. Learning assembles owners without leaf policy.
5. Add architecture-neutral generic behavior only with the Phase 4 ledger
   justification and an independent non-plugin basis. Otherwise stop for human
   direction; do not add an architecture mode, branch, checkpoint format, or
   objective policy.
6. Prove the plugin through focused configuration, initialization, forward,
   lifecycle, and checkpoint/replay tests, then run the bounded source audit.

## Bounded P4.7 audit report

`validation.architecture_audit.build_phase4_architecture_ingestion_audit`
returns the compact deterministic P4.7 report. It records the exact three
approved generic changes and their architecture-neutral justification, then
parses only the production source files in
runtime, learning, objectives, optimizers, steps, checkpoints, generic
architecture modules, every architecture module needed for the registration
scan, and the RWKV package. It rejects literal RWKV imports or identifiers in
generic owners, a direct validation import by the RWKV package, absence of the
declared RWKV package, and any registration call other than the one in
`architecture/rwkv7_reference/registration.py`. The current audit passes.

It intentionally does not track aliases, values, reflection, loaders, carrier
objects, comments, or transitive data flow. It does not prove a second plugin
would be correct; it proves this first plugin did not add a generic RWKV policy.

| Approved generic change | Architecture-neutral justification |
| --- | --- |
| Sparse categorical cross-entropy objective | Any token-logit architecture can provide `[B,T,V]` logits and integer `[B,T]` targets. |
| Runtime-owned initialization-key materializer | Any JAX architecture can consume a runtime-owned key without parsing a seed identity. |
| Runtime-supplied initialization material on `ArchitectureInitRequest` | Any JAX architecture can receive executable entropy while runtime derives it and learning composes it. |

No further generic change is approved. This checkpoint makes no claim of
arbitrary architecture compatibility, equation parity beyond the pinned RWKV
fixture domain, initialization parity, training-recipe parity, full BPTT,
weight-file compatibility, HF conversion/export, model quality, performance,
multi-device/TPU support, teacher/Tome/distillation work, a second architecture,
or Phase 5 behavior.
