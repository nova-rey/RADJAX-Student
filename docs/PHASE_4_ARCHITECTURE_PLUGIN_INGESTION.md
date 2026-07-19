# Phase 4 Architecture Plugin Ingestion

Phase 4 architecture-plugin ingestion locally accepted. The accepted first
plugin is `radjax.architecture.rwkv7_reference` version `1`, registered
explicitly by its caller and assembled through the existing generic learning,
runtime, objective, optimizer, checkpoint, and replay owners.

The generated [Phase 4 ingestion report](PHASE_4_ARCHITECTURE_PLUGIN_INGESTION_REPORT.json)
uses schema `radjax.phase4_architecture_ingestion_report.v1`. Its status is
derived only from typed executed facts: eager/JIT finite forward, loss, and
gradient evidence; parameter and carry movement; distinct prepared identities;
runtime-callable identity; and generic checkpoint/restore/next-step equality.
The current report is `pass` with evidence digest
`0c181935deb6a26d0ef73c5cca14b299e0344f19754693143c389a61331f2470`.

## Provenance and deterministic evidence

The report records the P4.4 fixture and generator/oracle provenance, including
the pinned BlinkDL/RWKV-LM `RWKV-v7/rwkv_v7_numpy.py` source commit
`442120a5b40f7d764328bebde94324bc8790806f` and source SHA-256
`dd683466cf97880c82879afbc8abb27a9596b12344a825d8325a1a1753597ee6`.
The two fresh P4.8 report generations were byte-identical; the committed JSON
has SHA-256 `58af42022246f9bdfaa595eb621f7d0fb47c3abbbeb385ce6d172f34d446de22`.

Equation parity remains the P4.4 pinned NumPy inference-reference claim on its
declared tiny float32 fixture domain only. The P4.8 report carries that
provenance and does not turn it into an initialization, training, or weights
compatibility claim. The P4.7 bounded audit also passes and records exactly the
three approved architecture-neutral changes: sparse categorical cross-entropy,
runtime-owned initialization-key materialization, and runtime-supplied
initialization material on `ArchitectureInitRequest`.

## Ownership and limits

Runtime owns initialization-reference validation and JAX-key materialization;
learning owns owner composition; the concrete plugin consumes supplied
initialization material and owns RWKV math. The plugin does not import runtime,
and only its concrete lazy implementation modules import JAX. Base architecture
contracts, static RWKV modules, registration, and package entrypoints remain
JAX-free at import time.

This local acceptance does not claim initialization parity, training-recipe
parity, cross-step BPTT, HF conversion/export, pretrained weights,
weight-file compatibility, model quality, performance, multi-device or TPU
execution, teacher/Tome/distillation work, a second architecture, Phase 5, or
remote CI success. Phase 4 local acceptance does not claim remote CI success.

The historical `students/` namespace remains deprecated compatibility code;
new Phase 4 architecture implementation belongs under
`radjax_student.architecture`. Its migration or removal remains a separately
scoped compatibility cleanup with an explicit inventory, migration plan,
deprecation handling, and regression proof.
