# Architecture Charter

Phase: 0 - Foundation

This charter defines the architectural boundaries of RADJAX-Student. It does
not define algorithms. It defines ownership, dependency direction, and where
future work belongs.

Normative inputs:

- [DESIGN_PHILOSOPHY.md](DESIGN_PHILOSOPHY.md)
- [RADJAX_DEVELOPMENT_ROADMAP.md](RADJAX_DEVELOPMENT_ROADMAP.md)

If implementation pressure conflicts with these documents, the implementation
changes first.

## Core Principle

RADJAX-Student is not the child layout of QRWKV-XLA. QRWKV-XLA is reference
material. Port behaviors, not structure.

Interfaces are the durable product. Reference implementations are disposable.

## System Boundary

The long-term conveyor belt is:

```text
RADJAX-Tome
-> validated behavioral artifact
-> RADJAX-Student runtime
-> architecture plugin
-> training schedule
-> checkpoint
-> evaluation
-> Hugging Face package
```

Every arrow is a contract boundary. Each stage should accept a validated input,
produce a validated output, and have one reason to change.

## Dependency Direction

`RADJAX-Contract` owns shared schemas, validation, compatibility contracts, and
artifact definitions. It must not know about Student internals.

`RADJAX-Tome` produces behavioral artifacts. It must not depend on Student.
Student consumes Tome output only through published Contract APIs and artifact
files. Student must not import Tome implementation modules.

`RADJAX-Student` owns artifact consumption, runtime, training, architecture
plugins, reports, Hugging Face export, and user-facing commands.

Forbidden default imports remain:

- `radjax_tome`
- `torch`
- `transformers`
- `datasets`
- `accelerate`

Those packages may appear only behind explicit optional integration boundaries
when the relevant phase adds them.

## Module Layout

The repository should evolve toward:

```text
src/radjax_student/
  artifacts/      # Contract-backed artifact readers and adapters
  runtime/        # device, precision, compilation, execution policy
  architecture/   # architecture plugin contracts and implementations
  training/       # training loop mechanisms
  schedules/      # research policy for training and loss schedules
  hf/             # Hugging Face config, checkpoint, save/load, export
  reports/        # explicit claims, diagnostics, and run summaries
  cli/            # public product commands and narrow entrypoints
  validation/     # Student-side compatibility and readiness checks
```

Current directories that do not match this target are transitional. They should
move only when a focused change can preserve behavior and tests. Empty
long-term packages are placement boundaries, not proof of implemented
capability.

## Architecture Plugins

Architecture plugins answer: how does this model compute?

Examples include RWKV, QRWKV, Mamba, Transformers, and future architectures.

Architecture plugins must not know:

- CPU vs GPU vs TPU placement
- sharding
- Pallas or accelerator kernels
- runtime compilation policy

If an implementation satisfies the Student Architecture contract, the rest of
the system should not care which architecture it is.

## Runtime Backends

Runtime answers: where and how does execution occur?

Runtime owns:

- device policy
- precision policy
- JIT and compilation policy
- checkpoint execution mechanics
- optional accelerator optimizations

Runtime must not know which architecture is executing beyond the stable
architecture plugin interface.

Fast paths must not become correctness paths. Portable correctness comes first,
then JIT, then device-aware policy, then optional accelerator kernels, then
architecture-specific fused implementations.

## Hugging Face Boundary

Hugging Face compatibility is a design constraint, not a late export hack.

Future config, tokenizer, vocabulary, checkpoint, model layout, save/load, and
inference APIs should preserve the information needed for native Hugging Face
packaging.

## Product And Research Paths

The public path should become boring and obvious:

```text
radjax-student doctor
radjax-student inspect
radjax-student train
radjax-student eval
radjax-student export
```

Research, debug, and migration tools may exist, but they should live under
explicit research/debug/migration namespaces and should not become the public
interface.

## Existing Code Review

Current implementation classification:

| Path | Bucket | Notes |
| --- | --- | --- |
| `artifacts/loaders.py` | Smoke/debug | Thin Contract-backed Tome inspection. Keep, but do not treat as complete Phase 1 compatibility checking. |
| `artifacts/targets.py` | Smoke/debug | Dense Tome loading is useful for NumPy smoke work, but production compressed payloads should become the primary training substrate. |
| `losses/dense_kl.py` | Smoke/debug | Dense teacher probability loss for small validated tests. |
| `losses/sparse_topk.py` | Smoke/debug | Early mechanism for compressed targets. Keep as implementation detail until production target contracts settle. |
| `students/base.py` | Core architecture candidate | Existing protocol is the seed of the architecture plugin contract, but it lives under the transitional `students/` namespace. |
| `students/registry.py` | Core architecture candidate | Registry behavior is useful, but the long-term namespace should be `architecture/`. |
| `students/tiny_debug/` | Smoke/debug | NumPy backend for import, registry, and training smoke tests only. |
| `training/distill.py` | Smoke/debug | One-step distillation smoke, not the product training loop. |
| `cli/train_student.py` | Smoke/debug | Early CLI shim, not the final paved-road command surface. |
| `tests/tome_fixtures.py` | Smoke/debug | Contract-valid fixtures for local tests. |

No current code is marked `Remove`.

## Documented Conflicts

- The repository currently uses `students/` for architecture-like concepts.
  The charter target is `architecture/`. Do not expand `students/` as a
  permanent public API; migrate in thin slices when adding the architecture
  contract.
- Runtime, schedules, HF export, reports, and Student validation packages now
  exist only as skeleton placement boundaries. New work in those areas should
  fill the target modules rather than hiding responsibilities in `training/` or
  `students/`.
- Dense Tome target loading is intentionally smoke/debug. It should not become
  the default training substrate once compressed production payloads are ready.
- Current CLI shape does not match the long-term product path. Future commands
  should converge on `doctor`, `inspect`, `train`, `eval`, and `export`.

## Exit Criterion

A new engineer or Codex session should be able to determine where a new feature
belongs from this charter and the linked normative documents without reading
existing implementation code.
