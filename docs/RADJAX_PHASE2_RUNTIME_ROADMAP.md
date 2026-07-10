# RADJAX Phase 2 Roadmap - Student Runtime

**Status:** Locked

**Phase:** 2 - Student Runtime

**Depends on:** Phase 1 - Contract Layer complete

## Phase Goal

Phase 2 builds the architecture-independent execution layer for RADJAX-Student.

By the end of Phase 2, RADJAX-Student should be able to:

```text
validated run configuration
-> inspect local execution environment
-> select a runtime backend
-> initialize deterministic runtime state
-> place arrays on CPU/GPU/TPU
-> compile and execute a tiny pure function
-> save and restore runtime-owned state
-> report exactly what happened
```

Phase 2 does not implement a student architecture, generic training, Tome
payload consumption, or model quality evaluation.

## Governing Separation

```text
Architecture plugin
    defines model math

Training core
    defines generic optimization mechanics

Runtime backend
    defines where and how computation executes
```

Architecture code must not encode device topology. Runtime code must not encode
RWKV, Mamba, transformer, or other architecture math. Training code must not
own device discovery or backend-specific compilation policy.

## Locked Sequence

### P2.1 - Runtime Contract and Terminology

Freeze the public runtime concepts, ownership boundaries, capability model, and
non-claims.

### P2.2 - Device and Environment Inspection

Implement architecture-independent inspection of JAX/JAXLIB versions,
platform, devices, process topology, precision support, distributed state, and
warnings.

### P2.3 - Runtime Backend Registry

Add a stable backend registry and selection seam. Initial real backend: `jax`.
Permit lightweight fake/test backends.

### P2.4 - Single-Device CPU Runtime Smoke

Prove initialization, array creation, placement, tiny execution,
synchronization, timing, teardown, and JSON reporting on one CPU.

### P2.5 - RNG and Reproducibility Contract

Freeze named deterministic RNG streams for model initialization, data order,
dropout, augmentation, evaluation, and runtime tests.

### P2.6 - Placement and Sharding Intent

Define portable placement vocabulary: `single_device`, `replicated`,
`data_sharded`, `model_sharded`, `automatic`, and `unspecified`.

### P2.7 - Compilation and Execution Boundary

Create one stable boundary for eager/debug execution, JIT, static arguments,
donation, synchronization, timing, and reporting.

### P2.8 - Runtime State Save/Restore Foundation

Define and prove a runtime-owned state envelope containing run identity, step,
RNG, runtime config, topology, precision, placement, and resume metadata.

### P2.9 - GPU/TPU Portability Smoke

Run the same small runtime path on available CPU, GPU, and TPU targets through
one code path.

### P2.10 - Runtime Golden Acceptance Gate

Promote config -> inspection -> selection -> initialization -> placement ->
compiled execution -> save/restore -> report into one maintained gate.

## Phase 2 Guarantee

If P2.10 passes, RADJAX-Student may claim:

```text
RADJAX-Student can initialize and execute architecture-independent
JAX runtime work deterministically across supported execution targets,
with explicit placement, compilation, state, and reporting boundaries.
```

## Claims Not Made

Phase 2 does not claim student architecture implementation, RWKV/Mamba/
transformer support, Tome payload loading, loss computation, optimizer
implementation, training, checkpointed model state, evaluation, Hugging Face
export, distributed scale, performance optimization, model quality, or parity
with any external project.

## Locked Status

```text
P2.1  Runtime Contract and Terminology          COMPLETE
P2.2  Device and Environment Inspection         COMPLETE
P2.3  Runtime Backend Registry                  COMPLETE
P2.4  Single-Device CPU Runtime Smoke           UNBLOCKED; NEXT
P2.5  RNG and Reproducibility Contract          BLOCKED ON P2.4
P2.6  Placement and Sharding Intent             BLOCKED ON P2.5
P2.7  Compilation and Execution Boundary        BLOCKED ON P2.6
P2.8  Runtime State Save/Restore Foundation     BLOCKED ON P2.7
P2.9  GPU/TPU Portability Smoke                 BLOCKED ON P2.8
P2.10 Runtime Golden Acceptance Gate            BLOCKED ON P2.9
```
