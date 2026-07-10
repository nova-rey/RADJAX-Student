# Roadmap

The long-range roadmap is [RADJAX_DEVELOPMENT_ROADMAP.md](RADJAX_DEVELOPMENT_ROADMAP.md).
This short checklist tracks the current scaffold.

## Phase 1 Status

```text
P1.1  TomeArtifactView seam                         COMPLETE
P1.2  Inferred run-defaults seam                    COMPLETE
P1.3  Production Tome gap review                    COMPLETE
P1.4  Cross-repository contract alignment plan     COMPLETE
P1.5  Tome/Contract production alignment           COMPLETE
P1.6  Student artifact-view correction              COMPLETE
P1.7  Student run-defaults correction               COMPLETE
P1.8  Student compatibility report                  COMPLETE
P1.9  Inspect/doctor CLI                            COMPLETE
P1.10 Production golden acceptance gate             COMPLETE

PHASE 1 - CONTRACT LAYER                            COMPLETE
PHASE 2 - STUDENT RUNTIME                           UNBLOCKED
```

## Phase 2 Status

The sequence is locked in
[RADJAX_PHASE2_RUNTIME_ROADMAP.md](RADJAX_PHASE2_RUNTIME_ROADMAP.md).

```text
P2.1  Runtime Contract and Terminology          COMPLETE
P2.2  Device and Environment Inspection         COMPLETE
P2.3  Runtime Backend Registry                  COMPLETE
P2.4  Single-Device CPU Runtime Smoke           COMPLETE
P2.5  RNG and Reproducibility Contract          UNBLOCKED; NEXT
P2.6  Placement and Sharding Intent             BLOCKED ON P2.5
P2.7  Compilation and Execution Boundary        BLOCKED ON P2.6
P2.8  Runtime State Save/Restore Foundation     BLOCKED ON P2.7
P2.9  GPU/TPU Portability Smoke                 BLOCKED ON P2.8
P2.10 Runtime Golden Acceptance Gate            BLOCKED ON P2.9
```

1. Keep tiny debug student and artifact inspection green.
2. Migrate student backend registry and current QRWKV backend in thin slices.
3. Add target loaders, sparse losses, checkpointing, and held-out evaluation.
4. Keep teacher inference out of this repo.
5. Complete Phase 0 foundation docs before promoting smoke/debug code to core
   APIs.
6. Use [STUDENT_SPLIT_CONTRACT.md](STUDENT_SPLIT_CONTRACT.md) as the product
   boundary for Phase 1 contract-layer work.
7. Treat the package skeleton as a placement map; do not interpret placeholder
   packages as implemented capability.
