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
PHASE 2 - STUDENT RUNTIME                           COMPLETE
PHASE 3 - GENERIC LEARNING CORE                     COMPLETE
PHASE 4 - ARCHITECTURE PLUGIN INGESTION             ACTIVE (P4.6)
```

## Phase 2 Status

The sequence is locked in
[RADJAX_PHASE2_RUNTIME_ROADMAP.md](RADJAX_PHASE2_RUNTIME_ROADMAP.md).

```text
P2.1  Runtime Contract and Terminology          COMPLETE
P2.2  Device and Environment Inspection         COMPLETE
P2.3  Runtime Backend Registry                  COMPLETE
P2.4  Single-Device CPU Runtime Smoke           COMPLETE
P2.5  RNG and Reproducibility Contract          COMPLETE
P2.6  Placement and Sharding Intent             COMPLETE
P2.7  Compilation and Execution Boundary        COMPLETE
P2.8  Runtime State Save/Restore Foundation     COMPLETE
P2.9  GPU/TPU Portability Smoke                 COMPLETE
P2.10 Runtime Golden Acceptance Gate            COMPLETE
```

## Phase 3 Status

The sequence is locked in
[RADJAX_PHASE3_GENERIC_LEARNING_CORE_ROADMAP.md](RADJAX_PHASE3_GENERIC_LEARNING_CORE_ROADMAP.md).

```text
P3.1  Generic Learning Contract                 COMPLETE
P3.2  Student Architecture Plugin Contract      COMPLETE
P3.3  Optimizer Contract                        COMPLETE
P3.4  Generic Batch and Objective Contract      COMPLETE
P3.5  Single Learning Step                      COMPLETE
P3.6  Model and Optimizer Checkpoint Contract   COMPLETE
P3.7  Generic Learning Loop                     COMPLETE
P3.8  Metrics, Hooks, and Reporting             COMPLETE
P3.9  Synthetic End-to-End Learning Smoke       COMPLETE
P3.10 Learning Core Golden Acceptance Gate      COMPLETE
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

## Current Integration Status

P3.11.1-P3.11.10 locally accepted

P3.11 integration closure complete

P3.12A locally accepted

P3.12B locally accepted

P3.12C locally accepted

P3.12D locally accepted

P4.6 checkpoints a real trained RWKV lifecycle through generic v3 persistence,
restores it into a fresh assembly, and proves next-step replay equality without
an RWKV checkpoint branch. It makes no broader equation-parity, training-recipe,
cross-step-BPTT, initialization-parity, weight-file, HF-conversion, or remote-CI
claim; see [P4.6 RWKV-7 Checkpoint, Restore, and Replay](P4_6_RWKV7_CHECKPOINT_REPLAY.md).

The closure makes no production architecture claim, no Tome payload consumption,
no distillation, no Hugging Face export, no accelerator-scale training, no
multi-device proof, no cross-hardware bitwise replay claim, no cross-version
bitwise replay claim, no performance claim, and no RadLads parity claim.
