# RADJAX Phase 3 Roadmap - Generic Learning Core

**Status:** Locked  
**Phase:** 3 - Generic Learning Core  
**Depends on:** Phase 2 - Student Runtime complete

## Phase Goal

Phase 3 builds the architecture-independent machinery by which a Student can
change over time:

```text
generic batch
-> architecture forward contract
-> objective evaluation
-> gradient computation
-> scoped parameter update
-> optimizer state transition
-> metrics
-> checkpoint
-> restore
-> deterministic continuation
```

Runtime owns where and how computation executes. Learning owns state change.
Architecture owns model math and parameter meaning. The behavior layer owns
Tome-specific objectives and pass policy.

## Scope Contract

`UpdateScope` controls which parameters may change and defaults to
`whole_student`. `ObjectiveScope` controls where the learning signal is
observed and defaults to `final_output`. Architecture plugins resolve region
identifiers into stable paths; generic learning does not interpret layers or
regions.

Future scoped updates prefer a stable parameter tree, stable optimizer-state
tree, and deterministic update mask. Frozen parameters may still participate in
forward computation.

## Locked Status

```text
P3.1  Generic Learning Contract                 COMPLETE
P3.2  Student Architecture Plugin Contract      COMPLETE
P3.3  Optimizer Contract                        COMPLETE
P3.4  Generic Batch and Objective Contract      COMPLETE
P3.5  Single Learning Step                      COMPLETE
P3.6  Model and Optimizer Checkpoint Contract   COMPLETE
P3.7  Generic Learning Loop                     UNBLOCKED; NEXT
P3.8  Metrics, Hooks, and Reporting             BLOCKED ON P3.7
P3.9  Synthetic End-to-End Learning Smoke       BLOCKED ON P3.8
P3.10 Learning Core Golden Acceptance Gate      BLOCKED ON P3.9
```

## Current Non-Claims

P3.1 does not provide an architecture plugin, parameter tree, named-region
resolution, gradients, optimizer updates, checkpoint files, loops, Tome loading,
functional-stage distillation, or model quality.
