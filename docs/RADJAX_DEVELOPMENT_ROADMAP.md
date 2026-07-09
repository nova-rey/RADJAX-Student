# RADJAX Development Roadmap

> **Purpose:** This document is the long-range roadmap for
> RADJAX-Student. It describes *capabilities*, not individual commits.
> Individual tasks may change; the direction should not.

## Guiding Principle

The first-generation QRWKV-XLA project answered:

> **Can this work?**

RADJAX-Student answers:

> **Can this become great software?**

The old repository is a reference implementation, not the structural
parent of this project.

Port behaviors.

Do not port architecture.

------------------------------------------------------------------------

# Phase 0 --- Foundation

**Question:** Do we know what we're building?

Deliverables:

-   Design Philosophy
-   Architecture Charter
-   Student Split Contract
-   Repository layout
-   Coding standards
-   Happy-path definition

**Exit Criteria**

A new contributor can explain the project after reading the
documentation.

------------------------------------------------------------------------

# Phase 1 --- Contract Layer

**Question:** Can we understand a Tome artifact?

Build:

-   Tome reader
-   Validation
-   Compatibility checking
-   Provenance
-   Reports
-   Diagnostics

The student should either say:

-   "I understand this artifact."

or

-   "I reject this artifact."

before allocating model weights.

------------------------------------------------------------------------

# Phase 2 --- Student Runtime

**Question:** Can we execute the conveyor belt?

Build the runtime pipeline:

Tome → Validation → Architecture Selection → Runtime Selection →
Training Schedule → Checkpoint → Evaluation → Reports

At this stage the runtime should not care whether it is training RWKV,
Mamba, or anything else.

------------------------------------------------------------------------

# Phase 3 --- Portable Training Core

**Question:** Can the same mathematical program execute everywhere?

Implement:

-   Portable JAX forward path
-   Optimizers
-   Losses
-   Metrics
-   Checkpointing

Supported execution targets:

-   CPU (debug/reference)
-   GPU (general use)
-   TPU (large-scale training)

Performance is secondary.

Correctness is mandatory.

------------------------------------------------------------------------

# Phase 4 --- First Student Architecture

**Question:** Does the plugin interface actually work?

Implement the first architecture plugin:

-   RWKV

This phase validates the plugin system---not RWKV itself.

------------------------------------------------------------------------

# Phase 5 --- Behavior Compilation

**Question:** Can teacher behavior become a student model?

Complete the first end-to-end path:

Teacher → Tome → Student → Checkpoint → Evaluation → Hugging Face
package

This is Version 1.

------------------------------------------------------------------------

# Phase 6 --- Architecture Ecosystem

**Question:** Can architectures be exchanged without changing
infrastructure?

Potential plugins:

-   RWKV
-   QRWKV
-   Mamba
-   Transformers
-   Future architectures

Adding a new architecture should require implementing one plugin---not
modifying unrelated systems.

------------------------------------------------------------------------

# Phase 7 --- Accelerator Optimization

**Question:** Can we make it fast?

Optimize independently of architecture.

Examples:

-   Pallas kernels
-   GPU fused kernels
-   TPU kernels
-   Precision policies
-   Sharding
-   Compilation caching

Fast paths are optional.

Correct paths remain universal.

------------------------------------------------------------------------

# Phase 8 --- Product UX

**Question:** Can someone actually use this?

The public interface should become boring:

-   `radjax-student doctor`
-   `radjax-student train`
-   `radjax-student eval`
-   `radjax-student inspect`
-   `radjax-student export`

Research utilities belong under research/debug namespaces---not the
primary workflow.

------------------------------------------------------------------------

# Phase 9 --- Production

**Question:** Would we trust this one year from now?

The project should be:

-   Stable
-   Predictable
-   Well documented
-   Portable
-   Reproducible
-   Maintainable

A fresh user should be able to install the project, run one
configuration, and obtain a complete end-to-end result.

------------------------------------------------------------------------

# Success Questions

  Phase                    Primary Question
  ------------------------ ----------------------------------------------
  Foundation               Do we know what we're building?
  Contract Layer           Can we understand the artifact?
  Student Runtime          Can we execute the pipeline?
  Portable Core            Can it run everywhere?
  First Architecture       Does one plugin prove the interface?
  Behavior Compilation     Can teacher behavior become a student model?
  Architecture Ecosystem   Can architectures be swapped freely?
  Optimization             Can we make it fast?
  Product UX               Can someone else use it?
  Production               Would we trust this in a year?

------------------------------------------------------------------------

# Canonical Principles

-   Reference implementations are disposable.
-   Interfaces are forever.
-   Port behaviors, not file layouts.
-   Hugging Face compatibility is a first-class design constraint.
-   JAX/XLA exists for portability, not TPU exclusivity.
-   Architecture plugins describe *how models think*.
-   Runtime backends describe *where models execute*.
-   Mechanism and policy remain separate.
-   Every boundary is validated.
-   Every report states what it does **not** claim.

------------------------------------------------------------------------

# The North Star

RADJAX-Student is a Hugging Face-aware, JAX/XLA-portable student runtime
that consumes validated behavioral artifacts and compiles teacher
behavior into interchangeable model architectures.

That is the product.

Everything else exists to serve that goal.
