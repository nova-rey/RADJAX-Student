# RADJAX Design Philosophy

> **Purpose:** This document is the project's compass. When
> implementation details, deadlines, or exciting optimizations start
> pulling the project in different directions, this document wins.

## The Mission

RADJAX is **not** an RWKV trainer.

RADJAX is a **behavior compiler**.

It consumes validated behavioral artifacts produced by RADJAX-Tome and
compiles that behavior into interchangeable student architectures.

The architecture being trained is an implementation detail.

------------------------------------------------------------------------

# First Principles

## 1. Contracts before code

Every major boundary is defined by an explicit contract before
implementation.

Code exists to satisfy contracts.

Contracts do not exist to justify code.

------------------------------------------------------------------------

## 2. Product before experiment

The original QRWKV-XLA repository proved the ideas could work.

RADJAX exists to turn those ideas into coherent software.

The old repository is reference material---not the architectural
template.

Port behaviors, not file layouts.

------------------------------------------------------------------------

## 3. One paved road

There should always be one obvious happy path.

Research modes, debugging utilities and experimental machinery remain
available, but they are not the public face of the project.

------------------------------------------------------------------------

## 4. Hugging Face is a first-class design constraint

Compatibility is not a final export step.

It influences:

-   configuration
-   tokenizer contracts
-   vocabulary
-   checkpoints
-   model layout
-   save/load behavior
-   inference APIs

If compatibility is added at the end, it is already too late.

------------------------------------------------------------------------

## 5. JAX/XLA for portability

We are not using JAX because TPUs are interesting.

We are using JAX because the same mathematical program should execute
correctly on:

-   CPU
-   GPU
-   TPU

Correctness comes first.

Performance comes second.

Hardware-specific optimization comes third.

------------------------------------------------------------------------

## 6. Optimization is layered

The stack should look like:

Level 0
:   Pure portable JAX implementation.

Level 1
:   JIT compilation.

Level 2
:   Device-aware sharding and runtime policies.

Level 3
:   Optional accelerator kernels (Pallas or equivalent).

Level 4
:   Architecture-specific fused implementations.

Fast paths must never become correctness paths.

------------------------------------------------------------------------

## 7. Architecture is a plugin

The project is not married to RWKV.

The student architecture is replaceable.

Examples:

-   RWKV
-   QRWKV
-   Mamba
-   Transformers
-   Future architectures

If an implementation satisfies the Student Architecture contract, the
rest of the system should not care.

------------------------------------------------------------------------

## 8. Runtime is separate from architecture

Architecture answers:

> "How does this model think?"

Runtime answers:

> "Where and how does it execute?"

These concerns remain independent.

------------------------------------------------------------------------

## 9. Validation is sacred

Nothing important should require trust.

Artifacts, compatibility, checkpoints, and reports should all be
machine-verifiable.

The project should prefer refusing to run over silently producing
questionable results.

------------------------------------------------------------------------

## 10. Claims are explicit

Every report should communicate:

-   what succeeded
-   what failed
-   what was validated
-   what is **not** being claimed

Passing a smoke test is not evidence of model quality.

------------------------------------------------------------------------

## 11. Modular conveyor belt

The pipeline should resemble a manufacturing line.

Each stage has one responsibility.

Each stage hands a validated artifact to the next.

Replacing one station should not require rebuilding the factory.

------------------------------------------------------------------------

## 12. Separate mechanism from policy

Mechanisms are infrastructure.

Policies are research.

Examples of mechanisms:

-   checkpoint writing
-   training loops
-   artifact loading
-   runtime execution

Examples of policies:

-   corridor schedules
-   exemplar schedules
-   loss weighting
-   compression strategies
-   architecture selection

Research should evolve without rewriting infrastructure.

------------------------------------------------------------------------

## The Long-Term Shape

RADJAX-Tome

↓

Validated behavioral artifact

↓

Student runtime

↓

Student architecture plugin

↓

Training schedule

↓

Checkpoint

↓

Evaluation

↓

Hugging Face package

Every boundary should be explicit.

Every artifact should be validated.

Every module should have one reason to change.

## Final Rule

When two implementations are possible:

Choose the one that makes the next architecture easier to add.

Choose the one that removes assumptions instead of introducing them.

Choose the one that makes the project simpler five years from now, even
if it takes an extra day today.

Build systems that survive success.
