# Student Split Contract

Phase: 0 - Foundation

This contract defines the practical product boundary for RADJAX-Student. It
builds on the [design philosophy](DESIGN_PHILOSOPHY.md), the
[development roadmap](RADJAX_DEVELOPMENT_ROADMAP.md), and the
[architecture charter](ARCHITECTURE_CHARTER.md).

P0.4 is foundation work only. It documents what this repository promises to do
without implementing training, JAX, architecture plugins, or new algorithmic
behavior.

## Mission

RADJAX-Student is a Hugging Face-aware, JAX/XLA-portable student runtime that
consumes validated behavioral artifacts and compiles teacher behavior into
interchangeable student model architectures.

It is a behavior compiler runtime. It is not an RWKV-only trainer, a
QRWKV-XLA continuation repo, a Tome builder, a corpus builder, a dense-logits
experiment repo, or a public junk drawer for training scripts.

## Product Boundary

RADJAX-Student owns:

- artifact consumption
- student compatibility validation
- runtime orchestration
- training mechanisms
- architecture plugin interfaces
- schedule policy interfaces
- checkpointing
- evaluation
- reports
- Hugging Face packaging and export
- user-facing CLI

RADJAX-Student does not own:

- teacher inference
- Tome generation
- corpus construction
- RADJAX-Contract schema ownership
- experimental one-off scripts as public UX

## Inputs

The primary input is a validated behavioral artifact produced outside this
repository.

Expected future input classes include:

- Tome artifact path
- Tome manifest
- cover page
- records
- compressed target payloads
- tokenizer and vocabulary contract
- provenance metadata
- optional schedule or run configuration file

P0.4 does not add readers beyond existing smoke/debug code. This section names
the input contract only.

## Outputs

The long-term product outputs are:

- trained student checkpoint
- run report
- validation report
- evaluation report
- resolved config
- claims-not-made section
- Hugging Face-compatible export package

P0.4 does not implement these outputs. This section names the output contract
only.

## Happy Path

The future public command path should be:

```text
radjax-student doctor
radjax-student inspect --tome <path>
radjax-student train --config <run.yaml>
radjax-student eval --checkpoint <path>
radjax-student export --checkpoint <path> --output <hf_dir>
```

The first true product path is:

```text
validated Tome
-> compatibility check
-> architecture plugin selection
-> runtime selection
-> training schedule
-> checkpoint
-> eval
-> report
-> Hugging Face export
```

## Non-Goals

P0.4 must not:

- implement training
- add JAX
- add torch
- add transformers
- add datasets
- add accelerator logic
- add new dense target behavior
- promote smoke/debug code to core APIs
- expand `students/` as a permanent public namespace
- create large abstractions without immediate contract value

## Claims Made

At P0.4, the repository may claim:

- the project product boundary is documented
- inputs and outputs are named
- the public happy path is defined
- non-goals are explicit
- existing smoke/debug code remains classified
- future implementation must respect P0.3 architecture boundaries

## Claims Not Made

At P0.4, the repository must not claim:

- training works
- JAX portability works
- Hugging Face export works
- architecture plugins are implemented
- Tome compatibility is complete
- model quality is proven
- RadLads parity is proven

## Relationship To RADJAX-Contract

RADJAX-Contract owns shared schemas, validation, compatibility contracts, and
artifact definitions.

RADJAX-Student may depend on RADJAX-Contract. RADJAX-Contract must not depend on
RADJAX-Student internals.

Student-side validation should use Contract-defined artifact files and published
Contract APIs. Student should add local compatibility checks only where they
describe whether this runtime can consume a valid Contract artifact.

## Relationship To RADJAX-Tome

RADJAX-Tome owns teacher inference and behavioral artifact production.

RADJAX-Student consumes Tome output only through Contract-defined artifacts and
published Contract APIs. It must not import Tome implementation modules or reach
into Tome internals.

## Relationship To QRWKV-XLA

QRWKV-XLA is reference material, not the parent architecture of
RADJAX-Student.

RADJAX-Student should port behaviors, not file layouts. Existing QRWKV-XLA code
should not be promoted into core APIs until it fits the architecture charter,
the Student split contract, and the relevant phase contract.

## Current-Code Classification Summary

The P0.3 classification remains in force:

| Area | Bucket | Summary |
| --- | --- | --- |
| `artifacts/loaders.py` | Smoke/debug | Thin Contract-backed Tome inspection. |
| `artifacts/targets.py` | Smoke/debug | Dense Tome loading for NumPy smoke work, not the production training substrate. |
| `legacy/losses/dense_kl.py` | Legacy/offline analysis | Dense teacher-probability loss; not a canonical objective. |
| `legacy/losses/sparse_topk.py` | Legacy/offline analysis | Compressed-target analysis mechanism; not a settled public API. |
| `students/base.py` | Core architecture candidate | Seed of the plugin contract in a transitional namespace. |
| `students/registry.py` | Core architecture candidate | Useful registry behavior that should migrate toward `architecture/`. |
| `students/tiny_debug/` | Smoke/debug | NumPy backend for import, registry, and training smoke tests. |
| `training/distill.py` | Smoke/debug | One-step distillation smoke, not the product training loop. |
| `cli/train_student.py` | Smoke/debug | Early CLI shim, not the final public command surface. |
| `tests/tome_fixtures.py` | Smoke/debug | Contract-valid fixtures for local tests. |

Dense Tome loading and sparse top-k loss remain smoke/debug until later phases
explicitly promote or replace them. The `students/` namespace remains
transitional.

## Phase 1 Entry Criteria

Phase 1 may begin only if:

- `docs/DESIGN_PHILOSOPHY.md` exists and is readable.
- `docs/RADJAX_DEVELOPMENT_ROADMAP.md` exists and is readable.
- `docs/ARCHITECTURE_CHARTER.md` exists and is readable.
- `docs/STUDENT_SPLIT_CONTRACT.md` exists and is readable.
- README or a docs index links to all four documents.
- The project states what it does and does not claim.
- No new implementation work has escaped foundation scope.
