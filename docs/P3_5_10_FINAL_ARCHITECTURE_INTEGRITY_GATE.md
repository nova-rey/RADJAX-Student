# P3.5.10 Final Architecture Integrity Gate

The final gate computes one immutable JSON receipt. `--write-receipt` writes it
explicitly to `docs/P3_5_ARCHITECTURE_INTEGRITY_RECEIPT.json`; the default CLI
does not modify the repository. Its thirteen flags cover
dependency direction, architecture/objective separation, the pure JAX path,
namespace and legacy isolation, HF and checkpoint preservation, documentation,
all prior phase receipts, import purity, and deterministic replay.

Every final-gate collection executes the complete evidence set twice and
compares canonical section evidence. The replay flag is false when either pass
differs. Section failures use stable `p35_*` blocker codes. Architecture/JAX,
HF, and checkpoint sections exercise their own negative cases rather than
trusting a focused-test receipt or a pre-failed injected result.

The gate is evidence for architecture and contract integrity only. It does
not claim production architecture training, Tome payload consumption,
behavioral distillation, HF export, model quality, or accelerator-scale
training.

The committed receipt currently passes all thirteen flags. That historical
P3.5 result does not bypass the later P3.11 integration closure, and no Phase
4 code may import the deprecated `students/` compatibility package.

## Current Integration Status

P3.11.1-P3.11.9 accepted

P3.11.10 next and unstarted

Phase 4 blocked
