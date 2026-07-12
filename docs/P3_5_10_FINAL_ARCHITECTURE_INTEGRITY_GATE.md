# P3.5.10 Final Architecture Integrity Gate

The final gate computes one immutable JSON receipt. `--write-receipt` writes it
explicitly to `docs/P3_5_ARCHITECTURE_INTEGRITY_RECEIPT.json`; the default CLI
does not modify the repository. Its thirteen flags cover
dependency direction, architecture/objective separation, the pure JAX path,
namespace and legacy isolation, HF and checkpoint preservation, documentation,
all prior phase receipts, import purity, and deterministic replay.

The gate is evidence for architecture and contract integrity only. It does
not claim production architecture training, Tome payload consumption,
behavioral distillation, HF export, model quality, or accelerator-scale
training.

The committed receipt currently passes all thirteen flags. Phase 4 is
unblocked for the first production architecture implementation, and it must
not import the deprecated `students/` compatibility package.
