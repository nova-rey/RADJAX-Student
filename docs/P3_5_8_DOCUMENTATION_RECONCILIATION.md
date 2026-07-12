# P3.5.8 Documentation Reconciliation

The maintained status is deliberately narrow.

**Proven**

- production Tome metadata inspection
- compatibility evaluation
- runtime lifecycle and execution policy
- scalar Phase 3 compatibility path, isolated under `radjax_student.legacy`
- checkpoint/resume mechanics
- pure JAX linear learning contract with autodiff and functional architecture carry
- eager/JIT equivalence through the runtime execution boundary
- HF preservation descriptors

**Not yet proven**

- real production architecture training
- Tome payload consumption
- behavioral distillation
- Hugging Face export
- model quality
- accelerator-scale training
- tensor-pytree checkpoint storage

The generic loop requires an explicit step executor. Its scalar path is legacy,
and the JAX path has no legacy fallback. Architecture carry is not runtime state.
Continuation checkpoint v2 stores scalar parameter mappings; an HF distribution
checkpoint remains a separate future contract.

Phase 3.5 is complete. The deprecated `students/` compatibility package is
architecturally dead and may be removed at P4.1; Phase 4 may now proceed without
importing it.
