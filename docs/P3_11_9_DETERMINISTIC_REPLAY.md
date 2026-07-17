# P3.11.9 Deterministic Replay Evidence

P3.11.9 executes the accepted validation-only stateful linear JAX conveyor
twice from fresh initialization in eager and JIT modes. Every replay includes
an uninterrupted six-step arm and a caller-bound checkpoint-v3 restore arm.
The replay runner never substitutes a smaller training implementation.

Same-mode uninterrupted/resumed and replay-A/replay-B evidence is bitwise
exact. Across eager and JIT, tree structure, shapes, dtypes, counters,
lifecycle identity, paths, hooks, metrics, RNG coordinates, and normalized
runtime receipt structure are executed comparisons. Integer and Boolean leaves
are exact; finite floating leaves and metrics use `rtol=1e-6` and `atol=1e-6`.
The recorded nested lifecycle, runtime, RNG, tolerance, cross-mode, and
verifier objects are exact-schema, canonical evidence contracts.

The committed artifact is
[P3_11_9_REPLAY_EVIDENCE.json](P3_11_9_REPLAY_EVIDENCE.json), schema
`radjax.p3_11_9_replay_evidence.v1`. Its canonical artifact identity is
`f2cf0c109521f437d9ae71f8067272985e28e688f599da91dc2763652edf277d` and
its executed evidence digest is
`8858b12c1d4fa29356a5cd152cd6c61536684d451739bc0832d3a6cb47fdc065`.
Run the read-only gate with:

```bash
python -m radjax_student.validation.p3_11_9_replay --check-recorded
```

The evidence includes caller-supplied HF/config/catalog/layout/state/carry
identities before restore, checkpoint manifest identity, runtime RNG and
placement evidence, and digests for all post-step state without retaining raw
arrays or raw keys.

It does not claim a production architecture, Tome payload consumption,
distillation, Hugging Face export, accelerator-scale or multi-device training,
cross-hardware or cross-version bitwise determinism, performance, RadLads
parity, or Phase 4 readiness.

## Current Integration Status

P3.11.1-P3.11.10 locally accepted

P3.11 integration closure complete

P3.12A locally accepted

P3.12B locally accepted

P3.12C next and unstarted

Phase 4 remains unstarted

Phase 4 requires successful required remote base/JAX CI or an explicit repository-owner waiver

The closure makes no production architecture claim, no Tome payload consumption,
no distillation, no Hugging Face export, no accelerator-scale training, no
multi-device proof, no cross-hardware bitwise replay claim, no cross-version
bitwise replay claim, no performance claim, and no RadLads parity claim.
