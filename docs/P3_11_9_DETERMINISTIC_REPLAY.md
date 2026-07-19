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
`9ba4734c2beba763448790dfd8949512bf316b23ebb77363b8a1947985c99aba` and
its executed evidence digest is
`e95217d0cba2457731dcf7f6ea7849ea70866f24c73f4716e3dc4da0ecad907b`.
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

P3.12C locally accepted

P3.12D next and unstarted

Phase 4 architecture-plugin ingestion locally accepted

Phase 4 local acceptance does not claim remote CI success

The closure makes no production architecture claim, no Tome payload consumption,
no distillation, no Hugging Face export, no accelerator-scale training, no
multi-device proof, no cross-hardware bitwise replay claim, no cross-version
bitwise replay claim, no performance claim, and no RadLads parity claim.
