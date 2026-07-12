# P3.11.9 Deterministic Replay Evidence

P3.11.9 executes the accepted validation-only stateful linear JAX conveyor
twice from fresh initialization in eager and JIT modes. Every replay includes
an uninterrupted six-step arm and a caller-bound checkpoint-v3 restore arm.
The replay runner never substitutes a smaller training implementation.

Same-mode uninterrupted/resumed and replay-A/replay-B evidence is bitwise
exact. Across eager and JIT, tree structure, shapes, dtypes, counters,
lifecycle identity, paths, hooks, metrics, RNG coordinates, and normalized
runtime receipt structure are exact; finite floating values use
`rtol=1e-6` and `atol=1e-6`.

The committed artifact is
[P3_11_9_REPLAY_EVIDENCE.json](P3_11_9_REPLAY_EVIDENCE.json), schema
`radjax.p3_11_9_replay_evidence.v1`. Its canonical artifact identity is
`8492513e63578c60d711420ac70149dc14e8eb2ed3a8fec702e3d740193a281c` and
its executed evidence digest is
`4a96633db2f08e3daa6af182181b18baa5967bafe35ba46ed2a1d74c38d0f926`.
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

P3.11.1-P3.11.9 accepted

P3.11.10 next and unstarted

Phase 4 blocked
