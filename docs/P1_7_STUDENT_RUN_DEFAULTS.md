# P1.7 Student Run Defaults

`infer_run_defaults(view)` and `infer_run_defaults_from_tome(path)` produce a
stable configuration seed from the accepted production Tome contract. They
infer facts, preserve requirements, and leave choices unresolved.

## Source Separation

`StudentRunDefaults` keeps these categories distinct:

- `artifact_facts`: identity, provenance-derived model/tokenizer facts,
  dimensions, validation statuses, and content/surface counts;
- `available_surfaces`: every generic surface plus optional corridor or exemplar
  metadata summaries;
- `required_capabilities`: deterministic versioned requirements aggregated from
  surfaces, the pass plan, and Contract inspection;
- `recommended_training_plan`: ordered surface references, checkpoint flags,
  prerequisites, capabilities, and target scopes;
- `required_from_user`: architecture, size/config, budget, and output location;
- `unresolved_by_phase`: runtime, precision, optimizer, schedule, loss,
  architecture-plugin, evaluation, and export policy;
- separate artifact and Student claims not made.

The JSON representation preserves this separation and contains no Python
classes or executable schedule objects.

## Surface Extensibility

The defaults root accepts any number of behavioral surfaces. Unknown kinds,
target scopes, semantic metadata, content-role strings, and required capability
names survive normalization. Known corridor/exemplar details are optional
summaries attached to their generic surface entries, not root assumptions.

Capability requirements and the P1.6 unsupported set are reported. All required
capabilities are also marked not yet evaluated because P1.7 does not compare
them with an architecture, runtime, or Student implementation. P1.8 owns the
compatibility verdict.

## Artifact Intent Is Not Execution

The canonical plan remains:

```text
corridor
-> checkpoint
-> exemplar
-> checkpoint
```

These records do not instantiate a schedule, execute checkpoints, choose an
optimizer, assign loss weights, allocate a model, or load payload tensors.

## Legacy Dense Smoke Defaults

The explicit `legacy_dense_v0` branch retains singular payload, compression,
and adapter values under `legacy_smoke_defaults`. The deprecated
`inferred_from_tome` property reads only that legacy mapping. Production defaults
never use `TomeInferredDefaults`, payload format, or adapter family.

P1.7 does not claim compatibility, implemented capabilities, selected
architecture/runtime/optimizer, executable schedules, payload loading, training,
JAX portability, checkpoint execution, Hugging Face export, or model quality.
