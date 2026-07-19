# Foundation Audit Closure

This closure resolves the focused post-P3.12 foundation findings without adding
new product capability or beginning Phase 4.

## Ownership corrections

Runtime owns callable declarations, source-derived identities, bindings, exact
references, and registry mechanics. It owns no application operation set and
does not import learning or steps. `learning/composition.py` is the narrow
application composition root: it registers the declared generic JAX learning
step and nothing else.

The former `radjax_student.losses` NumPy implementations now live only at
`radjax_student.legacy.losses`. They are retained for legacy/offline analysis,
not canonical training objectives. The canonical assembled JAX path does not
import them.

The canonical JAX training path is inspected as a literal reviewed source set.
Its JAX-free AST audit rejects runtime imports of application, architecture,
Tome, or RWKV namespaces; source-computed runtime import targets; and host
conversion calls in the reviewed canonical path. It also detects new
checkpoint-proof behavior under production owners even when a new module uses a
neutral filename. The four historical proof paths in the table below are exact
frozen exceptions, not a broad filename exemption.
It excludes Torch, TensorFlow, TensorFlow Probability, Transformers execution,
NumPy loss/model math, and host conversion of trainable arrays. NumPy remains
permitted in legacy analysis, artifact parsing, reports, and deterministic
metadata handling.

HF identity remains architecture-owned: initialization produces the descriptor
and derived preservation reference; assembly consumes them unchanged; checkpoint
and report paths validate descriptor-derived preservation evidence. The closure
audit validates both the current P3.12B recorded source evidence and these
production authority paths.

## Production and proof namespaces

Product compatibility decides configuration, artifact, or capability
consumability. Development proof runs gates, adversaries, receipts, and source
audits. Product code must not import proof-owned `validation` machinery;
validation may import production. New checkpoint-specific proof code belongs
under `radjax_student.validation`, never under production owners.

The following historical modules remain explicit exceptions because moving them
would create unrelated churn. They set no precedent for new production-owned
proof code.

| Path | Current role | Canonical production import | Validation-only in practice | Future disposition | Immediate action |
| --- | --- | --- | --- | --- | --- |
| `learning/p3_5_acceptance.py` | historical architecture gate | no | yes | migrate only in a dedicated compatibility cleanup | retain as frozen exception |
| `learning/p3_10_acceptance.py` | historical learning-core gate | no | yes | migrate only in a dedicated compatibility cleanup | retain as frozen exception |
| `learning/synthetic_smoke.py` | synthetic smoke evidence | no | yes | migrate only in a dedicated compatibility cleanup | retain as frozen exception |
| `learning/observability_acceptance.py` | observability acceptance evidence | no | yes | migrate only in a dedicated compatibility cleanup | retain as frozen exception |

Test support is now a local package under `tests.support`; subprocess evidence
requires resolution inside this repository rather than an installed `tests`
package.

## Phase 4 development-path lock

Phase 4 is **Architecture Plugin Ingestion and First Real Architecture**. It
does not exist merely to implement RWKV. It defines and proves a standardized
architecture-ingestion process, with RWKV as the first serious reference
architecture.

Every Phase 4 checkpoint must either deepen that first plugin or prove that the
generic framework accepts it without architecture-specific contamination.
Architecture-neutral needs belong in generic contracts (for example typed carry
schema, parameter-layout support, stable objective surface, or tied-weight
metadata). Architecture-specific behavior remains inside the plugin.

Forbidden generic contamination includes a runtime RWKV mode, RWKV-specific
checkpoint branch, learning-loop time-mix flag, or objective logic keyed on
RWKV. Phase 4 does not include teacher loading, Tome payload production,
distillation, datasets, distributed/multi-device/Pallas work, production CLI,
serving, full HF export, or a second architecture before ingestion acceptance.

## Claims not made

`historical_acceptance_modules_not_fully_relocated`; `validation_namespace_not_fully_split`; `arbitrary_architecture_ingestion_not_yet_proven`; `rwkv_not_implemented`; `phase4_not_started`; `full_hf_export_not_implemented`; `save_pretrained_not_implemented`; `from_pretrained_not_implemented`; `teacher_inference_not_implemented`; `tome_training_not_started`; `distributed_training_not_proven`; `multi_device_training_not_proven`; `tpu_training_not_proven`; `pallas_optimization_not_started`; `production_cli_not_implemented`; `model_quality_not_measured`.
