# P1.3 Production Tome Contract Gap Report

Phase: 1 - Contract Layer

Status: review complete; implementation remains paused

## 1. Executive Summary

The current Student artifact model does **not** accurately represent the
production Tome.

The explicit recommendation is:

> **D. Both RADJAX-Contract and RADJAX-Student need coordinated changes.**

P1.1 and P1.2 established useful public seams: Student opens an artifact through
Contract, rejects structural blockers before model allocation, and separates
artifact-derived defaults from user choices. Those seams should remain. Their
current data model, however, was built around the earlier toy
`manifest.json`/`records.jsonl` shape and one payload with one expected adapter.

The production Tome has a different cover-page schema and at least two distinct
behavioral training surfaces:

```text
corridor surface
-> checkpoint boundary
-> exemplar surface
-> checkpoint boundary
```

Production `cover_page.json` is an identity, provenance, validation, and
role-indexed contents document. The published Contract model instead requires a
different v0 cover page plus `manifest.json`, and does not expose the production
corridor mode table, packed assignments, selected exemplar index, selected
payload shards, or ordered pass semantics. P1.1 therefore cannot open a real
production Tome through its current Contract calls. Even if it could, its
singular payload summary would erase training meaning. P1.2's singular
`expected_adapter_family` has the same problem.

Phase 1 should remain paused until Contract owns the production schema and
validation surface, the producer's cover-page index is complete for all
training-critical files, and Student receives focused P1.1/P1.2 corrections.
No corridor loader, exemplar loader, schedule, loss, runtime, or training work
should proceed first.

### Evidence boundary

This review used the required local design documents, current P1.1/P1.2 code,
the published `radjax_contract` package, and the current RADJAX-Tome production
cover-page, production-build, corridor, exemplar-selection, and
exemplar-delivery definitions on `nova-rey/RADJAX-Tome` `main`.

The named `RADJAX_TOME_STUDENT_CONSUMER_HANDOFF.md` was not present in the
provided attachments, the local checkouts, or RADJAX-Tome `main` at review time.
The production semantics restated in the P1.3 specification were treated as
normative, and current producer code was used to resolve concrete file and field
details. The handoff should be checked into a shared repository or Contract test
fixture before corrective implementation begins.

## 2. What Currently Aligns

- P1.1 treats artifact opening as a boundary and returns a stable Student-owned
  view rather than exposing downstream code to ad hoc path inspection.
- P1.1 delegates shared parsing and validation to RADJAX-Contract and does not
  import `radjax_tome` implementation modules.
- P1.1 accumulates structured blockers and rejects an artifact before training
  or model allocation.
- P1.1 already exposes teacher, tokenizer, vocabulary, sequence-length,
  record-count, provenance, warning, and payload-routing concepts.
- P1.2 correctly distinguishes artifact-derived values, choices required from
  the user, values unresolved by phase, warnings, and claims not made.
- The production producer and P1.1 both recognize `cover_page.json` as an
  important entry point. The mismatch is the schema and authority assigned to
  it, not the value of the seam.
- The architecture charter already places artifact normalization under
  `artifacts/`, compatibility under `validation/`, and schedule policy under
  `schedules/`. Corrective work can stay modular.

These alignments explain why P1.1/P1.2 should be corrected rather than removed.
The abstractions are useful; the provisional fixture beneath them is incomplete.

## 3. Production Tome Semantic Model

### Entry point and identity

Production `cover_page.json` is the semantic front door. Its top-level identity
includes `artifact_kind`, `cover_page_version`, `tome_version`, `layout`,
producer information, creation time, source artifact type, validation status,
and claims not made. It also summarizes teacher, tokenizer, target, corpus, and
teacher-model provenance.

Its `contents` list is intended to route consumers by role. Each entry is a
relative path with a role, SHA-256, and byte size. A consumer should validate
the listed paths, sizes, and hashes and select known roles. It should not walk
the directory looking for suggestive filenames, consume absolute paths, or
depend on operational files such as run plans and progress logs.

The current producer implementation indexes core sidecars, top-level corridor
JSON files, a human corridor summary, and target shards. It does not currently
index the packed assignment arrays and metadata, selected exemplar index,
selected payload shard, leaderboards, or delivery report. That is an upstream
contents-index gap: training-critical files cannot be safely discovered and
verified from the front door alone.

### Behavioral surfaces

The production Tome is not one generic payload. It contains at least:

1. A corridor surface that describes artifact-local modes and assigns token
   positions to those modes.
2. An exemplar surface that provides selected position-level teacher
   distributions linked back to corridor modes.

The artifact also communicates a recommended order with checkpoint boundaries.
That order is a declarative plan supplied by the artifact, not an implemented
Student schedule.

### Corridor semantics

The corridor surface includes:

- a summary with observation basis, degradation status, tracked statistics,
  mode and fingerprint counts, assignment storage kind and count, selected
  exemplar linkage, and delivery provenance;
- a mode policy and artifact-local mode table with mode IDs, mode keys, bounds,
  counts, and tracked statistics;
- a packed assignment manifest with relative array paths, dtypes, shapes, and
  example metadata;
- position-level assignment arrays including example index, position, mode ID,
  weight, and fingerprint index;
- a retained fingerprint table used for analysis and diagnostics.

The critical distinction is:

```text
mode_id != fingerprint_id
```

Modes define the training contract. Fingerprints explain or diagnose the
observations from which modes were derived. A fingerprint ID must never be used
as a mode ID, and neither identifier is globally meaningful outside its Tome.

```text
fingerprints are diagnostic
modes are training-critical
```

Within the packed assignments, `position_example_index`, `position`, `mode_id`,
`weight`, and the example-index metadata are training-critical. The
`fingerprint_index` is diagnostic unless a future version explicitly declares
a training use.

### Exemplar semantics

The exemplar surface separates selection metadata from selected payload data.
The selected index records winner identity, selected position, score/rank
policy, and corridor linkage. Selected payload shards contain the behavioral
target at each winning position, including:

- selected example ID and selected position;
- top token IDs;
- top probabilities and top log-probabilities;
- top-selection mask and effective top-k;
- top mass and tail mass;
- bucket masses and teacher entropy;
- dynamic top-k policy metadata;
- corridor mode ID, corridor fingerprint ID, and linkage status.

`corridor_mode_id` is training linkage. `corridor_fingerprint_id` is diagnostic
lineage. Delivery path records whether equivalent selected payloads came from a
one-pass candidate capture or a selected-example rerun. It is provenance and
operational evidence, not a reason for Student to choose different target math.

### Recommended training plan

The normalized implication is:

```text
pass 1: consume corridor surface
checkpoint: required
pass 2: consume exemplar surface
checkpoint: required
```

The artifact may recommend this plan and declare required consumer
capabilities. Student must still resolve architecture, runtime, optimizer,
budget, and policy details in their owning phases.

## 4. P1.1 Gaps

P1.1's `open_tome_artifact()` calls `validate_tome()`,
`load_tome_cover_page()`, and `inspect_tome_for_consumption()`. This is the
correct dependency direction but the wrong current schema. The Contract calls
expect the Contract v0 cover page and a `TomeManifest`; a production Tome uses
the producer's v1 cover page and sidecars instead. A valid production artifact
will therefore be rejected or misread.

Specific P1.1 gaps:

- **Student bug:** `manifest.json` is treated as required authority even though
  production starts from `cover_page.json` and role-indexed sidecars.
- **Student bug:** `TomePayloadSummary` flattens a multi-surface artifact into
  one `payload_format`, compression, shard set, and expected adapter.
- **Student missing view:** identity lacks Tome version, layout, source artifact
  type, creation metadata, and cover-page version.
- **Student missing view:** provenance is a generic manifest dictionary rather
  than normalized corpus and teacher provenance.
- **Student missing view:** there is no contents index with role, relative path,
  hash, size, and validation result.
- **Student missing view:** validation status and producer claims not made are
  not preserved.
- **Student missing view:** corridor summary, mode table, assignment manifest,
  diagnostic fingerprints, selected exemplar index, selected payload shards,
  and corridor linkage are absent.
- **Naming mismatch:** `teacher_id`/`teacher_family`/`backend` do not match the
  production `model_id`/`model_family`/`backend_type` names.
- **Naming mismatch:** one `payload_format` and `compression` do not faithfully
  represent production `target_type`, score-pass shards, corridor artifacts,
  and selected dynamic top-k payloads.
- **Compatibility risk:** `shard_paths` come from the old manifest instead of
  validated content roles. This makes future shard layout and `.rtome`
  packaging harder to support.

P1.1 does not itself blindly walk directories, which is good. It must continue
to avoid doing so after correction: Contract should validate the front door and
return a role-indexed content model.

## 5. P1.2 Gaps

P1.2 inherits P1.1's provisional single-payload assumptions.

- **Student bug:** `expected_adapter_family` is singular. Production requires
  at least separate corridor and exemplar capabilities.
- **Student missing view:** no available training surfaces are reported.
- **Student missing view:** no required consumer capabilities are reported.
- **Student missing view:** no recommended pass order or checkpoint boundaries
  are reported.
- **Student missing view:** artifact validation status and artifact claims not
  made are not carried into run defaults.
- **Naming mismatch:** `artifact_role` is an old manifest split role, not the
  production source artifact type or a behavioral surface role.
- **Too specific:** `expected_adapter_family` assumes adaptation is the only
  routing decision and that one adapter consumes the whole Tome.
- **Too generic:** `payload_format` and `compression_family` hide the distinct
  contracts and payload parameters of each surface.
- **Future-phase concern:** `schedule_policy` should remain unresolved, but an
  artifact-recommended pass graph must not be discarded merely because Student
  has not selected a schedule implementation.

P1.2 should infer facts, not silently convert artifact recommendations into
user policy. The corrected shape should expose available surfaces, capability
requirements, and recommended pass references while leaving optimizer,
budgets, exact loss weighting, architecture, runtime, and override policy
unresolved.

## 6. RADJAX-Contract Gaps

The published Contract package does not currently expose the necessary
production semantics.

Its cover-page type uses `cover_page_kind=radjax_tome_cover_page`, string
version `0`, and required summary sections such as `behavioral_fingerprint`,
`splits`, and `student_consumption`. Production uses
`artifact_kind=radjax_tome`, numeric cover-page and Tome versions, layout,
role-indexed contents, validation, and claims. These are incompatible schemas,
not optional-field drift.

Contract also centers `TomeManifest` and a single `TomePayloadFormat`. It has
enum names for fingerprint-corridor and exemplar-reservoir payloads and a
summary-level behavioral mode/exemplar count, but it does not model or validate
the production corridor or selected-exemplar file contracts.

Required Contract work:

- adopt or version a production cover-page schema and loader;
- validate artifact kind, versions, layout, safe relative content paths,
  content roles, byte sizes, SHA-256 values, validation status, and claims;
- model normalized identity, tokenizer/vocabulary, corpus provenance, teacher
  provenance, and source artifact type;
- define typed, versioned corridor summary, mode table, packed assignment
  manifest, assignment-array metadata, and sequence-position semantics;
- distinguish training-critical mode IDs from diagnostic fingerprint IDs;
- define typed selected exemplar index and selected payload shard schemas,
  including dynamic top-k, masks, masses, entropy, and corridor linkage;
- expose available surfaces, required consumer capabilities, and a declarative
  recommended pass order;
- provide compatibility inspection for multiple surfaces instead of one
  `adapter_id`;
- preserve unknown content roles and unknown future surface kinds so valid
  extensions can be inspected and rejected by capability, not by parser crash.

There is also a **producer documentation/implementation mismatch** to resolve.
The production cover-page documentation says contents are bound by role, hash,
and size, but the current producer index omits nested packed-assignment files
and selected exemplar files. Contract cannot validate files that the producer
does not list. The producer should emit complete content entries for every
training-critical file, while operational and human-only files should be
explicitly classified rather than discovered by Student.

### Ownership classification

Every identified gap maps to one of the P1.3 ownership classes:

| Gap | Classification |
| --- | --- |
| Production artifacts are rejected because P1.1 requires the old manifest and Contract-v0 cover shape | Student bug |
| P1.1 lacks normalized identity, provenance, contents, validation, claims, and per-surface projections | Student missing adapter/view |
| P1.2 collapses all required consumption into one expected adapter | Student bug |
| Contract lacks the production cover page, corridor, packed-assignment, selected-index, selected-payload, capability, and pass-plan schemas/APIs | Contract missing schema/API |
| Producer documentation describes a contents-bound artifact while current output omits packed assignment and selected exemplar files from `contents` | Producer documentation mismatch |
| Teacher ID/family/backend and record/payload field names differ across producer, Contract, and Student where the underlying concepts align | Naming mismatch only |
| `.rtome` direct consumption, unknown future surface kinds, functional stages, target regions, and hybrid architecture behavior | Future-phase concern |

## Required Gap Table

| Production concept | Contract support | Student support | Status | Required action |
| --- | --- | --- | --- | --- |
| Cover page entry point | Different v0 summary schema | Loads Contract cover page but also requires old manifest | Contract and Student mismatch | Version the production cover page in Contract; make Student start from it |
| Artifact kind, Tome version, layout | Partial manifest kind; no production Tome/layout model | Not normalized | Contract missing schema/API | Add versioned identity and preserve it in P1.1 |
| Vocab contract | Existing typed vocab contract | Exposed by P1.1/P1.2 | Partial alignment | Bind production `vocab_contract.json` by content role and hash |
| Validation status | Contract validates its own old layout | Warnings only; no producer status | Missing | Normalize producer validation status and report identity |
| Claims not made | No production cover-page field | Student adds only local static non-claims | Missing | Preserve producer claims separately from Student/run claims |
| Contents index | No role/path/size index matching production | No normalized index | Contract missing schema/API | Add safe role-indexed content references and verification results |
| Corridor summary | Behavioral counts only | Flattened payload summary | Too generic | Add a typed corridor surface summary |
| Corridor mode table | Mode IDs/count only in old cover summary | Absent | Missing | Contract the policy, artifact-local IDs, keys, bounds, stats, and counts |
| Packed assignments | Generic shard descriptors only | Absent | Missing | Contract manifest plus typed array metadata and relative paths |
| Diagnostic fingerprints | Payload enum and count only | Absent | Misnamed/underspecified | Model as diagnostic lineage, never as mode assignments |
| Selected exemplar index | Summary count/kinds only | Absent | Missing | Contract selected winners, positions, ranks/scores, and mode linkage |
| Exemplar payload shards | Generic exemplar enum only | Absent | Missing | Add typed selected payload shard schema and content roles |
| Dynamic top-k | Generic dynamic payload enum | Only one generic adapter name | Too generic | Model policy, requested/effective k, mask, and payload fields per surface |
| Corridor linkage | No production linkage schema | Absent | Missing | Validate mode ID, fingerprint ID, and linkage status independently |
| Delivery path | Not modeled | Absent | Diagnostic-only gap | Preserve as provenance; do not branch target math on Path A vs Path B |
| Pass ordering | One adapter only | One expected adapter; schedule unresolved | Incorrect cardinality | Add declarative ordered passes and checkpoint boundaries |
| Required consumer capabilities | One implemented flag/adapter | Singular adapter family | Too specific | Replace with a set of versioned capabilities per surface |
| Corpus and teacher provenance | Generic/old summary support | Generic manifest provenance | Partial but wrong source | Normalize production sidecars and preserve hashes/confidence |

## 7. Diagnostic vs Training-Critical File Classification

The classification below is semantic. All listed files may still be required
for artifact validation or provenance even when they are not loss inputs.

| File or field | Classification | Reason |
| --- | --- | --- |
| `cover_page.json` | Compatibility and integrity gate | Authoritative identity, routing index, versions, hashes, validation, and claims |
| `vocab_contract.json` | Training-critical compatibility | Token IDs and output dimensions are meaningless without the vocabulary contract |
| `metadata.json` and target shards | Training-critical | Define and carry scored examples, model inputs, and corridor source statistics |
| `teacher_manifest.json` and corpus/teacher provenance | Compatibility/provenance gate | Bind teacher, corpus, tokenizer, policy, and lineage; not direct loss tensors |
| `validation_report.json` | Integrity gate | Determines whether the producer considers the artifact valid |
| `corridors/corridor_summary.json` | Routing and compatibility gate | Declares surface completeness, degradation, counts, policies, and storage |
| `corridors/corridor_modes.json` | Training-critical | Defines artifact-local mode semantics and bounds used by corridor training |
| `corridors/mode_assignments.json` | Training-critical | Routes assignment arrays and declares their schema/count/storage |
| `position_example_index.npy`, `position.npy`, `mode_id.npy`, `weight.npy`, `examples_metadata.jsonl` | Training-critical | Maps example positions to weighted corridor modes |
| `fingerprint_index.npy` | Diagnostic | Links assignments to diagnostic fingerprints; it is not the mode target |
| `corridors/corridor_fingerprints.json` | Diagnostic | Explains observed teacher behavior and mode derivation |
| `corridors/corridor_summary.txt` | Human diagnostic | Duplicates a readable summary and must not drive behavior |
| `leaderboards/selected_exemplars.json` | Training-critical index | Selects the exemplar identities and token positions to consume |
| `selected_exemplars/selected-exemplars-*.json` | Training-critical | Carries selected dynamic top-k teacher targets and corridor mode linkage |
| `leaderboards/leaderboard_report.json` | Diagnostic/research provenance | Explains selection competition; it is not the selected target set |
| `delivery_report.json` and `source_delivery_path` | Diagnostic/provenance | Records how equivalent selected payloads were materialized |
| `run_plan.json`, progress logs, timing fields, production progress | Operational/diagnostic | Useful for producer operations but not a Student training contract |

The producer's content index should include every training-critical file and
all integrity gates. Diagnostic files should be included when they are part of
the durable artifact contract. Operational files should not become implicit
Student dependencies.

## 8. Recommended Normalized Student View

The long-term view should be generic at its core and offer corridor/exemplar
typed projections for today's production artifact:

```text
TomeArtifactView
  identity
    artifact_kind
    cover_page_version
    tome_version
    layout
    source_artifact_type
  provenance
    teacher
    tokenizer_and_vocab
    corpus
    teacher_model
  validation
    producer_status
    contract_status
    blockers
    warnings
  claims_not_made
  contents_index: ContentRef[]
  surfaces: BehavioralSurfaceContract[]
  recommended_training_plan: TrainingPassRecommendation[]

BehavioralSurfaceContract
  surface_id
  surface_kind
  schema_version
  content_roles
  required_capabilities
  prerequisites
  target_scope
  semantics
```

For the current Tome, typed accessors may expose:

```text
view.corridor_contract
view.exemplar_contract
```

Those accessors should project entries from `surfaces`; they should not force
the root model to assume every future Tome has exactly corridor and exemplar.

The recommended plan should reference surface IDs rather than implementation
classes:

```text
[
  {surface_id: "corridor", checkpoint_after: true},
  {surface_id: "exemplar", checkpoint_after: true}
]
```

`required_capabilities` should be a set such as versioned corridor assignment
consumption and selected dynamic-top-k exemplar consumption, not one adapter
string. `target_scope` should permit an unspecified whole-model default today
and future architecture-region declarations without changing the generic
surface interface.

### Forward compatibility rules

Student and Contract must not hard-code:

- mode count or fingerprint count;
- artifact-local mode IDs, fingerprint IDs, or their current numeric/string
  representation;
- selected exemplar count or selected positions;
- requested or effective top-k;
- target shard, selected payload shard, or assignment array count;
- delivery path;
- current filenames when a validated content role supplies the reference;
- absolute producer paths;
- exactly two behavioral surfaces;
- whole-model targeting for every surface.

Stable concepts suitable for implementation are the versioned semantic entry
point, safe relative content references, content roles, hashes and sizes,
tokenizer/vocabulary identity, producer validation and claims, distinct
behavioral surfaces, explicit sequence-position semantics, artifact-local ID
scope, the separation of training modes from diagnostic fingerprints, declared
consumer capabilities, and a declarative pass order with checkpoint
boundaries. Unknown roles and surface kinds should remain inspectable even when
Student cannot consume them.

## 9. Required Corrective Work

Corrective work should occur in this order:

1. **Freeze normative evidence.** Check the consumer handoff into a shared
   repository and add a small sanitized production Tome fixture that contains
   both corridor and selected exemplar surfaces.
2. **Complete the producer index.** Ensure `cover_page.json` lists every
   training-critical sidecar, packed assignment file, selected index, and
   selected payload shard with stable roles, relative paths, sizes, and hashes.
3. **Extend RADJAX-Contract.** Add production cover-page and sidecar schemas,
   validation APIs, normalized surface contracts, capability inspection, and
   ordered pass recommendations. Preserve forward-compatible unknown roles.
4. **Patch P1.1.** Keep `TomeArtifactView`, but make it cover-page-first,
   remove the production requirement for the old manifest, expose validated
   content references, and preserve separate surface contracts.
5. **Patch P1.2.** Replace singular payload/adapter inference with available
   surfaces, required capabilities, and artifact-recommended pass order while
   keeping user and later-phase choices unresolved.
6. **Add production-shaped tests.** Prove hash/size tampering, missing required
   roles, bad mode linkage, mode/fingerprint confusion, dynamic top-k shape
   errors, unknown future roles, and absolute/path-escape references fail or
   warn according to the Contract.
7. **Resume Phase 1.** Continue compatibility reports and diagnostics only
   after Student can truthfully say it understands or rejects the production
   fixture before allocating model weights.

This should be a focused correction, not a wholesale rewrite. P1.1's public
opening seam and P1.2's source-separated defaults model remain useful.

## 10. Work Explicitly Deferred

This report does not add or authorize:

- corridor array loaders;
- selected exemplar loaders;
- target adapters;
- corridor or exemplar losses;
- training loops or schedule execution;
- checkpoint implementation;
- JAX, runtime, sharding, or accelerator behavior;
- architecture plugin behavior;
- CLI commands;
- `.rtome` extraction or streaming consumption;
- functional-stage distillation;
- architecture-region targeting;
- hybrid architecture experiments;
- model quality, performance, or parity evaluation.

## 11. Impact on Remaining Phase 1 Checkpoints

The next Phase 1 checkpoint must be production Contract alignment, not payload
loading. Compatibility checking, provenance reports, diagnostics, and future
CLI inspection all depend on the corrected cover-page and surface model.

P1.1 and P1.2 should be marked provisionally useful but production-incomplete.
Their focused corrections should land only after the Contract and producer
contents roles are stable. Any remaining Phase 1 checkpoint that assumes a
single payload, a single adapter, or `manifest.json` as the production root must
be revised before implementation.

The research extensibility constraint reinforces this ordering. RADJAX must be
able to consume multiple sequential behavioral surfaces without assuming those
surfaces are always corridor and exemplar or always train the entire
architecture. A generic `surfaces` collection, capability declarations,
surface-referenced pass plan, and optional target scope leave room for
functional-stage schedules, architecture-region targeting, hybrid students,
and additional behavioral surfaces. Implementing those research ideas remains
deferred.

## 12. Claims Not Made

This report does not claim:

- that the unavailable handoff document was reviewed directly;
- that any production Tome was successfully opened by Student;
- that current RADJAX-Contract validates the production Tome;
- that the proposed normalized field names are final schemas;
- that corridor or exemplar training semantics have been implemented;
- that the recommended pass order is an executable schedule;
- that all future behavioral surfaces are known;
- that whole-model or architecture-region targeting is settled;
- that training, checkpointing, JAX portability, Hugging Face export, model
  quality, delivery-path quality parity, or RADLADS parity works.

The claim made is narrower: the current Contract/P1.1/P1.2 model is not
sufficient for the production artifact, the ownership of the principal gaps is
now identified, and correcting those contract seams before loaders or training
will prevent the consumer layer from hardening around the wrong shape.
