# P1.4 Production Tome Contract Alignment Plan

Phase: 1 - Contract Layer

Status: planning complete; cross-repository implementation not started

## 1. Executive Summary

P1.3 established that the current production Tome, RADJAX-Contract, and
RADJAX-Student do not share one artifact model. P1.4 converts that diagnosis
into an ordered implementation plan. It does not implement any producer,
Contract, Student reader, payload loader, schedule, or training behavior.

The dependency order is mandatory:

```text
1. Freeze normative evidence
2. Complete Tome content indexing
3. Add production Tome schemas to Contract
4. Add shared production-shaped golden fixture
5. Patch Student P1.1
6. Patch Student P1.2
7. Resume remaining Phase 1 work
```

Each repository owns one kind of truth:

- RADJAX-Tome owns what it emits and the producer-side recommendation.
- RADJAX-Contract owns shared meaning, schemas, validation, and compatibility
  inspection.
- RADJAX-Student owns normalized consumption and Student-specific readiness.

Student must not compensate for missing upstream truth by walking directories,
guessing filenames, importing producer code, or embedding a private copy of the
schema. A failed upstream gate stops downstream implementation.

The P1.1 and P1.2 public seams survive:

```python
view = open_tome_artifact(path)
defaults = infer_run_defaults(view)
```

Their provisional single-manifest, single-payload, and single-adapter internals
do not survive. They will be corrected only after the production contract and
golden fixture are accepted.

### Evidence baseline

This plan is based on:

- RADJAX-Tome `main` at `c8fb9ac4d92a33d8342c2249bae939e21221125a`;
- RADJAX-Contract `main` at `a914601ebd59f5690c329e12a1c1e26dd27db70f`;
- RADJAX-Student `main` at `6289af21e0d8717199ab2961df2912f1d329956b`;
- [P1.3 Production Tome Gap Report](P1_3_PRODUCTION_TOME_GAP_REPORT.md);
- current production cover-page, corridor, and selected-exemplar emission code.

`RADJAX_TOME_STUDENT_CONSUMER_HANDOFF.md` was not present in the supplied
attachments or any involved repository at this baseline. P1.4A therefore
requires it to be recovered, reviewed against the current producer, and frozen
before schema implementation. No implementation may treat this plan as a
replacement for missing normative evidence.

## 2. Cross-Repository Ownership Map

| Concern | RADJAX-Tome owns | RADJAX-Contract owns | RADJAX-Student owns |
| --- | --- | --- | --- |
| Consumer handoff | Producer review and factual corrections | Canonical durable copy and version history | Review against consumer needs |
| Cover-page entry point | Emit `cover_page.json` last from finalized durable files | Define and validate the versioned model | Open through Contract APIs |
| Artifact identity | Emit kind, versions, layout, source type, creation data | Typed identity schema and compatibility rules | Expose normalized identity |
| Content roles | Emit a complete deterministic role index | Define/version roles, cardinality, and classification | Consume known roles; preserve unknown metadata |
| Relative paths | Emit artifact-relative POSIX paths only | Reject absolute, escaping, duplicate, or unsafe references | Never repair or reinterpret unsafe paths |
| Hash and size integrity | Compute values from finalized bytes | Recompute and validate SHA-256 and byte size | Surface Contract blockers |
| Required/optional content | Declare entries and surface requirements truthfully | Model and validate conditional requirements | Report missing required capabilities/content |
| Producer validation | Emit validation status and report | Bind status to a validated report reference | Preserve producer and Contract status separately |
| Teacher provenance | Emit model identity and source evidence | Type/version shared provenance | Expose normalized provenance |
| Corpus provenance | Emit source, split, and integrity evidence | Type/version shared provenance | Expose normalized provenance |
| Tokenizer/vocabulary | Emit tokenizer identity and vocab contract | Define and validate compatibility | Expose facts for architecture/HF checks |
| Behavioral surfaces | Emit all available surfaces | Define generic extensible surface model | Expose a surface collection |
| Corridor surface | Emit summary, modes, assignments, and diagnostics | Type and validate every durable corridor contract | Expose normalized corridor projection |
| Mode IDs | Emit artifact-local training identifiers | Give mode IDs a distinct typed semantic identity | Never treat them as fingerprint IDs |
| Fingerprint IDs | Emit diagnostic lineage identifiers | Type separately from mode IDs | Expose as diagnostic lineage only |
| Exemplar surface | Emit selected index, payload shards, and linkage | Type and validate exemplar records/payloads | Expose normalized exemplar projection |
| Dynamic top-k | Emit policy, masks, effective k, values, and masses | Validate shapes, domains, and cross-field consistency | Report capability requirements only in P1.1/P1.2 |
| Delivery path | Emit truthful provenance | Preserve as diagnostic provenance | Do not branch target semantics on delivery path |
| Required capabilities | Emit versioned capability identifiers per surface/pass | Model sets and inspect consumer compatibility | Compare with Student support in later Phase 1 work |
| Recommended pass order | Emit surface references and checkpoint boundaries | Model declaratively; validate references | Expose as inferred artifact fact |
| Schedule execution | No | No | Later Student schedule phase |
| Golden fixture | Deterministic generation recipe and producer conformance | Canonical accepted bytes and fixture access | Consume exact Contract fixture in default CI |
| Artifact view | No | Shared parsed contracts only | Own `TomeArtifactView` and convenience projections |
| Run defaults | No | No Student policy | Own source-separated inferred defaults |
| Compatibility report | Producer facts only | Shared structural validation result | Own Student capability/readiness result |
| Operational reports | Emit and classify intentionally | Validate only when declared durable contract content | Ignore unless a later explicit feature consumes them |
| Payload loading/training | No Student behavior | No | Deferred Student phases |

Dependency direction remains:

```text
RADJAX-Tome -> artifact files <- RADJAX-Contract <- RADJAX-Student
```

Tome must not depend on Student. Contract must not depend on either
implementation. Student must not import Tome.

## 3. Required RADJAX-Tome Changes

### 3.1 Semantic front door

Tome must keep `cover_page.json` as the semantic front door. It must be written
after every durable content file is finalized so hashes and byte sizes describe
the bytes a consumer will read. The cover page must not hash itself.

The producer must emit:

- artifact kind;
- cover-page version and Tome version;
- layout and source artifact type;
- deterministic creation/producer metadata;
- teacher, corpus, tokenizer, and vocabulary provenance;
- producer validation status and validation-report reference;
- claims not made;
- complete durable contents index;
- available behavioral surfaces;
- required consumer capabilities;
- recommended pass order with checkpoint boundaries.

### 3.2 Content reference shape

Every durable content entry must contain:

```text
role
path
sha256
size_bytes
required
classification
```

`path` must be a relative POSIX path rooted at the unpacked artifact. Absolute
paths, `..`, empty paths, duplicate paths, symlink escapes, and platform-specific
producer paths are forbidden. Entries should be emitted in deterministic path
order.

`classification` is one of:

```text
training_critical
integrity_or_provenance
diagnostic
human_readable
operational
```

Operational files should normally stay outside the durable contents index. If
an operational file is intentionally made durable, its role and classification
must be explicit and Contract-versioned.

### 3.3 Initial role registry

Contract owns the final names and versions; Tome must emit the accepted names.
The first production registry must cover at least:

| Proposed role | Cardinality | Classification | Requirement |
| --- | --- | --- | --- |
| `target_store_metadata` | one | `integrity_or_provenance` | always required |
| `vocab_contract` | one | `training_critical` | always required for compatibility |
| `teacher_manifest` | one | `integrity_or_provenance` | always required |
| `emission_config` | one | `integrity_or_provenance` | always required |
| `validation_report` | one | `integrity_or_provenance` | always required as integrity gate |
| `target_shard` | one or more | `training_critical` | required by current production surfaces |
| `corridor_summary` | one | `integrity_or_provenance` | required as corridor routing gate |
| `corridor_mode_table` | one | `training_critical` | required by corridor surface |
| `corridor_assignment_manifest` | one | `training_critical` | required by corridor surface |
| `corridor_assignment_position_example_index` | one | `training_critical` | required by packed assignments |
| `corridor_assignment_position` | one | `training_critical` | required by packed assignments |
| `corridor_assignment_mode_id` | one | `training_critical` | required by packed assignments |
| `corridor_assignment_weight` | one | `training_critical` | required by packed assignments |
| `corridor_assignment_examples_metadata` | one | `training_critical` | required by packed assignments |
| `corridor_assignment_fingerprint_index` | zero or one | `diagnostic` | optional diagnostic linkage |
| `corridor_fingerprints` | zero or one | `diagnostic` | optional but expected in the golden fixture |
| `selected_exemplar_index` | one | `training_critical` | required by exemplar surface |
| `selected_exemplar_payload_shard` | one or more | `training_critical` | required by exemplar surface |
| `exemplar_delivery_report` | zero or one | `diagnostic` | optional delivery provenance |
| `exemplar_leaderboard_report` | zero or one | `diagnostic` | optional |
| `corridor_human_summary` | zero or one | `human_readable` | optional |

Current packed assignment arrays and selected exemplar files are not all in the
producer's content index. P1.4B is incomplete until they are listed, hashed,
sized, and covered by producer validation.

### 3.4 Surface and plan emission

Tome must emit a generic surface collection. The current artifact should have
distinct corridor and exemplar entries, but the format must permit additional
surface kinds and more than two entries.

Each surface must declare:

- artifact-local `surface_id`;
- extensible `surface_kind`;
- schema version;
- required and optional content roles;
- required consumer capability set;
- prerequisite surface IDs;
- explicit target scope;
- versioned semantic metadata.

The producer should currently emit a whole-model or unspecified target scope.
It must not invent named architecture regions before that research contract
exists.

The recommended training plan must reference surface IDs, not Python classes,
adapter classes, CLI commands, or schedule implementations. The first plan is:

```text
corridor pass -> checkpoint -> exemplar pass -> checkpoint
```

The recommendation is artifact metadata. It is not executable schedule code.

### 3.5 Producer prohibitions

Tome must not ask Student to:

- discover durable files by directory walking;
- infer a role from a filename;
- follow absolute paths in reports;
- distinguish delivery paths to recover target semantics;
- import Tome implementation code;
- treat fingerprints as mode assignments;
- infer pass order from file presence.

## 4. Required RADJAX-Contract Changes

Contract must add an additive, versioned production Tome model. Existing v0
fixtures may remain as legacy coverage, but they must not define production
semantics.

### 4.1 Artifact identity and content references

Contract must own typed models for:

```text
TomeArtifactIdentity
  artifact_kind
  cover_page_version
  tome_version
  layout
  source_artifact_type
  created_by
  created_at

TomeContentRef
  role
  path
  sha256
  size_bytes
  required
  classification
  known_role
```

Role values must preserve unknown strings. A strict enum constructor that
throws on a future role is not acceptable. `known_role` and capability
inspection determine consumability; parsing preserves the original value.

Validation must enforce safe relative containment, uniqueness, cardinality,
required roles, file existence, exact size, and exact SHA-256. Unknown content
may be structurally valid and inspectable while still causing an unsupported
capability blocker when required by a surface.

### 4.2 Validation and provenance

Contract must model and validate:

- producer validation status and referenced report;
- producer claims not made;
- tokenizer identity, tokenizer hash, vocabulary size, and special tokens;
- corpus provenance, source hashes, split role, and disjointness evidence when
  declared;
- teacher model identity, family/backend, revision, weights/config hashes,
  source kind, local/download policy, and confidence fields when declared.

Producer validation and Contract validation are separate facts. A producer
status of `pass` does not replace Contract validation, and Contract must not
rewrite producer status.

### 4.3 Generic behavioral surfaces

Contract should define a generic model equivalent to:

```text
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

Requirements:

- surface IDs are unique and artifact-local;
- surface kinds preserve unknown values;
- content-role references resolve to indexed content;
- required capabilities are a set of versioned strings;
- prerequisites reference existing surface IDs and form an acyclic graph;
- `target_scope` is versioned and extensible;
- unknown surfaces parse without failure but are not automatically consumable.

Contract parses and validates the declaration. It does not execute the surface
or choose a schedule.

### 4.4 Corridor contract

Typed, versioned corridor models must cover:

- summary completeness and degradation status;
- observation basis and sequence-position semantics;
- mode policy and tracked statistics;
- artifact-local mode table, mode keys, bounds, and assignment counts;
- packed assignment manifest, storage kind, array metadata, and example
  metadata;
- array dtype, shape, rank, length, and cross-array consistency;
- optional diagnostic fingerprint table and fingerprint-index array;
- selected exemplar linkage status where declared.

Mode and fingerprint identifiers must be structurally distinct, for example by
dedicated `CorridorModeId` and `CorridorFingerprintId` wrappers or equivalent
typed fields. Both remain opaque and artifact-local. Contract must make this
invalid state unrepresentable:

```text
fingerprint ID used where a mode ID is required
```

Position semantics must declare index base, valid sequence range, padding/mask
behavior, and whether assignments cover all valid positions or an explicitly
degraded subset.

### 4.5 Exemplar contract

Typed, versioned exemplar models must cover:

- selected exemplar index and selected payload shards;
- selected example identity and selected position;
- rank/score/selection policy metadata;
- top token IDs, probabilities, and log-probabilities;
- top-selection mask and effective top-k;
- top mass, tail mass, bucket masses, and teacher entropy;
- dynamic top-k policy and requested/effective bounds;
- corridor mode linkage, diagnostic fingerprint linkage, and linkage status;
- delivery-path provenance without making it target semantics.

Validation must check record uniqueness, index-to-payload agreement, selected
position range, vector/mask lengths, effective-k consistency, token ID range,
finite numeric values, probability domains, versioned mass tolerances, bucket
shape, and corridor-link resolution.

### 4.6 Capability and pass-plan inspection

The singular `adapter_id` consumption plan must gain a production multi-surface
replacement. Contract must expose:

```text
available_surfaces
required_capabilities
recommended_training_plan
structural_blockers
unsupported_capability_blockers
warnings
```

The plan schema should contain a version and ordered pass records:

```text
pass_id
surface_id
required_capabilities
checkpoint_after
```

All surface references and capability requirements must validate. The schema
must not contain executable callables, implementation classes, optimizer names,
runtime choices, or architecture selections.

## 5. Required Shared Fixture and Handoff Changes

### 5.1 Canonical normative location

The canonical handoff location will be:

```text
RADJAX-Contract/docs/RADJAX_TOME_STUDENT_CONSUMER_HANDOFF.md
```

Contract is the neutral schema authority shared by producer and consumer. Tome
and Student should link to this file rather than maintain competing normative
copies. Changes require review from both producer and Student perspectives and
must identify the first Contract version that implements them.

### 5.2 Canonical golden fixture location

The canonical static fixture will be installed from:

```text
RADJAX-Contract/src/radjax_contract/testing/fixtures/
  production_multi_surface_v1/
    FIXTURE_PROVENANCE.json
    artifact/
      cover_page.json
      ... indexed artifact contents ...
```

Contract should expose a testing-only resource helper such as:

```python
from radjax_contract.testing import production_tome_fixture_path
```

The fixture must be package data so Tome, Contract, and Student default CI can
read the same immutable bytes without a network request, repository submodule,
or copied downstream fixture. Production runtime code must not depend on the
testing namespace.

`FIXTURE_PROVENANCE.json` sits outside the artifact root and records fixture ID,
fixture schema version, producer commit, Contract acceptance commit, generation
command/seed, artifact tree digest, and claims not made. It must not contain
machine-specific absolute paths.

### 5.3 Fixture contents

Use 12 deterministic synthetic examples. The fixture must include:

- a valid production cover page with a complete contents index;
- vocabulary contract and tokenizer identity/hash;
- producer validation report;
- teacher and corpus provenance using fake/offline identities;
- target metadata and small target shards;
- at least two non-empty corridor modes;
- packed assignments and all training-critical arrays;
- example-index metadata;
- diagnostic fingerprint table and fingerprint-index array;
- four selected exemplars across valid positions;
- selected index and at least one selected payload shard;
- dynamic effective top-k values that differ, for example `2, 3, 4, 5`;
- masks, probabilities, log-probabilities, top/tail mass, bucket masses, and
  entropy for each selected exemplar;
- valid mode linkage and separate diagnostic fingerprint linkage;
- corridor and exemplar surface declarations;
- capability sets;
- recommended corridor-to-exemplar passes with both checkpoint boundaries;
- valid relative paths, hashes, and sizes;
- explicit non-claims for training, quality, parity, and network verification.

The artifact must remain small enough for normal source control and default CI.
It must use only stdlib/NumPy-compatible files and require no JAX, torch,
transformers, accelerator, teacher model, or network access.

### 5.4 Fixture production and acceptance

Tome owns a deterministic fake-backend generation recipe. Contract accepts the
result only after its production validators pass and the handoff checklist is
satisfied. The frozen fixture is not regenerated implicitly during tests.

Tome tests must prove a fresh deterministic emission conforms semantically to
the same Contract schemas. Contract tests validate the frozen bytes. Student
tests consume the exact installed Contract resource. Cross-repository
acceptance records the three tested commit SHAs and the fixture tree digest.

Any semantic fixture change requires a new fixture version or an explicitly
reviewed correction with a new tree digest. Silent byte replacement is
forbidden.

## 6. Required Student P1.1 Corrections

Keep:

```python
view = open_tome_artifact(path)
```

Keep `TomeArtifactView` as the Student-owned stable boundary. Replace its
provisional internals only after P1.4A-G pass.

The corrected view must expose:

```text
TomeArtifactView
  artifact_dir
  identity
  provenance
  validation
  claims_not_made
  contents_index
  surfaces
  recommended_training_plan
  warnings
```

Requirements:

- call published production Contract parsing, validation, and inspection APIs;
- make `cover_page.json` the production entry point;
- remove production dependence on provisional `manifest.json`;
- preserve producer and Contract validation separately;
- surface all Contract structural blockers before returning a view;
- expose an ordered immutable content index without opening payload tensors;
- expose an arbitrary-length surface collection;
- expose required capability sets and the declarative pass plan;
- preserve unknown roles/surfaces for inspection;
- make unsupported required capabilities explicit rather than pretending the
  artifact is understood;
- avoid raw directory walking, filename inference, raw JSON parsing, and Tome
  imports;
- allocate no model and run no training.

Typed convenience projections may include:

```text
view.corridor_contract
view.exemplar_contract
```

They are lookups over `surfaces`, not root cardinality assumptions. Their
absence, duplication, or unsupported version must have explicit behavior.

The provisional `manifest`, `payload_summary`, `payload_format`, and inferred
single-adapter fields must not remain production sources of truth. If legacy
fixture support remains temporarily, it must be an explicit legacy adapter with
separate tests and no effect on the production view.

## 7. Required Student P1.2 Corrections

Keep:

```python
defaults = infer_run_defaults(view)
```

Keep the source separation:

- `inferred_from_tome`;
- `required_from_user`;
- `unresolved_by_phase`;
- warnings;
- Student claims not made.

Replace production inference of singular `payload_format` and
`expected_adapter_family` with:

```text
available_surfaces
required_capabilities
recommended_training_plan
artifact_validation_status
artifact_claims_not_made
```

`available_surfaces` must preserve artifact order and identify each surface by
ID, kind, and schema version. `required_capabilities` must be a deterministic
deduplicated set/list, not a single adapter. The recommended plan must remain
data that references surface IDs and checkpoint boundaries.

Producer claims and Student run non-claims must remain separate fields. Passing
artifact validation must not imply Student compatibility, training success, or
model quality.

User-required fields remain exactly the choices the Tome cannot know:

```text
student_architecture
student_size_or_config
training_budget
output_dir
```

P1.2 must not choose an architecture, runtime, optimizer, schedule
implementation, loss weighting, evaluation policy, or Hugging Face export
details.

## 8. Ordered Implementation Sequence

Each checkpoint has one owning repository and a hard entry/exit gate.

| Checkpoint | Owner | Work | Entry gate | Exit gate |
| --- | --- | --- | --- | --- |
| P1.4A - Freeze evidence | Contract with Tome/Student review | Recover handoff, install canonical copy, approve role/fixture requirements | P1.4 plan accepted | Handoff exists and all three repositories link to one version |
| P1.4B - Complete content index | Tome | Emit all durable roles, hashes, sizes, classifications, surfaces, capabilities, and pass plan | P1.4A complete | Producer validates complete deterministic index; no training-critical file is unindexed |
| P1.4C - Production cover schema | Contract | Add identity, content reference, provenance, validation, and claims models | P1.4B format frozen | Real producer cover page parses; path/hash/size tampering fails |
| P1.4D - Corridor schema | Contract | Type summary, modes, packed assignments, positions, and diagnostics | P1.4C complete | Corridor contract validates; mode/fingerprint confusion fails |
| P1.4E - Exemplar schema | Contract | Type selected index, payloads, dynamic top-k, masses, and linkage | P1.4D ID/link types stable | Exemplar contract and linkage validate; malformed dynamic top-k fails |
| P1.4F - Multi-surface inspection | Contract | Add generic surfaces, capability sets, prerequisites, and pass recommendations | P1.4D/E complete | Unknown surfaces preserve; unsupported required capabilities block consumption |
| P1.4G - Golden fixture | Tome then Contract | Deterministically emit and freeze shared fixture | P1.4B-F complete | Frozen fixture passes Tome and Contract checks in default CI |
| P1.4H - Correct P1.1 | Student | Replace internals behind `open_tome_artifact()` and `TomeArtifactView` | P1.4G available through pinned Contract | Production fixture opens; malformed variants fail before tensor/model allocation |
| P1.4I - Correct P1.2 | Student | Infer surfaces, capabilities, plan, validation, and artifact claims | P1.4H complete | Defaults serialize normalized facts and leave user/later-phase choices unset |
| P1.4J - Cross-repo acceptance | All; Contract records matrix | Run pinned producer, Contract, and Student acceptance matrix | P1.4A-I green independently | Three SHAs, fixture digest, validations, and explicit non-claims recorded |

No checkpoint may be implemented downstream of a failed exit gate. Combining
commits is allowed only when ownership remains isolated and the same gates are
demonstrably evaluated in order.

## 9. Acceptance Criteria by Repository

### 9.1 RADJAX-Tome acceptance

- `cover_page.json` remains the semantic entry point and is emitted after
  durable contents are finalized.
- Every training-critical and integrity/provenance file is indexed.
- Every indexed path is relative, POSIX-normalized, safe, and unique.
- Every indexed hash and byte size matches finalized bytes.
- Stable roles, classifications, required flags, and cardinalities are
  documented.
- Packed assignment manifest, arrays, and example metadata are indexed.
- Corridor summary and mode table are indexed and declared as one surface.
- Selected exemplar index and every payload shard are indexed and declared as
  a separate surface.
- Diagnostic fingerprints and delivery evidence are classified intentionally.
- Teacher, corpus, tokenizer, and vocab provenance are preserved.
- Capability sets and a surface-referenced pass order with checkpoint
  boundaries are emitted.
- Producer validation fails on omitted required roles, unsafe paths, or stale
  hash/size values.
- No Student imports or Student-specific implementation names are introduced.

### 9.2 RADJAX-Contract acceptance

- The production cover page parses into typed identity/content/provenance
  models and validates against actual bytes.
- Unknown roles and surface kinds preserve their raw values without parser
  failure.
- Unknown required capabilities make consumption unsupported, not structurally
  unreadable.
- Content paths reject absolute paths, traversal, duplicates, and containment
  escapes.
- Corridor and exemplar contracts are typed and versioned.
- Mode IDs and fingerprint IDs are structurally distinct and artifact-local.
- Packed array metadata and cross-array lengths validate.
- Dynamic top-k masks, effective values, vectors, masses, entropy, and token
  domains validate.
- Exemplar index/payload and corridor linkage validate.
- Multiple surfaces, capability sets, prerequisites, and ordered pass records
  validate.
- Pass plans contain no executable schedule behavior.
- The frozen fixture passes default CI without optional ML dependencies or
  network access.
- Legacy v0 behavior is either preserved additively or retired through an
  explicit versioned migration, never silently reinterpreted.

### 9.3 RADJAX-Student acceptance

- `open_tome_artifact()` opens the exact installed production fixture.
- The reader uses Contract APIs and does not import `radjax_tome`.
- The reader does not walk directories, infer filenames, or parse producer JSON
  privately.
- The returned view exposes identity, provenance, validation, claims, content
  references, arbitrary surfaces, capabilities, and recommended passes.
- Corridor/exemplar projections are optional typed lookups over surfaces.
- Producer and Contract validation facts remain distinguishable.
- Unknown optional roles/surfaces remain inspectable.
- Unknown required capabilities yield explicit compatibility blockers.
- Malformed paths, hashes, sizes, mode links, dynamic top-k, and pass references
  fail with stable explicit blockers.
- No target tensors are loaded and no model is allocated by P1.1.
- `infer_run_defaults()` reports surfaces, capabilities, validation, artifact
  claims, and pass order.
- User-required fields remain unset and later-phase choices remain unresolved.
- No runtime, optimizer, schedule implementation, architecture, or training is
  selected or executed.

### 9.4 Cross-repository acceptance

- The handoff has one canonical version and all repositories link to it.
- The fixture has one canonical byte set, fixture ID, and tree digest.
- Tome can emit a semantically conforming artifact from deterministic fake
  inputs.
- Contract validates the frozen fixture and expected malformed variants.
- Student opens the same fixture through a pinned Contract version.
- The acceptance receipt records all three commit SHAs, Contract version,
  fixture digest, checks run, results, and claims not made.
- The full matrix runs in default CPU CI with no network and no JAX, torch,
  transformers, datasets, or accelerate imports.

## 10. Rollback and Failure Conditions

Stop and revise the owning upstream checkpoint before Student work if:

- the handoff remains unavailable, ambiguous, or inconsistent with production;
- Tome omits a training-critical or integrity role;
- Tome emits stale hashes/sizes or unsafe/absolute paths;
- Contract requires consumers to know producer filenames;
- Contract flattens multiple surfaces into one payload or one adapter;
- mode IDs and fingerprint IDs are not structurally distinguishable;
- dynamic top-k cannot be validated from typed fields;
- selected payloads cannot be joined to the selected index and corridor modes;
- capability requirements are singular, implicit, or implementation-specific;
- pass ordering names an executable class or cannot reference arbitrary
  surfaces;
- unknown future roles/surfaces crash parsing;
- the fixture omits production semantics or needs heavy/network dependencies;
- Student must import Tome, walk directories, or parse raw producer files to
  proceed.

Rollback policy:

1. Do not patch around the failure downstream.
2. Keep the last accepted schema/fixture version immutable.
3. Revert the owning unaccepted change or issue a new additive schema/fixture
   version; do not silently change meaning under an existing version.
4. Keep Student pinned to the last accepted Contract commit/version until the
   new cross-repository gate passes.
5. Preserve legacy tests while an additive production path is being proven.
6. Rerun every downstream gate after an upstream schema or fixture change.
7. Record the failed assumption and resolution in the owning repository's
   append-only Bible or decision log.

A rollback restores the last accepted contract boundary. It does not authorize
Student-local copies of missing upstream semantics.

## 11. Work Explicitly Deferred

P1.4 does not implement or authorize:

- corridor batch or array loading;
- exemplar batch or payload loading;
- target adapters;
- corridor or exemplar losses;
- JAX or XLA behavior;
- training loops;
- checkpoint execution;
- runtime selection;
- architecture plugins;
- schedule execution or policy selection;
- Hugging Face export;
- `.rtome` direct consumption;
- functional-stage distillation;
- architecture-region targeting;
- hybrid architecture experiments;
- quality, performance, scale, delivery-path quality parity, or RADLADS parity
  experiments.

The alignment schemas may reserve extensible `surface_kind`, `target_scope`,
capability, and pass-reference fields. Reserving those seams is not an
implementation of future research.

## 12. Revised Phase 1 Roadmap

```text
P1.1  TomeArtifactView seam                         COMPLETE, PROVISIONAL
P1.2  Inferred run-defaults seam                    COMPLETE, PROVISIONAL
P1.3  Production Tome gap review                    COMPLETE
P1.4  Cross-repository contract alignment plan     COMPLETE
P1.5  Tome/Contract production alignment           NEXT; EXECUTES P1.4A-G
P1.6  Student artifact-view correction              BLOCKED ON P1.5
P1.7  Student run-defaults correction               BLOCKED ON P1.6
P1.8  Student compatibility report                  BLOCKED ON P1.7/P1.4J
P1.9  Inspect/doctor CLI                            BLOCKED ON P1.8
P1.10 Production golden acceptance gate             BLOCKED ON P1.9
```

P1.5 executes P1.4A-G and produces the canonical fixture before Student code is
patched. P1.10 does not create a late fixture; it promotes the already-frozen
fixture and malformed variants into the maintained Phase 1 exit gate across
inspection, compatibility reporting, and CLI behavior.

Phase 1 exit requires Student to say either "I understand this artifact" or "I
reject this artifact" from production-shaped evidence before allocating model
weights.

The governing rule is:

```text
No corridor/exemplar loader work until production contract alignment is complete.
```

## 13. Claims Not Made

P1.4 does not claim:

- the missing consumer handoff has been recovered or approved;
- producer content-index changes are implemented;
- the proposed role or type names are final;
- RADJAX-Contract parses or validates the production Tome;
- the shared golden fixture or testing helper exists;
- RADJAX-Student opens a production Tome;
- Student P1.1 or P1.2 corrections are implemented;
- corridor or exemplar payload loading works;
- capability support or schedule execution exists;
- training, checkpointing, runtime, JAX portability, architecture plugins, or
  Hugging Face export works;
- functional-stage research or architecture-region targeting is implemented;
- model quality, performance, scale, delivery-path quality parity, or RADLADS
  parity is proven.

The claim made is limited: ownership, dependency order, schemas to define,
fixture evidence, repository acceptance gates, and rollback conditions are now
specified well enough to begin upstream alignment without pushing producer or
Contract responsibilities into Student.
