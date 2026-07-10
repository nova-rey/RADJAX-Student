# P1.6 Student Artifact View

`open_tome_artifact(path)` is the stable Student-owned entry point for Tome
metadata. P1.6 keeps that seam and replaces its provisional production
assumptions with the accepted RADJAX-Contract production API.

## Production Path

The production path is cover-page-first and exposes:

- normalized artifact identity and creation metadata;
- teacher, tokenizer, target, corpus, teacher-model, and producer provenance;
- separate producer, Contract, and Student-interpretation validation facts;
- artifact claims not made;
- the ordered validated content-reference index;
- arbitrary behavioral surfaces with target scope and semantic metadata;
- required and currently unsupported capability sets;
- the Contract-validated declarative training-pass plan.

Student does not require `manifest.json` for production artifacts. It does not
parse producer JSON, walk directories, guess filenames, or copy Contract
schemas. Contract blocker strings are retained by `TomeArtifactError`.

## Metadata Projections

`view.corridor_contract` and `view.exemplar_contract` are optional projections
over Contract-validated surfaces. They summarize routing metadata only.

The corridor projection exposes stat names, mode policy and count, assignment
count/storage, stat support depth, capability requirements, and content
references. It keeps mode and fingerprint identifier types distinct and does
not expose assignment arrays or fingerprint records.

The exemplar projection exposes selected count, payload-shard references,
cover-page dynamic-top-k parameters, corridor linkage requirements,
capabilities, and content references. It does not expose selected payload
records. Delivery route is not a target-behavior input.

The pass plan remains data. Checkpoint flags do not execute checkpoints or
select a Student schedule.

## Legacy Dense Smoke Support

Existing dense-logits tests and debug utilities remain available through an
explicit `legacy_dense_v0` branch. Legacy `manifest`, singular payload, and
adapter fields are optional compatibility fields and are never production
sources of truth.

## Accepted Boundary

Student pins RADJAX-Contract to the P1.5 acceptance receipt commit
`ff8f6e9af976fc599ee31173d4f177fb1250b4d7` and opens the canonical fixture
through `radjax_contract.testing.production_tome_fixture_path()`.

P1.6 does not claim implemented capabilities, corrected production run defaults,
compatibility, payload loading, model allocation, training, schedule execution,
checkpoint execution, JAX portability, Hugging Face export, or model quality.
