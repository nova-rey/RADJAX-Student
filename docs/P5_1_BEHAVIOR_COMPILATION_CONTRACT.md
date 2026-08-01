# P5.1 Behavior-Compilation Contract Freeze

P5.1 freezes the first native-v3 Tome-to-Student admission boundary. It is a
JAX-free contract checkpoint: it validates and describes a declared artifact;
it does not load training values, construct batches, allocate a model, choose
an objective, or execute learning.

## Authorities and Pins

- RADJAX-Student baseline: `650091e46d33935f2ec4cbc5909e1413214e4ba7`.
- RADJAX-Contract implementation pin: `b1209f21fef9405776a757f1a5749d3152bbc3c6`
  (`v0.6.0`). The local current publication receipt is
  `5dd71e9ab81dfa54ffb66c6c027a4a805e266c9c`.
- RADJAX-Tome producer: `354532569dea21e7846ab312994ce3318daa1c13`.
- Student-consumption contract: `radjax_tome_student_consumption_contract`,
  publication version `4.0.0`, exact profile `native_v3_student_v4`.

The historical `production_v2` validator and fixture remain only for explicit
historical v2 inputs. Native-v3 P5 admission never invokes them and never
upgrades v2 identity into v3/v4 identity.

## Contract-Owned Admission

Student discovers the installed Contract v4 tree through
`tome_student_consumption_v4_contract_root()`. Before trusting it,
`load_native_v3_contract_assets()` verifies every listed `SHA256SUMS` asset and
rejects a missing or injected asset. The checked tree includes the closed
profile, manifest, identity, descriptor, validation-result, and cover schemas;
error vocabulary; canonical JSON/JSONL and semantic-declaration recipes;
descriptor vector; valid vector; adversarial catalog; and profile declaration.

Student admits an artifact only through Contract's explicit v4 resolver. The
profile is negotiated exactly: no v2/v3 fallback, filename inference,
directory walking for payload discovery, Tome implementation import, or
producer-schema copy is allowed. Contract resolves the declared resources,
validates inventory bindings and byte integrity, validates joins and semantic
identity, and returns typed descriptor metadata. Student preserves that
descriptor as an immutable `NativeV3StudentConsumptionView`.

The native-v3 surface is `radjax_tome_cover_v3_student_consumption_v4` with
the `manifests/student_consumption_v4.json` declaration. It declares the
vocabulary/sequence contract, explicit target-shard, registry, corridor,
exemplar, validation, and provenance resources, their encodings/axes/joins,
and the distinction between batch semantics and integrity-bound provenance.
Delivery receipt and authority reference are required and validated, but are
not batch-semantic identity inputs. Delivery path therefore remains provenance,
not a target-semantics switch.

## Frozen Student Interfaces and Ownership

`NativeV3ContractAssets` contains only the installed Contract identity,
publication version, root, profile, and immutable asset digests.
`NativeV3StudentConsumptionView` binds an artifact path to those verified assets
and Contract's `StudentConsumptionV4Descriptor`. P5.1 does not expose resource
bytes. P5.2 may open only a descriptor-declared resource through Contract's
verified resource-opening API and must preserve the explicit resource ID, role,
instance, encoding, axes, locator, raw digest, and semantic digest.

- Contract owns profile/schema/error/recipe/vector meaning and validation.
- Tome owns native-v3 artifact production.
- Student owns fail-closed admission, immutable projection, and later safe
  extraction; it does not redefine payload semantics.
- Learning owns pass execution and checkpoint boundaries; objectives own loss
  semantics; architecture owns math; runtime owns execution/RNG.

The accepted plan remains data and must be executed only as `corridor` then a
checkpoint boundary then `exemplar` then a second checkpoint. Mode identifiers
and diagnostic fingerprint identifiers remain distinct. Carry that crosses a
learning-step or checkpoint boundary is stop-gradient; P5 makes no cross-step
BPTT claim.

## Authority Evidence

The installed v4 Contract asset tree checksum verification succeeded. Its
conformance catalog defines the exact profile/no-fallback, required-evidence,
and deterministic-adversarial cases. A current Tome package was generated in a
temporary local verification directory and admitted by the Student P5.1 API
with `strict=True`:

- base native-v3 semantic identity:
  `sha256:17456ebc3fbb013c8154e65b977337ec871be21aa4da68aba11712d054642363`;
- v4 consumption semantic identity:
  `sha256:38161318b527ea799390b355eb335e068755085ee9c7d154d51a327e0c671a7d`;
- resolved resources: four corridor, two exemplar, and four validation.

This proves local Student execution of the Contract-defined native-v3 v4
admission path against current Tome output. It is not a committed production
fixture, model-quality result, training receipt, or package claim.

## Error Taxonomy and Non-Claims

Admission retains Contract's deterministic `TSC*` issue codes. It fails closed
on an unavailable/unsafe transport, unsupported cover/profile, invalid
inventory binding, invalid resource encoding, structural join failure,
semantic digest mismatch, or invalid evidence sidecar. Safe noncanonical
transport remains a Contract warning unless the caller selects `strict=True`.

P5.1 does not claim payload materialization, batch collation, tokenization,
corridor or exemplar loss execution, parameter movement, optimizer execution,
checkpointing, replay, evaluation, HF packaging, model quality, pretrained or
weight-file compatibility, Transformers loadability, accelerator scale, or
remote CI success.
