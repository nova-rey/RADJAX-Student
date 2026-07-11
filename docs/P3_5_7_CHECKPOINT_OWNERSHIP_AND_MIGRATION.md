# P3.5.7 Checkpoint Ownership And Migration

The existing `learning_checkpoint.v2` format remains canonical. P3.5.7 adds
an additive checkpoint role and payload-descriptor metadata while preserving
loading of older v2 manifests that do not contain those optional fields.

The supported role is `radjax_continuation`. It owns logical architecture,
learning, optimizer, and batch-source state. Runtime handles, device objects,
executables, and raw JAX keys are not serialized. `hf_distribution` is a
separate future role; the continuation save/load APIs reject using one format
as the other, and `reject_implicit_hf_conversion()` is the explicit boundary
until a validated conversion implementation exists.

No v3 bump is justified by this additive metadata. A future tensor-pytree
payload codec may require a new version only if its serialized structure is
incompatible with v2.
