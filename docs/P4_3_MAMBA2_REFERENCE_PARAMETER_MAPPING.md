# Mamba-2 Reference Mapping (v2.2.4)

This is the reviewed mapping for the accepted `radjax.architecture.mamba2_reference`
profile. The authoritative oracle uses the unchanged upstream
`MambaConfig`/`MambaLMHeadModel` configuration from
`state-spaces/mamba@95d8aba8a8c75aedcaa6143713b11e745e7cd0d9`:

| field | frozen value | evidence |
| --- | ---: | --- |
| `d_model` | 8 | `full_token_step_fixture.json` and generator |
| `n_layer` | 2 | `full_token_step_fixture.json` and generator |
| `expand`, `d_ssm` | 2, 16 | upstream `ssm_cfg` |
| `nheads`, `headdim` | 4, 4 | upstream `ssm_cfg` |
| `d_state`, `ngroups` | 4, 1 | upstream `ssm_cfg` |
| `d_conv` | **4** | upstream `ssm_cfg`, fixture, carry/catalog shapes |
| `chunk_size` | 4 | upstream `ssm_cfg` |
| `dtype`, batch | `float32`, 1 | oracle settings |
| `tie_embeddings` | **true** | upstream `MambaConfig`, fixture, initializer |

The earlier planning value `d_conv=3` is superseded; it is not source-faithful
for this accepted oracle. The convolution state is `[1, 24, 4]` per layer and
the SSM state is `[1, 4, 4, 4]` per layer. Both are persistent carry families.

Logical parameter paths follow the upstream roles. The embedding and
`lm_head.weight` paths are distinct catalog identities but share the declared
`token_embedding` tied-weight group and are materialized as the same JAX array
at initialization. An initialized-checkpoint restore preserves equal values and
the tied catalog declaration. The accepted generic optimizer updates logical
parameter leaves independently after a learning step, so this checkpoint makes
no optimizer-level shared-update claim; adding such semantics would be generic
optimizer work outside the bounded reconciliation. This does not claim PyTorch
checkpoint loading, Transformers compatibility, or pretrained-weight
equivalence.

The complete full-LM fixture starts from deliberately asymmetric deterministic
nonzero convolution and SSM caches. `evidence/mamba2_oracle/witness.json`
provides the independent zero-cache case and asymmetric SSD-core cross-check.
For layer index `l` and flattened cache index `i`, the full-fixture generator
initializes convolution state as `0.001*i + 0.01*(l+1)` and SSM state as
`0.002*i + 0.02*(l+1)` before the first token; this exercises every persistent
leaf in both state families. The fixture and reproduction-generator digests,
source file hashes, capture
provenance, and regeneration command are recorded in
`evidence/mamba2_oracle/README.md`; no fixture regeneration is part of this
reconciliation.
