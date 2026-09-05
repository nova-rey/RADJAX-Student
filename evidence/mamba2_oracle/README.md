# Mamba-2 v2.2.4 CUDA oracle witness

This evidence is generated independently of the Mamba-2 Student implementation
and is an upstream reference witness only.

## Authority and environment

- Repository: `https://github.com/state-spaces/mamba`
- Commit: `95d8aba8a8c75aedcaa6143713b11e745e7cd0d9` (`v2.2.4`)
- Container image: `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel`
- Resolved image digest: `sha256:e96c6896ecfbb50d89c87bf94110206ef444f27268c5f72201eb29fba9c90331`
- GPU: NVIDIA GeForce RTX 3060, compute capability 8.6, 12 GiB
- NVIDIA driver: `610.43.02`
- Python: 3.11 (official pinned PyTorch image)
- Torch: `2.4.0+cu124`; Triton: `3.0.0`
- `mamba-ssm`: `2.2.4`; `causal-conv1d`: `1.4.0`
- Other pinned packages: `transformers==4.44.2`, `huggingface-hub==0.24.7`,
  `einops==0.8.0`, `numpy==1.26.4`, `packaging==24.1`,
  `ninja==1.11.1.1`

The installed Python sources were verified against the checked-out commit.
The per-file hashes are recorded in `witness.json`; the required Mamba-2,
block, selective-state-update, softplus, SSD-combined, and normalization files
all matched.

## Command and reproducibility

The isolated guest ran:

```text
CUBLAS_WORKSPACE_CONFIG=:4096:8 python /workspace/m2oracle/oracle.py > /workspace/m2oracle/witness-zero.json
```

The generator digest was
`0b97510ebe7b34109edfcbbf87f4ac69c5b36e18a32e38d07bdac35dbc6e437e`.
Two fresh processes produced the same witness digest:

```text
b5105ee1d88eeaa3defbaf1821ad73818d36f18970b37ff0d682a610d7a80745
```

The witness records TF32 settings, matmul precision, deterministic-operation
settings, evaluation/inference mode, configuration, selected nonuniform
parameters, tokens, per-token logits, initial/final convolution and SSM
states, and the asymmetric SSD fixture/reference comparison.

## Frozen profile and claim boundary

The fixture profile is two layers, model width 8, four heads, head width 4,
SSM state width 4, convolution width 4, `float32`, batch size 1, vocabulary
16, and chunk capacity 4. `dt_limit` is passed to the reference as its required
unbounded value but represented in JSON as `{\"min\":0.0,\"max\":\"UNBOUNDED\"}`.

The witness exercises the unchanged official full-language-model token-step
path with preallocated official caches, for both zero and deliberately
asymmetric nonzero initial states. It establishes a bounded recurrent-equation
reference for this fixture only. It does not claim optimized-kernel parity,
pretrained-weight compatibility, PyTorch-checkpoint compatibility, training
recipe parity, Transformers compatibility, production performance, or whole
sequence equivalence. The sequence path remains an M2.4 Student test concern.

## Complete token-step fixture

`full_token_step_fixture.json` is the complete V16/T4 token-step witness used
by the M2.4 parity test. It was generated in the same isolated image and pinned
package set on a cheapest practical RTX A4000 (compute capability 8.6); the
temporary Vast instance was destroyed immediately after capture. The generator
used seed `20260905`, deliberately nonuniform convolution/SSM cache leaves, and
recorded every model parameter, four token logits, and both final state families.
Its canonical fixture digest is
`7708ae2eacbbaa45d9060604a43528ff1bb2ebb5835b78de5bfa92a31b348806` and its
checked-in file SHA-256 is
`2389f47c71c60a3fcdd9122cd5efe89637895ea30bc8543b9409541e49b91682`.
