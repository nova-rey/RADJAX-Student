# P3.5.6 Hugging Face Preservation Contract

`radjax_student.hf` now contains dependency-free typed descriptors for future
Hugging Face packaging. The descriptor preserves logical architecture
identity, tokenizer identity, special-token IDs, parameter shape/dtype and the
three distinct parameter names: logical RADJAX path, JAX pytree path, and HF
distribution key. It also preserves tied-weight declarations, architecture
state metadata, compatibility metadata, and unknown fields.

Architecture configuration is authoritative. `from_architecture()` creates a
validated projection and `validate_against()` rejects conflicting descriptors.
Mapping names are checked for runtime/device/sharding/fused-layout tokens, so
logical HF identity cannot encode an execution layout. This checkpoint adds no
Transformers or safetensors dependency and does not implement export.
