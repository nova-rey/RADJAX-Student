# P5.0A Configurable RWKV-7 Instantiation

P5.0A adds one stateless, configurable instantiation path to the existing
RWKV-7 plugin identity/version. Only positive vocabulary size and maximum host
context capacity vary; all RWKV structural dimensions remain the Phase 4
float32 reference values.

Complete neutral Hugging Face tokenizer, vocabulary, and special-token
identities travel as validated `ArchitectureConfig` metadata. The plugin does
not own a language type, tokenizer hash shortcut, or configuration state.

The frozen 16x4 Phase 4 configuration, fixture descriptor, kernels, and exact
parity boundary are unchanged. Configurable host batches accept one sequence
of length `1..context_length`; JAX derives token range from materialized
embedding shape. This local synthetic 64x5 proof makes no Tome, behavioral
batch, objective, learning, checkpoint, or evaluation claim.
