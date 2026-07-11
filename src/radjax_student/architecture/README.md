# Architecture

Architecture plugins answer how a model computes. P3.2 establishes the stable
contract here with a non-numerical test double; it does not implement an RWKV,
Mamba, transformer, or other concrete model.

This package is the long-term home for architecture plugin contracts and
implementations such as RWKV, QRWKV, Mamba, Transformers, and future student
families.

Plugins own parameter identity, model math, named-region membership, and
objective-surface declarations. They do not own runtime policy, device
placement, sharding, Pallas kernels, optimizer policy, checkpoint scheduling,
Tome parsing, or training-loop policy.
