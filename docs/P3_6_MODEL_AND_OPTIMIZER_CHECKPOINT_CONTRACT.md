# P3.6 Model and Optimizer Checkpoint Contract

P3.6 adds `radjax_student.checkpoints`, a deterministic layered persistence
contract. Schema `learning_checkpoint.v2` contains a runtime reference, learning
progress, architecture state and parameters, optimizer state, checkpoint-owned
batch-source state, component manifest, and SHA-256 integrity data.

Ownership mirrors the system: runtime identifies execution, learning owns
progress, architecture owns parameters and auxiliary state, optimizer owns its
moments and tracked paths, and the batch source owns resumable source state.
The v2 component set is `architecture.json`, `learning.json`, `optimizer.json`,
and `source.json`; the latter is owned by `batch_source`. Every component is
listed in the manifest with a size and SHA-256 digest, and the ownership map is
covered by the manifest digest.

`source.json` always exists, even when `source_state` is `null`, keeping the
manifest shape stable. Source state is restricted to normalized finite JSON
values with string mapping keys. Open handles, callables, bytes, sets, and
non-finite floats are rejected. The loader validates the manifest, all hashes,
runtime reference, source component shape, architecture, and optimizer before a
caller exposes restored state. Callers restore source state into a candidate
source and only publish the restored destination after every validation passes.

This is a scalar-contract checkpoint proof. It does not claim distributed or
sharded checkpoints, Tome persistence, production checkpoint performance, or
generic source implementation compatibility beyond a source's own validated
`state_dict`/`load_state_dict` contract.
