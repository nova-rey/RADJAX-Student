# P3.2 Student Architecture Plugin Contract

P3.2 defines the stable plugin boundary between generic learning and concrete
Student model math. The public contract is `radjax_student.architecture`.

## Ownership

An architecture plugin owns its configuration, parameter identities, parameter
metadata, named-region membership, objective-surface declarations, batch
compatibility checks, parameter initialization boundary, and forward-math
boundary.

It does not own optimizer policy, learning loops, runtime device selection, JIT
policy, checkpoint scheduling, Tome parsing, or behavior-pass policy. Runtime
continues to own execution; generic learning continues to own state changes.

## Stable Parameter Identity

`ParameterDescriptor` and `ParameterCatalog` establish stable dotted parameter
paths. Catalogs sort paths deterministically, reject duplicates, and carry no
arrays, backend values, or Python object IDs. Architecture plugins resolve
`UpdateScope` intent against these paths. This gives later scoped updates a
stable-tree and deterministic-mask basis without removing subtrees.

`NamedRegion` is architecture-owned. The generic learning core does not inspect
or infer region meaning. Overlap is allowed only as an explicit, deterministic
architecture declaration and is reported as a warning when relevant.

## Objective and Execution Boundaries

`IntermediateSurfaceDescriptor` makes optional objective surfaces discoverable
without requiring layers, hidden states, or teacher/student correspondence.
Every trainable plugin must resolve `final_output`; intermediate and
architecture-specific surfaces are optional capabilities.

`ArchitectureInitRequest` carries runtime-owned RNG identity by reference, not
a raw global RNG object. `ArchitectureInitResult`, `ForwardRequest`, and
`ForwardResult` may hold opaque runtime values in memory, while their serialized
forms expose only metadata and presence flags. P3.2 neither executes JAX nor
performs numerical model initialization or forward computation.

## Test Double

`FakeArchitecturePlugin` declares a three-path, non-numerical model socket:
`trunk.weight`, `trunk.bias`, and `head.weight`. It proves whole-student,
named-region, explicit-path, final-output, and intermediate-surface contracts.
It is not an RWKV, Mamba, transformer, or production Student architecture.

## Claim

RADJAX-Student has a stable architecture-plugin contract for configuration,
parameter identity, initialization, forward math, batch compatibility, targeted
update resolution, objective-surface resolution, and capability reporting. It
does not yet claim a concrete architecture, numerical forward execution,
gradients, optimizer updates, Tome loading, or training.
