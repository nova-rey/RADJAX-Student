# P3.5.5 Legacy And Debug Isolation

The production package roots expose contract and metadata APIs only. The
former dense Tome target loader remains directly importable from
`radjax_student.artifacts.targets` for compatibility, but is no longer
re-exported by `radjax_student.artifacts`.

The former NumPy tiny training smoke is available from
`radjax_student.legacy.training`. `radjax_student.training` is an empty
production namespace, and its old `distill` module is a warning-emitting
compatibility shim. The old CLI module remains independently executable but
is not registered in the default product command parser.

The canonical tiny backend is `radjax_student.debug`; the legacy namespace
does not define a competing implementation or registry.
