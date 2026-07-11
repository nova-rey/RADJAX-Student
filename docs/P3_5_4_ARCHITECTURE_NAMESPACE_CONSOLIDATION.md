# P3.5.4 Architecture Namespace Consolidation

P3.5.4 makes `radjax_student.architecture.ArchitectureRegistry` the only
public architecture registry. The package root has no transitional exports,
and the tiny NumPy implementation lives under the explicit
`radjax_student.debug` namespace.

The old `radjax_student.students` package remains importable only as a
deprecated compatibility path. It is not imported by the production root,
architecture, learning, runtime, or objective modules. The compatibility
implementation delegates its tiny backend import to the canonical debug
implementation and emits `DeprecationWarning`; the removal checkpoint is
P4.1 architecture implementation.

This checkpoint does not claim that all legacy training and dense target
exports are removed. Those are P3.5.5 blockers and remain visible in the
machine-readable audit until that checkpoint is complete.
