# P3.11.3 Parameter Layout and Scope Routing

`ParameterTreeLayout` is the canonical bijection between stable logical paths
and mapping-only JAX pytree paths. It records shape, dtype, role, regions,
trainability, exportability, optional HF key, and tied-weight identity.

The production JAX execution plan performs these operations in order:

1. The architecture resolves `ObjectiveScope` against its metadata.
2. The architecture resolves `UpdateScope` against its parameter catalog.
3. The layout validates the catalog and materialized parameter tree.
4. The layout derives the boolean update mask from resolved logical paths.

`ForwardResult.surface_for()` accepts only an architecture-resolved selection or
a canonical surface ID. Its old scope interpretation is confined to the
explicit pre-P3.11 compatibility adapter and is not used by the new execution
plan.
