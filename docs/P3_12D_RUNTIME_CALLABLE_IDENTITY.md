# P3.12D Runtime Callable Identity

P3.12D closes the last P3.12 execution-authority gap: a runtime execution
request now carries a typed callable reference derived from the actual,
registered production callable rather than trusting an unrelated caller string.

The runtime owns `RuntimeCallableDeclaration`, source-derived
`RuntimeCallableIdentity`, `RuntimeCallableReference`, binding, final prepared
execution identity, and exact in-process cache-key identity.  The supported
surface is deliberately narrow: a declared top-level production function.  It
does not claim identity for arbitrary Python functions, closures, decorators,
partials, callable instances, compiled executables, or transitive dependency
semantics.

The canonical generic operation is
`radjax.learning.generic_jax_step`, version `1`.  Steps owns its extracted
kernel; runtime validates the declaration, derives the normalized AST source
digest, and returns the binding.  Learning requests that binding through the
P3.12C assembler.  A caller cannot pair the request reference with a foreign
callable.

Preparation intentionally remains incomplete until actual arguments are
available.  At compile/dispatch, runtime derives the one authoritative
`RuntimePreparedExecutionIdentity`.  It binds callable reference, backend,
runtime implementation version, mode, full compilation options, input contract,
static argument contract, actual static values, donation contract, placement,
and required capabilities.  Eager and JIT share the callable identity but have
distinct prepared identities.  Timings are telemetry and are excluded from all
identity evidence.

`PreparedExecutionIdentityCache` stores identity evidence only, not compiled
executables.  Reuse is permitted only for exact equality; a changed callable,
mode, static value, input contract, placement, backend, or runtime version
cannot reuse the key.  Persistent, cross-process, cross-machine, distributed,
multi-device, and TPU cache claims are not made.

The runtime-owned initialization reference is derived by
`initialization_reference_from_root_seed`; learning consumes that reference and
does not manufacture a raw initialization-key identity.

The executable receipt uses schema
`radjax.p3_12d_runtime_callable_identity_receipt.v1`.  It requires exactly 18
ordered positive proofs and 40 ordered adversarial cases.  Each adversary runs
twice from fresh inputs and requires exact expected/observed blocker equality
and callable-derived boundary equality.  The JAX-free source audit uses schema
`radjax.p3_12d_callable_identity_audit.v1` and rejects competing binders,
production-to-validation dependencies, caller identity material, unsafe
identity primitives, permissive matching, non-literal inventories, and unsafe
request surfaces.

## Validation migration

| Path | Prior callable identity | Classification | D disposition |
| --- | --- | --- | --- |
| `learning/assembly.py` | free `function_id` factory | accepted happy path | migrated to runtime binding/reference |
| `steps/jax_loop.py` | raw generic step dispatch | accepted happy path | receives exact runtime binding |
| `steps/jax_step.py` | raw callable beside request ID | accepted happy path | dispatches the runtime-bound top-level kernel |
| `runtime/portability.py` | free scale-add callable ID | runtime happy path | explicit runtime declaration/binding |
| P3.11.8/P3.11.9/P3.11.10 validation runners | lower-level historical contracts | focused lower-level tests | retained only where they intentionally test legacy owner boundaries |
| P3.12C runner | canonical product path | accepted happy path | executes through the assembler-owned binding |
| eager/JIT generic-step tests | direct runtime observations | focused product tests | require callable reference and final prepared digest |

P3.12D does not introduce RWKV, Hugging Face export, teacher inference,
distillation, model quality claims, a production CLI, Phase 4, or P3.12E.
P3.12 is closed only after the recorded receipt and all historical gates have
been regenerated and verified.
