# Runtime Backends

Runtime backends are deferred in this scaffold. Future work should add reference
and Pallas runtimes while keeping Pallas opt-in and default CI CPU-safe.

Runtime answers where and how execution occurs. It owns device policy,
precision policy, compilation policy, checkpoint execution mechanics, and
optional accelerator optimizations.

Runtime must not know which architecture is executing beyond the stable
architecture plugin interface. Fast paths must never become correctness paths.
