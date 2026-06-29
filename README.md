# RADJAX-Student

RADJAX-Student consumes RADJAX artifacts and trains/evaluates modular recurrent
student models using JAX/XLA-first infrastructure.

It does not load teacher models or run teacher inference. Teacher artifacts must
be produced externally by RADJAX-Tome.

The initial scaffold uses NumPy for a tiny debug student smoke so default CI does
not require JAX, TPU, Pallas, torch, or transformers.

