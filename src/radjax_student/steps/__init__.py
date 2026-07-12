"""Single-step learning execution; no loop, checkpoint, or Tome behavior."""

from radjax_student.steps.loop import (
    LearningLoopConfig,
    LearningLoopResult,
    LearningStepExecutor,
    SyntheticBatchSource,
    run_learning_loop,
)

__all__ = [
    "LearningLoopConfig",
    "LearningLoopResult",
    "LearningStepExecutor",
    "SyntheticBatchSource",
    "run_learning_loop",
    "JaxLearningLifecycle",
    "JaxLoopExecutor",
]


def __getattr__(name):
    if name in {"JaxLearningLifecycle", "JaxLoopExecutor"}:
        from radjax_student.steps.jax_loop import JaxLearningLifecycle, JaxLoopExecutor

        return {
            "JaxLearningLifecycle": JaxLearningLifecycle,
            "JaxLoopExecutor": JaxLoopExecutor,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
