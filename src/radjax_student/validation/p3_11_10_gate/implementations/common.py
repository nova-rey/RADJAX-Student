"""JAX-free evidence primitives for literal P3.11.10C experiments.

The helpers in this module deliberately know nothing about the inventory.  They
only record concrete before/after inputs and wrap a callable selected by an
experiment function.  Every inventory entry is represented by one named
function in a section module.
"""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any

from radjax_student.validation.p3_11_9_replay.canonical import (
    ReplayCanonicalError,
    canonical_digest,
)
from radjax_student.validation.p3_11_10_gate.models import ObservedFailure


def callable_identity(value: Callable[..., Any]) -> str:
    """Return a stable identity for a concrete callable."""

    try:
        source = inspect.getsource(value)
    except (OSError, TypeError):
        source = None
    return canonical_digest(
        {
            "module": getattr(value, "__module__", type(value).__module__),
            "qualname": getattr(value, "__qualname__", type(value).__qualname__),
            "source": (
                hashlib.sha256(source.encode("utf-8")).hexdigest()
                if source is not None
                else None
            ),
        }
    )


def _memory_projection(value: Any) -> Any:
    """Project an actual in-memory input into a finite canonical identity.

    This is deliberately an input projection, rather than experiment metadata:
    mappings, values, array shape/dtype, and canonical bytes all contribute to
    the resulting digest.  It lets a literal experiment prove that an object or
    structured-dtype input changed before the public NPZ boundary rejects it.
    """

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, bytes):
        return {"bytes": hashlib.sha256(value).hexdigest(), "size": len(value)}
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return {"nonfinite_float": repr(value)}
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _memory_projection(item) for key, item in sorted(value.items())
        }
    if isinstance(value, (tuple, list)):
        return [_memory_projection(item) for item in value]
    if is_dataclass(value):
        return {
            "dataclass": type(value).__qualname__,
            "fields": {
                item.name: _memory_projection(getattr(value, item.name))
                for item in fields(value)
                if item.repr or item.name not in {"_handle", "_backend", "_request"}
            },
        }
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return {
                "typed": type(value).__qualname__,
                "value": _memory_projection(to_dict()),
            }
        except (TypeError, ValueError):
            pass
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - package boundary
        np = None
    if np is not None and (
        isinstance(value, np.ndarray)
        or (
            hasattr(value, "shape")
            and hasattr(value, "dtype")
            and hasattr(value, "__array__")
        )
    ):
        # JAX values cross this JAX-free boundary through NumPy's array
        # protocol, so the identity describes the concrete input values.
        array = np.asarray(value)
        descriptor: dict[str, Any] = {
            "array": True,
            "shape": list(array.shape),
            "dtype": array.dtype.str,
            "memory_order": (
                "F"
                if array.ndim > 1
                and array.flags.f_contiguous
                and not array.flags.c_contiguous
                else "C"
            ),
        }
        if array.dtype.hasobject:
            descriptor["object_repr"] = repr(array.tolist())
        elif array.dtype.fields is not None:
            descriptor["structured_dtype"] = repr(array.dtype.descr)
            descriptor["bytes"] = hashlib.sha256(array.tobytes()).hexdigest()
        else:
            if array.dtype.byteorder == ">" or (
                array.dtype.byteorder == "=" and not np.little_endian
            ):
                array = array.astype(array.dtype.newbyteorder("<"), copy=False)
            array = np.ascontiguousarray(array) if array.ndim else array
            descriptor["bytes"] = hashlib.sha256(array.tobytes(order="C")).hexdigest()
        return descriptor
    if hasattr(value, "__dict__") and isinstance(value.__dict__, dict):
        return {
            "object": type(value).__qualname__,
            "attributes": {
                str(key): _memory_projection(item)
                for key, item in sorted(value.__dict__.items())
                if not str(key).startswith("_")
            },
        }
    return {"type": type(value).__qualname__}


def memory_digest(value: Any) -> str:
    """Digest a concrete finite-JSON or array-bearing public input."""

    return canonical_digest(_memory_projection(value))


def public_boundary(
    boundary: str,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Attach a boundary identity to a real public-call wrapper.

    Case experiments choose a concrete wrapper.  The probe, not the case,
    obtains the boundary name from that wrapper when invocation begins.
    """

    def decorate(value: Callable[..., Any]) -> Callable[..., Any]:
        value.__p31110_public_boundary__ = boundary
        return value

    return decorate


def directory_digest(directory: Path) -> str:
    """Digest the real directory tree, including each relative file and bytes."""

    if not directory.is_dir():
        return canonical_digest({"exists": False})
    entries: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory).as_posix()
        if path.is_dir():
            entries[f"{relative}/"] = "directory"
        elif path.is_file():
            entries[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        else:
            entries[relative] = "other"
    return canonical_digest({"exists": True, "entries": entries})


@dataclass(frozen=True)
class MutationDelta:
    public_input_kind: str
    canonical_path: str
    operation: str
    baseline_input_digest: str
    mutated_input_digest: str
    before_digest: str
    after_digest: str
    value_summary: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value
            for value in (self.public_input_kind, self.canonical_path, self.operation)
        ):
            raise ValueError("mutation identity fields must be nonempty")
        if self.baseline_input_digest == self.mutated_input_digest:
            raise ValueError("the concrete public input did not change")

    @property
    def digest(self) -> str:
        return canonical_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "public_input_kind": self.public_input_kind,
            "canonical_path": self.canonical_path,
            "operation": self.operation,
            "baseline_input_digest": self.baseline_input_digest,
            "mutated_input_digest": self.mutated_input_digest,
            "before_digest": self.before_digest,
            "after_digest": self.after_digest,
            "value_summary": dict(self.value_summary),
        }


@dataclass
class BoundaryProbe:
    """Instrumentation around the actual selected public callable."""

    boundary: str
    public_callable: Callable[..., Any]
    public_input_digest: str
    events: list[str] = field(default_factory=list)
    observed_exception: BaseException | None = None
    returned_value: Any = None
    post_boundary_reached: bool = False

    @property
    def callable_identity(self) -> str:
        return callable_identity(self.public_callable)

    @property
    def trace_digest(self) -> str:
        return canonical_digest(
            {
                "boundary": self.boundary,
                "callable": self.callable_identity,
                "input": self.public_input_digest,
                "events": self.events,
                "exception": None
                if self.observed_exception is None
                else type(self.observed_exception).__name__,
                "post_boundary_reached": self.post_boundary_reached,
            }
        )

    def call_catching(self, value: Any) -> BoundaryProbe:
        actual_boundary = getattr(
            self.public_callable, "__p31110_public_boundary__", None
        )
        if actual_boundary != self.boundary:
            self.events.append("boundary_identity_missing_or_mismatched")
            self.observed_exception = RuntimeError("public boundary identity mismatch")
            return self
        self.events.extend(("invocation_started", "intended_boundary_entered"))
        try:
            self.returned_value = self.public_callable(value)
        except BaseException as error:
            self.observed_exception = error
            self.events.extend(("failure_observed", "execution_stopped"))
            return self
        self.events.append("boundary_exited")
        self.post_boundary_reached = True
        self.events.append("post_boundary_reached")
        return self


@dataclass(frozen=True)
class ExperimentExecution:
    baseline_input: Any
    mutated_input: Any
    mutation_delta: MutationDelta
    probe: BoundaryProbe


@dataclass(frozen=True)
class GateExecutionContext:
    repository_root: Path
    temporary_root: Path


ExperimentFunction = Callable[[GateExecutionContext], ExperimentExecution]


@dataclass(frozen=True)
class GateCaseImplementation:
    """One literal experiment function and no per-case parameterization."""

    function: ExperimentFunction
    case_id: str = ""
    expected_boundary: str = ""
    execution_class: str = ""
    preparation_identity: str = ""
    mutation_identity: str = ""
    invocation_identity: str = ""
    observation_adapter_identity: str = ""

    def bind(
        self,
        *,
        case_id: str,
        expected_boundary: str,
        execution_class: str,
    ) -> GateCaseImplementation:
        """Attach registry-owned identity without exposing it to the function.

        The literal experiment receives only ``GateExecutionContext``.  The
        registry is the sole inventory-aware layer, so case metadata cannot be
        used to manufacture a mutation or an observed failure.
        """

        function_identity = callable_identity(self.function)
        return replace(
            self,
            case_id=case_id,
            expected_boundary=expected_boundary,
            execution_class=execution_class,
            preparation_identity=canonical_digest(
                {"function": function_identity, "stage": "preparation"}
            ),
            mutation_identity=canonical_digest(
                {"function": function_identity, "stage": "mutation"}
            ),
            invocation_identity=canonical_digest(
                {"function": function_identity, "stage": "invocation"}
            ),
            observation_adapter_identity=callable_identity(observe_failure),
        )

    @property
    def behavior_identity(self) -> str:
        """Identity of actual experiment behavior, deliberately excluding ID.

        Registry validation uses this alongside the literal function object so
        two entries cannot be relabeled copies of the same experiment.
        """

        return canonical_digest(
            {
                "function": callable_identity(self.function),
                "preparation": self.preparation_identity,
                "mutation": self.mutation_identity,
                "invocation": self.invocation_identity,
                "observer": self.observation_adapter_identity,
                "boundary": self.expected_boundary,
                "execution_class": self.execution_class,
            }
        )

    @property
    def identity(self) -> str:
        """Full stable registered implementation identity."""

        return canonical_digest(
            {
                "case_id": self.case_id,
                "behavior": self.behavior_identity,
                "preparation": self.preparation_identity,
                "mutation": self.mutation_identity,
                "invocation": self.invocation_identity,
                "observer": self.observation_adapter_identity,
                "expected_boundary": self.expected_boundary,
                "execution_class": self.execution_class,
            }
        )


def execute_memory_experiment(
    context: GateExecutionContext,
    *,
    baseline: Any,
    mutated: Any,
    public_input_kind: str,
    canonical_path: str,
    operation: str,
    value_summary: Mapping[str, Any],
    public_callable: Callable[[Any], Any],
    baseline_callable: Callable[[Any], Any] | None = None,
) -> ExperimentExecution:
    """Validate a real baseline, digest it, then invoke the mutated input."""

    if baseline_callable is not None:
        baseline_callable(baseline)
    before = memory_digest(baseline)
    after = memory_digest(mutated)
    delta = MutationDelta(
        public_input_kind=public_input_kind,
        canonical_path=canonical_path,
        operation=operation,
        baseline_input_digest=before,
        mutated_input_digest=after,
        before_digest=before,
        after_digest=after,
        value_summary=value_summary,
    )
    boundary = getattr(public_callable, "__p31110_public_boundary__", "")
    probe = BoundaryProbe(boundary, public_callable, after)
    probe.events.extend(("preparation_completed", "mutation_applied"))
    probe.call_catching(mutated)
    return ExperimentExecution(baseline, mutated, delta, probe)


def execute_directory_experiment(
    context: GateExecutionContext,
    *,
    baseline_directory: Path,
    mutated_directory: Path,
    public_input_kind: str,
    canonical_path: str,
    operation: str,
    value_summary: Mapping[str, Any],
    public_callable: Callable[[Path], Any],
    baseline_callable: Callable[[Path], Any] | None = None,
    baseline_public_input: Any | None = None,
    mutated_public_input: Any | None = None,
) -> ExperimentExecution:
    """Record real filesystem state around one public filesystem boundary."""

    if baseline_callable is not None:
        baseline_callable(baseline_directory)
    # A filesystem public boundary can also have caller-owned request identity
    # (expected lifecycle fields, runtime reference, and so on).  Digest that
    # real request object when supplied rather than pretending the directory
    # alone was the entire public input.
    before = (
        directory_digest(baseline_directory)
        if baseline_public_input is None
        else memory_digest(baseline_public_input)
    )
    after = (
        directory_digest(mutated_directory)
        if mutated_public_input is None
        else memory_digest(mutated_public_input)
    )
    delta = MutationDelta(
        public_input_kind=public_input_kind,
        canonical_path=canonical_path,
        operation=operation,
        baseline_input_digest=before,
        mutated_input_digest=after,
        before_digest=before,
        after_digest=after,
        value_summary=value_summary,
    )
    boundary = getattr(public_callable, "__p31110_public_boundary__", "")
    probe = BoundaryProbe(boundary, public_callable, after)
    probe.events.extend(("preparation_completed", "mutation_applied"))
    probe.call_catching(mutated_directory)
    return ExperimentExecution(baseline_directory, mutated_directory, delta, probe)


@dataclass(frozen=True)
class FailureAdapter:
    exception_type: type[BaseException]
    boundary: str
    code: str
    message_digests: tuple[str, ...]

    def matches(self, error: BaseException, boundary: str) -> bool:
        return (
            isinstance(error, self.exception_type)
            and boundary == self.boundary
            and hashlib.sha256(str(error).encode()).hexdigest() in self.message_digests
        )


OBSERVED_FAILURE_ADAPTERS: tuple[FailureAdapter, ...] = (
    FailureAdapter(
        ReplayCanonicalError,
        "replay_schema_validation",
        "replay_canonical_error",
        (
            "32e5b3f62097abb5d9732a3bc2b89869f3eb035fa00927ee62259338df1e6b92",
            "c9b4b5a971877d3ae73ed58253a226d25d6f4b0c82c479207df22f65d71e11cb",
            "4f86167d06b797cbb85ce5269e7ab647a26990ea7fffa8470b6a26214c06306f",
            "d32d3091d7a1fe94642bfdc8062993d18c21c013172ffa5eb00d18ff1dafb79c",
            "153ecd8cbd0dd58a7a377299ec3b75bfdea1a4959f2ece5215a4c62936121312",
            "711e328219c77d6f31f187a9b46485d6eb5537a74f577899504f33f809676820",
            "8b2438dfffafd507b69340cc172770f8fe7291a48e86043f815dd2756dbfe0d0",
            "6fce9556b65d24f2ec2e983894227422bd06c91a5dd471971bf72b04100f67c7",
            "2853ceacbdf9f915a05505b12c3f620d7683d5a629010979f7ed5440617e323f",
            "cca95c1898248dab5d013685a79a48a49d4f053b8e2d3064e4d868cb7f504f90",
            "7b7e830769c3a8bb448224b320272e29b6cdb7b877d9406c482923c7445241d3",
            "0670b4c8cb05904a7ac8178fbd68301135731234870a5db89aa27a20ef2803f5",
            "c062d53de862ad4f1e1ee2bc0c7cc95fef6372b41265072f8f76ac7bb7a52ae6",
            "dbd03dd24ca5e47ad6ee6bc5e3b350d2867c0264a48bf0cca59ef3c6bea42f54",
            "6c2abedc18dd9fcbbd20298ffbc78cba63ed15594a0daaaf6c8ad9a9d5b30d7c",
            "74c1e0918bf27786be63842ef28c1c4e807ab977a13b70750df0240283ccf224",
        ),
    ),
    FailureAdapter(
        ReplayCanonicalError,
        "resume_replay_validation",
        "replay_canonical_error",
        (
            "e0955e0fb40b2243f7ce01c4090606dbeea671c631c9d9a4a9b5feb8f1910395",
            "a18e954cdb77dc3b206fb1b3b8c94f1801a2a7eabe98a3de312a874e5e7097b5",
            "c9b4b5a971877d3ae73ed58253a226d25d6f4b0c82c479207df22f65d71e11cb",
            "535f525d687bd9f3e592b6a39de1f8b5d538042bbc6c40d7a49c3be07bc891ae",
            "c062d53de862ad4f1e1ee2bc0c7cc95fef6372b41265072f8f76ac7bb7a52ae6",
            "a306ebbee8cc91a68f6cea3e62d378ad6380bb21f1143f332dfbcd9fee7e2ff3",
            "dbd03dd24ca5e47ad6ee6bc5e3b350d2867c0264a48bf0cca59ef3c6bea42f54",
            "ea49c379c7acc7fa787e6a14108658956a4219382514798a9be7224ec73683bf",
        ),
    ),
    FailureAdapter(
        AttributeError,
        "loop_executor_validation",
        "AttributeError",
        ("429d393ebe92f8f3bf42194d215ff55a2e774444752eaebc18855f34d6e1d966",),
    ),
    FailureAdapter(
        KeyError,
        "learning_batch_validation",
        "KeyError",
        (
            "fe53f3d01262360a537861881a6afbd6e45e28e59241459424d3f82df886cfd3",
            "aa82f808f686cb827460142d5ed7e17a9ddd6e9ae95edd4cd3f4c4b04bf2805a",
        ),
    ),
    FailureAdapter(
        TypeError,
        "checkpoint_restore_validation",
        "TypeError",
        (
            "9db94df995bf44e2c962363069697a8c12d2099de7951a93163d13c013810d80",
            "1417f225bc5cc3bd1e3e202de9db09f6f8bab5f90a7e70981ca747336c6d6d26",
        ),
    ),
    FailureAdapter(
        TypeError,
        "learning_batch_validation",
        "TypeError",
        ("5020861e9461293626c22586d757fee385016821236e5094a4463ccfd5aae72d",),
    ),
    FailureAdapter(
        TypeError,
        "loop_executor_validation",
        "TypeError",
        ("5ce55e6cb9bb054f7691c4957013bc304c57c72219e36833b51523c9b387c9d2",),
    ),
    FailureAdapter(
        TypeError,
        "parameter_layout_validation",
        "TypeError",
        ("19b90033878d1e08531aa375bac51a1842d7b71ae7f1239343da18ba5f0b1014",),
    ),
    FailureAdapter(
        TypeError,
        "registry_validation",
        "TypeError",
        (
            "7d58a8627d25045b47abf282b85b2e2fff8f342c6a38d027416b9cc22f5a19b4",
            "d3b934ce3b01c5efecb654902be87a3deeec3a918204053ed372e710810ab9b1",
        ),
    ),
    FailureAdapter(
        ValueError,
        "checkpoint_restore_validation",
        "ValueError",
        (
            "3b935c68665c4881465900815293905d1835143ea429e02a425cc74119ec9e06",
            "9e9c99169caaf2ed0d28b744bcb2772f76af813c88792fac9fd3ab14f66a7dd1",
            "b7aa0765b00799d06652515a16362b47a87bcf0754c40c476fcd051b8cd1e988",
        ),
    ),
    FailureAdapter(
        ValueError,
        "learning_batch_validation",
        "ValueError",
        (
            "4cf62b3f2bb25f05da55b4840c062d605b3e5a34bd0eaa6a359bd362ddac5dd3",
            "393998f43b4436295017ad925822ba3b9fdea82677ba83f4d5c8acc46209680e",
            "7db29921daab1592fa49b6bdcba699833598b453a1589c2627df8b3033c13384",
            "7d1af08c5ecfeaa010c092c0fe2dec55c7c5b66791c3c7e799027a3c245b5b21",
            "842db8b2d06b5ad5dfaa9ad6ac0f406686f06b552b39fce674deaee57d1ed439",
        ),
    ),
    FailureAdapter(
        ValueError,
        "loop_executor_validation",
        "ValueError",
        (
            "1c9c4e07c6ddaacd81b36a4dc19edb29e2e2958ce28acfe5488e5bb16ad40e5c",
            "67b1a02fcd48c7a1f505c159352a47076d5a26fa04574292b4fadfa85260b267",
            "7119a6309f2ef7f33554b9be1733c0b2e24159b448948d69dd23ed013078690b",
            "43ad1aba004ac574a53018bcbdd93f76cc56913bfc7a7ea1bae29df67d7b16da",
            "e9943ad826cb6a4ae33a7ebc797538606529b4a7f0275c90062d21a1e651b867",
            "63a2d406e5acfb46d3a7f8a6fe12431c1f1143e1198035302d2b9225ef42bf1e",
            "842db8b2d06b5ad5dfaa9ad6ac0f406686f06b552b39fce674deaee57d1ed439",
        ),
    ),
    FailureAdapter(
        ValueError,
        "optimizer_registry_validation",
        "ValueError",
        (
            "8486cc8391e0350ccb2447707aa3dda9211fbe22b1eb14d499114679d9e30f71",
            "14f58a3935eb5aaeae20013c22986a0b78e081a29732e8568850896ff5511a11",
            "c6dc17a534cdee34cc49bb412239df371b80282c1cf7c56cec88136e75128a83",
            "1a0b984c61fa67789ff0713a61d82e67dedffe49d970e42c052423383e00afa6",
            "7efa240e8f8aa93183df6a509b1e6bbdd0fea3f741585f65518eaf9767e60e19",
            "3e943230c4394aadb7c2f39029e2571f93263bd14b652aeeae80bcd8dc5572d2",
        ),
    ),
    FailureAdapter(
        ValueError,
        "parameter_layout_validation",
        "ValueError",
        (
            "842429db2c11fe7958fabf4d2603667516e67034bb6fc0629883d440040016ee",
            "dc1af5c0ea9380705cb5327d2fe59e704c4ea714959078b432de6f459551a0bc",
            "0fd4a291230888d5fe56478cf2c5fd8cdedeb412e088132cbd66e50efca69de5",
            "3242f44edf68d1443aac019ff817ea3a87b8a48ca7bdccad019f244541b0a38a",
            "7efa240e8f8aa93183df6a509b1e6bbdd0fea3f741585f65518eaf9767e60e19",
        ),
    ),
    FailureAdapter(
        ValueError,
        "registry_validation",
        "ValueError",
        (
            "f61255bcc6440fbfbc56b735fcbc638130d2c89242aa0b4c95d424fe0a4ce894",
            "0d3094d6fb0b6d0a2f899028d60b16badc11dc7990e49afc2e931eb0d6b4c76d",
        ),
    ),
    FailureAdapter(
        ValueError,
        "replay_schema_validation",
        "ValueError",
        ("0e288bfb4957109d68aae7df49604b8cfadfc9c24dcd44b963f041cccfa8c300",),
    ),
    FailureAdapter(
        ValueError,
        "runtime_rng_validation",
        "ValueError",
        ("639d3479fd1cee182dbb9b6eeab977d5bfa1c337acd43df94dfeab413717d361",),
    ),
)


def observe_failure(probe: BoundaryProbe) -> ObservedFailure | None:
    """Normalize only the actual exception captured by a boundary probe."""

    error = probe.observed_exception
    if error is None:
        return None
    code = getattr(error, "code", None)
    if code is None:
        adapter = next(
            (
                value
                for value in OBSERVED_FAILURE_ADAPTERS
                if value.matches(error, probe.boundary)
            ),
            None,
        )
        code = adapter.code if adapter is not None else "unrecognized_public_failure"
    return ObservedFailure(
        code=str(code),
        boundary=probe.boundary,
        exception_type=type(error).__name__,
        phase="public_boundary",
        message_digest=hashlib.sha256(str(error).encode()).hexdigest(),
        details={
            "public_callable": probe.callable_identity,
            "exception_type": type(error).__name__,
        },
    )


__all__ = [
    "BoundaryProbe",
    "ExperimentExecution",
    "ExperimentFunction",
    "FailureAdapter",
    "GateCaseImplementation",
    "GateExecutionContext",
    "MutationDelta",
    "callable_identity",
    "directory_digest",
    "execute_directory_experiment",
    "execute_memory_experiment",
    "memory_digest",
    "observe_failure",
    "public_boundary",
]
