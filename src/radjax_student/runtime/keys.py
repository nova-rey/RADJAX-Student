"""Backend-neutral, immutable RNG identity for the runtime contract."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

RUNTIME_KEYS_VERSION = "runtime_keys.v1"
RUNTIME_KEY_STREAM_NAMES: tuple[str, ...] = (
    "model_initialization",
    "data_order",
    "dropout",
    "augmentation",
    "evaluation",
    "runtime_tests",
)
JAX_KEY_BRIDGE_VERSION = "runtime_jax_key_bridge.v1"
JAX_KEY_SLOTS: tuple[str, ...] = (
    "initialization",
    "dropout",
    "augmentation",
    "architecture_stochastic_state",
    "optimizer_stochastic_state",
    "evaluation",
    "runtime_tests",
)


@dataclass(frozen=True)
class RuntimeInitializationKeyReference:
    """Serializable runtime-owned initialization-key identity, never a raw key."""

    schema_version: str
    stream: RuntimeKeyStream
    slot: str = "initialization"

    def __post_init__(self) -> None:
        if self.schema_version != "runtime_initialization_key_reference.v1":
            raise ValueError("unsupported initialization key reference schema")
        if not isinstance(self.stream, RuntimeKeyStream):
            raise TypeError("stream must be RuntimeKeyStream")
        if self.slot != "initialization":
            raise ValueError(
                "initialization key reference must use initialization slot"
            )

    @property
    def identity(self) -> str:
        return f"runtime_keys.v1:initialization:{self.stream.root_seed}"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "stream": self.stream.to_dict(),
            "slot": self.slot,
            "identity": self.identity,
        }


@dataclass(frozen=True)
class RuntimeKeyStream:
    """A deterministic named seed derivation, not a backend key object."""

    name: str
    root_seed: int
    lineage: tuple[str, ...]
    derived_seed: int
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if self.name not in RUNTIME_KEY_STREAM_NAMES:
            raise ValueError(f"unknown runtime key stream: {self.name}")
        _require_seed(self.root_seed, "root_seed")
        lineage = tuple(self.lineage)
        if lineage != ("root_seed", self.name):
            raise ValueError("runtime key lineage must be root_seed then stream name")
        _require_seed(self.derived_seed, "derived_seed")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("runtime key metadata must be a mapping")
        object.__setattr__(self, "lineage", lineage)
        object.__setattr__(self, "metadata", _freeze_json_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "root_seed": self.root_seed,
            "lineage": list(self.lineage),
            "derived_seed": self.derived_seed,
            "metadata": _json_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RuntimeKeyStream:
        return cls(
            name=_string(payload["name"], "name"),
            root_seed=_seed(payload["root_seed"], "root_seed"),
            lineage=_strings(payload.get("lineage", ()), "lineage"),
            derived_seed=_seed(payload["derived_seed"], "derived_seed"),
            metadata=_mapping(payload.get("metadata", {}), "metadata"),
        )


@dataclass(frozen=True)
class RuntimeKeys:
    """One root seed and a fixed, isolated hierarchy of named stream identities."""

    root_seed: int
    version: str
    streams: tuple[RuntimeKeyStream, ...]

    def __post_init__(self) -> None:
        _require_seed(self.root_seed, "root_seed")
        if self.version != RUNTIME_KEYS_VERSION:
            raise ValueError(f"unsupported runtime keys version: {self.version}")
        streams = tuple(self.streams)
        if any(not isinstance(item, RuntimeKeyStream) for item in streams):
            raise TypeError("streams must contain RuntimeKeyStream values")
        if tuple(item.name for item in streams) != RUNTIME_KEY_STREAM_NAMES:
            raise ValueError("runtime key streams must use the public contract order")
        if any(item.root_seed != self.root_seed for item in streams):
            raise ValueError("runtime key stream root seeds must match the tree root")
        expected = tuple(
            _stream(self.root_seed, name) for name in RUNTIME_KEY_STREAM_NAMES
        )
        if streams != expected:
            raise ValueError("runtime key streams must match deterministic derivation")
        object.__setattr__(self, "streams", streams)

    @classmethod
    def from_seed(cls, root_seed: int) -> RuntimeKeys:
        _require_seed(root_seed, "root_seed")
        return cls(
            root_seed=root_seed,
            version=RUNTIME_KEYS_VERSION,
            streams=tuple(
                _stream(root_seed, name) for name in RUNTIME_KEY_STREAM_NAMES
            ),
        )

    @property
    def model_initialization(self) -> RuntimeKeyStream:
        return self.stream("model_initialization")

    @property
    def initialization_reference(self) -> RuntimeInitializationKeyReference:
        return RuntimeInitializationKeyReference(
            "runtime_initialization_key_reference.v1", self.model_initialization
        )

    @property
    def data_order(self) -> RuntimeKeyStream:
        return self.stream("data_order")

    @property
    def dropout(self) -> RuntimeKeyStream:
        return self.stream("dropout")

    @property
    def augmentation(self) -> RuntimeKeyStream:
        return self.stream("augmentation")

    @property
    def evaluation(self) -> RuntimeKeyStream:
        return self.stream("evaluation")

    @property
    def runtime_tests(self) -> RuntimeKeyStream:
        return self.stream("runtime_tests")

    def stream(self, name: str) -> RuntimeKeyStream:
        if name not in RUNTIME_KEY_STREAM_NAMES:
            raise KeyError(f"unknown runtime key stream: {name}")
        return self.streams[RUNTIME_KEY_STREAM_NAMES.index(name)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_seed": self.root_seed,
            "version": self.version,
            "streams": [item.to_dict() for item in self.streams],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RuntimeKeys:
        raw_streams = payload.get("streams", ())
        if not isinstance(raw_streams, (list, tuple)):
            raise TypeError("streams must be a list or tuple")
        return cls(
            root_seed=_seed(payload["root_seed"], "root_seed"),
            version=_string(payload["version"], "version"),
            streams=tuple(
                RuntimeKeyStream.from_dict(_mapping(item, "stream"))
                for item in raw_streams
            ),
        )


def initialization_reference_from_root_seed(
    root_seed: int,
) -> RuntimeInitializationKeyReference:
    """Return the runtime-owned initialization reference without exposing a key."""
    return RuntimeKeys.from_seed(root_seed).initialization_reference


def _stream(root_seed: int, name: str) -> RuntimeKeyStream:
    lineage = ("root_seed", name)
    return RuntimeKeyStream(
        name=name,
        root_seed=root_seed,
        lineage=lineage,
        derived_seed=_derived_seed(root_seed, lineage),
        metadata={"derivation": "sha256", "version": RUNTIME_KEYS_VERSION},
    )


def _derived_seed(root_seed: int, lineage: tuple[str, ...]) -> int:
    payload = "\0".join((RUNTIME_KEYS_VERSION, str(root_seed), *lineage))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def jax_key_words(
    stream: RuntimeKeyStream,
    *,
    global_step: int,
    micro_step: int,
    slot: str,
    invocation_index: int = 0,
) -> tuple[int, int]:
    """Derive canonical JAX key words from runtime-owned stream identity."""

    if not isinstance(stream, RuntimeKeyStream):
        raise TypeError("stream must be RuntimeKeyStream")
    for name, value in (
        ("global_step", global_step),
        ("micro_step", micro_step),
        ("invocation_index", invocation_index),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if slot not in JAX_KEY_SLOTS:
        raise ValueError("slot must be a declared runtime JAX key slot")
    payload = "\0".join(
        (
            JAX_KEY_BRIDGE_VERSION,
            str(stream.root_seed),
            stream.name,
            str(global_step),
            str(micro_step),
            slot,
            str(invocation_index),
        )
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return (
        int.from_bytes(digest[0:4], "big"),
        int.from_bytes(digest[4:8], "big"),
    )


def _require_seed(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _seed(value: object, name: str) -> int:
    _require_seed(value, name)
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _strings(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple")
    result = tuple(value)
    if any(not isinstance(item, str) or not item for item in result):
        raise ValueError(f"{name} must contain nonempty strings")
    return result


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _freeze_json_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(
        {str(key): _json_value(item) for key, item in value.items()}
    )


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_json_value(item) for item in value)
    raise TypeError(
        f"runtime key metadata is not JSON-serializable: {type(value).__name__}"
    )
