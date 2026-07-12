"""Canonical logical, JAX-pytree, and optional HF parameter identities."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


def _stable_path(value: str, label: str) -> str:
    if not isinstance(value, str) or not value or value.startswith(("/", ".")):
        raise ValueError(f"{label} must be a stable relative path")
    return value


def _keypath(value: tuple[str, ...]) -> tuple[str, ...]:
    result = tuple(value)
    if not result or any(not isinstance(part, str) or not part for part in result):
        raise ValueError("jax_keypath must contain nonempty mapping keys")
    return result


@dataclass(frozen=True)
class ParameterTreeLayoutEntry:
    """One leaf identity; JAX trees are deliberately mapping-only in P3.11."""

    logical_path: str
    jax_keypath: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str
    role: str
    region_ids: tuple[str, ...] = ()
    trainable: bool = True
    exportable: bool = False
    hf_distribution_key: str | None = None
    tied_weight_group: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "logical_path", _stable_path(self.logical_path, "logical_path")
        )
        object.__setattr__(self, "jax_keypath", _keypath(self.jax_keypath))
        shape = tuple(self.shape)
        if any(
            not isinstance(size, int) or isinstance(size, bool) or size < 0
            for size in shape
        ):
            raise ValueError("shape must contain nonnegative integer dimensions")
        if not isinstance(self.dtype, str) or not self.dtype:
            raise ValueError("dtype must be nonempty")
        if not isinstance(self.role, str) or not self.role:
            raise ValueError("role must be nonempty")
        if not isinstance(self.trainable, bool) or not isinstance(
            self.exportable, bool
        ):
            raise TypeError("trainable and exportable must be booleans")
        regions = tuple(sorted(set(self.region_ids)))
        if any(not isinstance(region, str) or not region for region in regions):
            raise ValueError("region_ids must contain nonempty strings")
        if self.hf_distribution_key is not None:
            object.__setattr__(
                self,
                "hf_distribution_key",
                _stable_path(self.hf_distribution_key, "hf_distribution_key"),
            )
        if self.exportable != (self.hf_distribution_key is not None):
            raise ValueError(
                "exportable entries require exactly one HF distribution key"
            )
        if self.tied_weight_group == "":
            raise ValueError("tied_weight_group must be nonempty when specified")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "region_ids", regions)
        object.__setattr__(
            self, "metadata", MappingProxyType(dict(sorted(self.metadata.items())))
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "logical_path": self.logical_path,
            "jax_keypath": list(self.jax_keypath),
            "shape": list(self.shape),
            "dtype": self.dtype,
            "role": self.role,
            "region_ids": list(self.region_ids),
            "trainable": self.trainable,
            "exportable": self.exportable,
            "hf_distribution_key": self.hf_distribution_key,
            "tied_weight_group": self.tied_weight_group,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ParameterTreeLayout:
    """Immutable bijection used by architecture, optimizer, checkpoints, and HF."""

    architecture_id: str
    entries: tuple[ParameterTreeLayoutEntry, ...]
    schema_version: str = "parameter_tree_layout.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.architecture_id, str) or not self.architecture_id:
            raise ValueError("architecture_id must be nonempty")
        if self.schema_version != "parameter_tree_layout.v1":
            raise ValueError("unsupported parameter tree layout schema")
        entries = tuple(self.entries)
        if not entries or any(
            not isinstance(entry, ParameterTreeLayoutEntry) for entry in entries
        ):
            raise TypeError("entries must contain ParameterTreeLayoutEntry values")
        entries = tuple(sorted(entries, key=lambda entry: entry.logical_path))
        logical = [entry.logical_path for entry in entries]
        keypaths = [entry.jax_keypath for entry in entries]
        hf_keys = [entry.hf_distribution_key for entry in entries if entry.exportable]
        if len(logical) != len(set(logical)) or len(keypaths) != len(set(keypaths)):
            raise ValueError("logical paths and JAX keypaths must be bijective")
        if len(hf_keys) != len(set(hf_keys)):
            raise ValueError("exportable HF distribution keys must be unique")
        object.__setattr__(self, "entries", entries)

    @property
    def logical_paths(self) -> tuple[str, ...]:
        return tuple(entry.logical_path for entry in self.entries)

    @property
    def exportable_entries(self) -> tuple[ParameterTreeLayoutEntry, ...]:
        return tuple(entry for entry in self.entries if entry.exportable)

    def entry_for_logical_path(self, logical_path: str) -> ParameterTreeLayoutEntry:
        return next(
            entry for entry in self.entries if entry.logical_path == logical_path
        )

    def entry_for_jax_keypath(
        self, keypath: tuple[str, ...]
    ) -> ParameterTreeLayoutEntry:
        return next(
            entry for entry in self.entries if entry.jax_keypath == tuple(keypath)
        )

    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "architecture_id": self.architecture_id,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class JaxOptimizerStateDescriptor:
    """Canonical description of numerical optimizer leaves, not their values."""

    optimizer_id: str
    optimizer_capability: str
    optimizer_schema_version: str
    layout_digest: str
    state_keypaths: tuple[tuple[str, ...], ...]
    schema_version: str = "jax_optimizer_state_descriptor.v1"

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value
            for value in (
                self.optimizer_id,
                self.optimizer_capability,
                self.optimizer_schema_version,
                self.layout_digest,
            )
        ):
            raise ValueError("optimizer state descriptor identities must be nonempty")
        paths = tuple(_keypath(path) for path in self.state_keypaths)
        if len(paths) != len(set(paths)):
            raise ValueError("optimizer state keypaths must be unique")
        object.__setattr__(self, "state_keypaths", tuple(sorted(paths)))
