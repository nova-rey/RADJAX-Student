"""Runtime-owned identity binding for supported production callables.

The registry is deliberately narrow: it recognizes declared, top-level production
functions only.  It never accepts caller-provided identity material.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import partial
from typing import Any

CALLABLE_DECLARATION_SCHEMA_VERSION = "radjax.runtime_callable_declaration.v1"
CALLABLE_IDENTITY_SCHEMA_VERSION = "radjax.runtime_callable_identity.v1"
CALLABLE_REFERENCE_SCHEMA_VERSION = "radjax.runtime_callable_reference.v1"
PREPARED_IDENTITY_SCHEMA_VERSION = "radjax.runtime_prepared_execution_identity.v1"


class RuntimeCallableError(ValueError):
    """Stable runtime callable-identity rejection."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


def _canonical(value: object) -> str:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256((_canonical(value) + "\n").encode()).hexdigest()


def _sha(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or set(value) - set("0123456789abcdef")
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty identifier")
    return value


def _strict_mapping(payload: object, fields: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping) or set(payload) != fields:
        raise ValueError(f"{name} fields are missing or unknown")
    return payload


@dataclass(frozen=True)
class RuntimeCallableDeclaration:
    schema_version: str
    callable_id: str
    callable_version: int
    owner: str
    implementation_module: str
    implementation_qualname: str
    input_contract_id: str
    output_contract_id: str
    claims_not_made: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != CALLABLE_DECLARATION_SCHEMA_VERSION:
            raise RuntimeCallableError(
                "execution_callable_declaration_invalid",
                "unsupported callable declaration schema",
            )
        try:
            for name in (
                "callable_id",
                "owner",
                "implementation_module",
                "implementation_qualname",
                "input_contract_id",
                "output_contract_id",
            ):
                _identifier(getattr(self, name), name)
        except ValueError as exc:
            raise RuntimeCallableError(
                "execution_callable_declaration_invalid", str(exc)
            ) from exc
        if (
            isinstance(self.callable_version, bool)
            or not isinstance(self.callable_version, int)
            or self.callable_version <= 0
        ):
            raise RuntimeCallableError(
                "execution_callable_declaration_invalid",
                "callable_version must be positive",
            )
        if (
            not self.implementation_module.startswith("radjax_student.")
            or ".validation" in self.implementation_module
            or ".tests" in self.implementation_module
        ):
            raise RuntimeCallableError(
                "execution_callable_unsupported",
                "callable declaration module must be production-owned",
            )
        if "<" in self.implementation_qualname or "." in self.implementation_qualname:
            raise RuntimeCallableError(
                "execution_callable_declaration_invalid",
                "callable declaration must name a top-level function",
            )
        claims = tuple(self.claims_not_made)
        if len(set(claims)) != len(claims) or not all(
            isinstance(item, str) and item for item in claims
        ):
            raise RuntimeCallableError(
                "execution_callable_declaration_invalid",
                "claims_not_made must be unique nonempty strings",
            )
        object.__setattr__(self, "claims_not_made", claims)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "callable_id": self.callable_id,
            "callable_version": self.callable_version,
            "owner": self.owner,
            "implementation_module": self.implementation_module,
            "implementation_qualname": self.implementation_qualname,
            "input_contract_id": self.input_contract_id,
            "output_contract_id": self.output_contract_id,
            "claims_not_made": list(self.claims_not_made),
        }

    @classmethod
    def from_dict(cls, payload: object) -> RuntimeCallableDeclaration:
        item = _strict_mapping(
            payload,
            {
                "schema_version",
                "callable_id",
                "callable_version",
                "owner",
                "implementation_module",
                "implementation_qualname",
                "input_contract_id",
                "output_contract_id",
                "claims_not_made",
            },
            "callable declaration",
        )
        return cls(
            item["schema_version"],
            item["callable_id"],
            item["callable_version"],
            item["owner"],
            item["implementation_module"],
            item["implementation_qualname"],
            item["input_contract_id"],
            item["output_contract_id"],
            tuple(item["claims_not_made"]),
        )

    @property
    def digest(self) -> str:
        return _digest(self.to_dict())


@dataclass(frozen=True)
class RuntimeCallableIdentity:
    schema_version: str
    callable_id: str
    callable_version: int
    owner: str
    implementation_module: str
    implementation_qualname: str
    implementation_source_digest: str
    input_contract_digest: str
    output_contract_digest: str
    declaration_digest: str
    callable_identity_digest: str

    def __post_init__(self) -> None:
        if self.schema_version != CALLABLE_IDENTITY_SCHEMA_VERSION:
            raise ValueError("unsupported callable identity schema")
        for name in (
            "callable_id",
            "owner",
            "implementation_module",
            "implementation_qualname",
        ):
            _identifier(getattr(self, name), name)
        if (
            isinstance(self.callable_version, bool)
            or not isinstance(self.callable_version, int)
            or self.callable_version <= 0
        ):
            raise ValueError("callable_version must be positive")
        for name in (
            "implementation_source_digest",
            "input_contract_digest",
            "output_contract_digest",
            "declaration_digest",
            "callable_identity_digest",
        ):
            _sha(getattr(self, name), name)
        expected = _digest(
            {
                key: value
                for key, value in self.to_dict().items()
                if key not in {"schema_version", "callable_identity_digest"}
            }
        )
        if self.callable_identity_digest != expected:
            raise ValueError("callable identity digest is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "schema_version",
                "callable_id",
                "callable_version",
                "owner",
                "implementation_module",
                "implementation_qualname",
                "implementation_source_digest",
                "input_contract_digest",
                "output_contract_digest",
                "declaration_digest",
                "callable_identity_digest",
            )
        }

    @classmethod
    def from_dict(cls, payload: object) -> RuntimeCallableIdentity:
        item = _strict_mapping(
            payload,
            {
                "schema_version",
                "callable_id",
                "callable_version",
                "owner",
                "implementation_module",
                "implementation_qualname",
                "implementation_source_digest",
                "input_contract_digest",
                "output_contract_digest",
                "declaration_digest",
                "callable_identity_digest",
            },
            "callable identity",
        )
        return cls(**item)


@dataclass(frozen=True)
class RuntimeCallableReference:
    schema_version: str
    callable_id: str
    callable_version: int
    callable_identity_digest: str

    def __post_init__(self) -> None:
        if self.schema_version != CALLABLE_REFERENCE_SCHEMA_VERSION:
            raise RuntimeCallableError(
                "execution_callable_reference_invalid",
                "unsupported callable reference schema",
            )
        try:
            _identifier(self.callable_id, "callable_id")
        except ValueError as exc:
            raise RuntimeCallableError(
                "execution_callable_reference_invalid", str(exc)
            ) from exc
        if (
            isinstance(self.callable_version, bool)
            or not isinstance(self.callable_version, int)
            or self.callable_version <= 0
        ):
            raise RuntimeCallableError(
                "execution_callable_reference_invalid",
                "callable_version must be positive",
            )
        try:
            _sha(self.callable_identity_digest, "callable_identity_digest")
        except ValueError as exc:
            raise RuntimeCallableError(
                "execution_callable_reference_invalid", str(exc)
            ) from exc

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "callable_id": self.callable_id,
            "callable_version": self.callable_version,
            "callable_identity_digest": self.callable_identity_digest,
        }

    @classmethod
    def from_dict(cls, payload: object) -> RuntimeCallableReference:
        return cls(
            **_strict_mapping(
                payload,
                {
                    "schema_version",
                    "callable_id",
                    "callable_version",
                    "callable_identity_digest",
                },
                "callable reference",
            )
        )


@dataclass(frozen=True)
class RuntimePreparedExecutionIdentity:
    """Final identity, intentionally created only when actual arguments exist."""

    schema_version: str
    callable_reference: RuntimeCallableReference
    backend_id: str
    runtime_id: str
    runtime_implementation_version: str
    mode: str
    compilation_options_digest: str
    input_signature_digest: str
    static_argument_contract_digest: str
    static_argument_value_digest: str
    donation_contract_digest: str
    placement_plan_identity: str | None
    required_capabilities_digest: str
    prepared_execution_digest: str

    def __post_init__(self) -> None:
        if self.schema_version != PREPARED_IDENTITY_SCHEMA_VERSION:
            raise ValueError("unsupported prepared execution identity schema")
        if not isinstance(self.callable_reference, RuntimeCallableReference):
            raise TypeError("callable_reference must be RuntimeCallableReference")
        for name in (
            "backend_id",
            "runtime_id",
            "runtime_implementation_version",
            "mode",
        ):
            _identifier(getattr(self, name), name)
        if self.placement_plan_identity is not None:
            _identifier(self.placement_plan_identity, "placement_plan_identity")
        for name in (
            "compilation_options_digest",
            "input_signature_digest",
            "static_argument_contract_digest",
            "static_argument_value_digest",
            "donation_contract_digest",
            "required_capabilities_digest",
            "prepared_execution_digest",
        ):
            _sha(getattr(self, name), name)
        expected = _digest(
            {
                key: value
                for key, value in self._digest_payload().items()
                if key != "prepared_execution_digest"
            }
        )
        if self.prepared_execution_digest != expected:
            raise ValueError("prepared execution digest is invalid")

    def _digest_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "callable_reference": self.callable_reference.to_dict(),
            "backend_id": self.backend_id,
            "runtime_id": self.runtime_id,
            "runtime_implementation_version": self.runtime_implementation_version,
            "mode": self.mode,
            "compilation_options_digest": self.compilation_options_digest,
            "input_signature_digest": self.input_signature_digest,
            "static_argument_contract_digest": self.static_argument_contract_digest,
            "static_argument_value_digest": self.static_argument_value_digest,
            "donation_contract_digest": self.donation_contract_digest,
            "placement_plan_identity": self.placement_plan_identity,
            "required_capabilities_digest": self.required_capabilities_digest,
            "prepared_execution_digest": self.prepared_execution_digest,
        }

    def to_dict(self) -> dict[str, object]:
        return self._digest_payload()

    @classmethod
    def from_dict(cls, payload: object) -> RuntimePreparedExecutionIdentity:
        fields = {
            "schema_version",
            "callable_reference",
            "backend_id",
            "runtime_id",
            "runtime_implementation_version",
            "mode",
            "compilation_options_digest",
            "input_signature_digest",
            "static_argument_contract_digest",
            "static_argument_value_digest",
            "donation_contract_digest",
            "placement_plan_identity",
            "required_capabilities_digest",
            "prepared_execution_digest",
        }
        item = _strict_mapping(payload, fields, "prepared execution identity")
        return cls(
            schema_version=item["schema_version"],
            callable_reference=RuntimeCallableReference.from_dict(
                item["callable_reference"]
            ),
            backend_id=item["backend_id"],
            runtime_id=item["runtime_id"],
            runtime_implementation_version=item["runtime_implementation_version"],
            mode=item["mode"],
            compilation_options_digest=item["compilation_options_digest"],
            input_signature_digest=item["input_signature_digest"],
            static_argument_contract_digest=item["static_argument_contract_digest"],
            static_argument_value_digest=item["static_argument_value_digest"],
            donation_contract_digest=item["donation_contract_digest"],
            placement_plan_identity=item["placement_plan_identity"],
            required_capabilities_digest=item["required_capabilities_digest"],
            prepared_execution_digest=item["prepared_execution_digest"],
        )


def final_prepared_execution_identity(
    *,
    reference: RuntimeCallableReference,
    backend_id: str,
    runtime_id: str,
    runtime_implementation_version: str,
    mode: str,
    compilation_options: Mapping[str, Any],
    input_signature: Mapping[str, Any],
    static_contract: Mapping[str, Any],
    static_values: Mapping[str, Any],
    donation_contract: Mapping[str, Any],
    placement_plan_identity: str | None,
    required_capabilities: tuple[str, ...],
) -> RuntimePreparedExecutionIdentity:
    """Derive identity at the compile boundary from actual static values."""
    digest_payload = {
        "schema_version": PREPARED_IDENTITY_SCHEMA_VERSION,
        "callable_reference": reference.to_dict(),
        "backend_id": backend_id,
        "runtime_id": runtime_id,
        "runtime_implementation_version": runtime_implementation_version,
        "mode": mode,
        "compilation_options_digest": _digest(compilation_options),
        "input_signature_digest": _digest(input_signature),
        "static_argument_contract_digest": _digest(static_contract),
        "static_argument_value_digest": _digest(static_values),
        "donation_contract_digest": _digest(donation_contract),
        "placement_plan_identity": placement_plan_identity,
        "required_capabilities_digest": _digest(list(required_capabilities)),
    }
    payload = {**digest_payload, "callable_reference": reference}
    return RuntimePreparedExecutionIdentity(
        **payload,
        prepared_execution_digest=_digest(digest_payload),
    )


@dataclass(frozen=True)
class RuntimeCallableBinding:
    callable: Callable[..., Any] = field(repr=False, compare=False)
    declaration: RuntimeCallableDeclaration
    identity: RuntimeCallableIdentity

    def __post_init__(self) -> None:
        if (
            not callable(self.callable)
            or not isinstance(self.declaration, RuntimeCallableDeclaration)
            or not isinstance(self.identity, RuntimeCallableIdentity)
        ):
            raise TypeError(
                "runtime callable binding requires callable, declaration, and identity"
            )
        if self.reference.callable_id != self.declaration.callable_id:
            raise RuntimeCallableError(
                "execution_callable_identity_mismatch",
                "binding identity does not match declaration",
            )

    @property
    def reference(self) -> RuntimeCallableReference:
        return RuntimeCallableReference(
            CALLABLE_REFERENCE_SCHEMA_VERSION,
            self.identity.callable_id,
            self.identity.callable_version,
            self.identity.callable_identity_digest,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "declaration": self.declaration.to_dict(),
            "identity": self.identity.to_dict(),
        }


def _source_digest(function: Callable[..., Any]) -> str:
    try:
        source = inspect.getsource(function)
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError) as exc:
        raise RuntimeCallableError(
            "execution_callable_source_unavailable", "callable source is unavailable"
        ) from exc
    function_nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if len(function_nodes) != 1 or function_nodes[0].name != function.__name__:
        raise RuntimeCallableError(
            "execution_callable_source_invalid",
            "callable source is not one top-level function",
        )
    return _digest(
        ast.dump(function_nodes[0], annotate_fields=True, include_attributes=False)
    )


def bind_runtime_callable(
    *, callable: Callable[..., Any], declaration: RuntimeCallableDeclaration
) -> RuntimeCallableBinding:
    """Validate and bind one actual supported production callable."""
    if not isinstance(declaration, RuntimeCallableDeclaration):
        raise RuntimeCallableError(
            "execution_callable_declaration_invalid",
            "declaration must be RuntimeCallableDeclaration",
        )
    if (
        isinstance(callable, partial)
        or not inspect.isfunction(callable)
        or callable.__name__ == "<lambda>"
        or "<locals>" in callable.__qualname__
        or callable.__closure__ is not None
    ):
        raise RuntimeCallableError(
            "execution_callable_unsupported",
            "callable is not a supported top-level production function",
        )
    if (
        callable.__module__ != declaration.implementation_module
        or callable.__qualname__ != declaration.implementation_qualname
    ):
        raise RuntimeCallableError(
            "execution_callable_identity_mismatch",
            "actual callable does not match its declaration",
        )
    source_digest = _source_digest(callable)
    input_digest = _digest({"input_contract_id": declaration.input_contract_id})
    output_digest = _digest({"output_contract_id": declaration.output_contract_id})
    base = {
        "callable_id": declaration.callable_id,
        "callable_version": declaration.callable_version,
        "owner": declaration.owner,
        "implementation_module": declaration.implementation_module,
        "implementation_qualname": declaration.implementation_qualname,
        "implementation_source_digest": source_digest,
        "input_contract_digest": input_digest,
        "output_contract_digest": output_digest,
        "declaration_digest": declaration.digest,
    }
    identity = RuntimeCallableIdentity(
        CALLABLE_IDENTITY_SCHEMA_VERSION, **base, callable_identity_digest=_digest(base)
    )
    return RuntimeCallableBinding(callable, declaration, identity)


class RuntimeCallableRegistry:
    """Exact, non-discovering registry of already-bound production callables."""

    def __init__(self) -> None:
        self._bindings: dict[tuple[str, int], RuntimeCallableBinding] = {}

    def register(self, binding: RuntimeCallableBinding) -> None:
        if not isinstance(binding, RuntimeCallableBinding):
            raise TypeError("binding must be RuntimeCallableBinding")
        key = (binding.reference.callable_id, binding.reference.callable_version)
        existing = self._bindings.get(key)
        if existing is not None:
            raise RuntimeCallableError(
                "execution_callable_unregistered",
                "duplicate or conflicting callable registration",
            )
        self._bindings[key] = binding

    def resolve(self, reference: RuntimeCallableReference) -> RuntimeCallableBinding:
        if not isinstance(reference, RuntimeCallableReference):
            raise RuntimeCallableError(
                "execution_callable_reference_invalid",
                "reference must be RuntimeCallableReference",
            )
        binding = self._bindings.get(
            (reference.callable_id, reference.callable_version)
        )
        if binding is None:
            raise RuntimeCallableError(
                "execution_callable_unregistered",
                "callable reference is not registered",
            )
        if binding.reference != reference:
            raise RuntimeCallableError(
                "execution_callable_request_mismatch",
                "callable reference does not match registered binding",
            )
        return binding

    def lookup(self, callable_id: str, callable_version: int) -> RuntimeCallableBinding:
        """Resolve one exact production declaration without aliases or discovery."""
        binding = self._bindings.get((callable_id, callable_version))
        if binding is None:
            raise RuntimeCallableError(
                "execution_callable_unregistered",
                "callable declaration is not registered",
            )
        return binding


def build_default_runtime_callable_registry() -> RuntimeCallableRegistry:
    """Return the explicit production registry without module discovery."""
    # Local import prevents optional JAX learning code from entering base imports.
    from radjax_student.steps.jax_step import (
        GENERIC_JAX_LEARNING_STEP_DECLARATION,
        execute_jax_learning_step_kernel,
    )

    registry = RuntimeCallableRegistry()
    registry.register(
        bind_runtime_callable(
            callable=execute_jax_learning_step_kernel,
            declaration=GENERIC_JAX_LEARNING_STEP_DECLARATION,
        )
    )
    return registry
