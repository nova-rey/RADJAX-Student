"""Portable persistence for the small, runtime-owned P2.8 state envelope."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from radjax_student.runtime.errors import RuntimeContractError, RuntimeIssue
from radjax_student.runtime.inspection import (
    RuntimeInspection,
    inspect_runtime_environment,
)
from radjax_student.runtime.keys import RuntimeKeys
from radjax_student.runtime.models import (
    RUNTIME_STATE_SCHEMA_VERSION,
    ExecutionContext,
    RuntimeConfig,
    RuntimeState,
    json_value,
)

RUNTIME_STATE_ARTIFACT_KIND = "radjax_student.runtime_state"
RUNTIME_STATE_MANIFEST_FILE = "manifest.json"
RUNTIME_STATE_FILE = "runtime_state.json"
RUNTIME_STATE_FILES: tuple[str, ...] = (
    RUNTIME_STATE_MANIFEST_FILE,
    RUNTIME_STATE_FILE,
)
RUNTIME_STATE_HASH_ALGORITHM = "sha256"
RUNTIME_STATE_CLAIMS_NOT_MADE: tuple[str, ...] = (
    "model_checkpoint_not_saved",
    "optimizer_checkpoint_not_saved",
    "training_resume_not_proven",
    "topology_migration_not_proven",
    "compiled_executables_not_persisted",
    "execution_equivalence_not_proven",
)
RUNTIME_STATE_WARNING_CODES: tuple[str, ...] = (
    "runtime_state_topology_changed",
    "runtime_state_environment_changed",
    "runtime_state_resume_not_execution_proof",
    "runtime_state_contains_runtime_only",
    "runtime_state_unknown_optional_metadata",
)
RUNTIME_STATE_BLOCKER_CODES: tuple[str, ...] = (
    "runtime_state_path_unsafe",
    "runtime_state_exists",
    "runtime_state_missing",
    "runtime_state_manifest_invalid",
    "runtime_state_schema_unsupported",
    "runtime_state_hash_mismatch",
    "runtime_state_size_mismatch",
    "runtime_state_config_invalid",
    "runtime_state_rng_invalid",
    "runtime_state_step_invalid",
    "runtime_state_resume_incompatible",
    "runtime_state_save_failed",
    "runtime_state_load_failed",
    "runtime_state_internal_error",
)
_FORBIDDEN_STATE_KEYS = frozenset(
    {
        "architecture_state",
        "compiled_executables",
        "jax_devices",
        "jax_keys",
        "model_parameters",
        "optimizer_state",
        "tome_payload",
        "training_batches",
    }
)
_STATE_FIELDS = frozenset(
    {
        "schema_version",
        "runtime_id",
        "global_step",
        "root_seed",
        "runtime_keys",
        "runtime_config",
        "environment_summary",
        "topology_summary",
        "precision_policy",
        "placement_policy",
        "backend_id",
        "resume_metadata",
        "claims_not_made",
    }
)


@dataclass(frozen=True)
class RuntimeStateIntegrity:
    algorithm: str
    state_digest: str
    manifest_digest: str
    verified: bool

    def __post_init__(self) -> None:
        if self.algorithm != RUNTIME_STATE_HASH_ALGORITHM:
            raise ValueError("unsupported runtime-state integrity algorithm")
        for name in ("state_digest", "manifest_digest"):
            value = getattr(self, name)
            if not _is_digest(value):
                raise ValueError(f"{name} must be a SHA-256 hexadecimal digest")
        if not isinstance(self.verified, bool):
            raise TypeError("verified must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "state_digest": self.state_digest,
            "manifest_digest": self.manifest_digest,
            "verified": self.verified,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RuntimeStateIntegrity:
        return cls(
            algorithm=_string(payload.get("algorithm"), "integrity.algorithm"),
            state_digest=_string(payload.get("state_digest"), "integrity.state_digest"),
            manifest_digest=_string(
                payload.get("manifest_digest"), "integrity.manifest_digest"
            ),
            verified=_bool(payload.get("verified"), "integrity.verified"),
        )


@dataclass(frozen=True)
class RuntimeStateManifest:
    artifact_kind: str
    schema_version: str
    files: tuple[str, ...]
    hashes: Mapping[str, str]
    sizes: Mapping[str, int]
    created_by: str
    claims_not_made: tuple[str, ...]
    integrity: RuntimeStateIntegrity

    def __post_init__(self) -> None:
        if self.artifact_kind != RUNTIME_STATE_ARTIFACT_KIND:
            raise ValueError("unsupported runtime-state artifact kind")
        _require_schema(self.schema_version)
        files = tuple(self.files)
        if files != (RUNTIME_STATE_FILE,):
            raise ValueError(
                "runtime-state manifest must contain only runtime_state.json"
            )
        _validate_internal_file_names(files)
        hashes = _hash_mapping(self.hashes)
        sizes = _size_mapping(self.sizes)
        if set(hashes) != set(files) or set(sizes) != set(files):
            raise ValueError(
                "runtime-state manifest files, hashes, and sizes must match"
            )
        if not isinstance(self.created_by, str) or not self.created_by:
            raise ValueError("created_by must be a nonempty string")
        claims = _unique_strings(self.claims_not_made, "claims_not_made")
        if not isinstance(self.integrity, RuntimeStateIntegrity):
            raise TypeError("integrity must be RuntimeStateIntegrity")
        object.__setattr__(self, "files", files)
        object.__setattr__(self, "hashes", MappingProxyType(hashes))
        object.__setattr__(self, "sizes", MappingProxyType(sizes))
        object.__setattr__(self, "claims_not_made", claims)

    def base_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": self.artifact_kind,
            "schema_version": self.schema_version,
            "files": list(self.files),
            "hashes": dict(self.hashes),
            "sizes": dict(self.sizes),
            "created_by": self.created_by,
            "claims_not_made": list(self.claims_not_made),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.base_dict(), "integrity": self.integrity.to_dict()}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RuntimeStateManifest:
        _reject_unknown_fields(
            payload,
            {
                "artifact_kind",
                "schema_version",
                "files",
                "hashes",
                "sizes",
                "created_by",
                "claims_not_made",
                "integrity",
            },
            "runtime-state manifest",
        )
        return cls(
            artifact_kind=_string(payload.get("artifact_kind"), "artifact_kind"),
            schema_version=_string(payload.get("schema_version"), "schema_version"),
            files=_strings(payload.get("files"), "files"),
            hashes=_string_mapping(payload.get("hashes"), "hashes"),
            sizes=_integer_mapping(payload.get("sizes"), "sizes"),
            created_by=_string(payload.get("created_by"), "created_by"),
            claims_not_made=_strings(payload.get("claims_not_made"), "claims_not_made"),
            integrity=RuntimeStateIntegrity.from_dict(
                _mapping(payload.get("integrity"), "integrity")
            ),
        )


@dataclass(frozen=True)
class RuntimeStateBundle:
    state: RuntimeState
    manifest: RuntimeStateManifest
    integrity: RuntimeStateIntegrity

    def __post_init__(self) -> None:
        if not isinstance(self.state, RuntimeState):
            raise TypeError("state must be RuntimeState")
        if not isinstance(self.manifest, RuntimeStateManifest):
            raise TypeError("manifest must be RuntimeStateManifest")
        if self.integrity != self.manifest.integrity:
            raise ValueError("bundle integrity must match manifest integrity")


@dataclass(frozen=True)
class RuntimeStateSaveReceipt:
    status: Literal["pass"]
    output_dir: str
    schema_version: str
    files: tuple[str, ...]
    hashes: Mapping[str, str]
    sizes: Mapping[str, int]
    blockers: tuple[RuntimeIssue, ...] = ()
    warnings: tuple[RuntimeIssue, ...] = ()
    claims_not_made: tuple[str, ...] = RUNTIME_STATE_CLAIMS_NOT_MADE

    def __post_init__(self) -> None:
        if self.status != "pass":
            raise ValueError("runtime-state save receipt status must be pass")
        _require_schema(self.schema_version)
        _validate_internal_file_names(self.files)
        object.__setattr__(self, "hashes", MappingProxyType(_hash_mapping(self.hashes)))
        object.__setattr__(self, "sizes", MappingProxyType(_size_mapping(self.sizes)))
        object.__setattr__(self, "blockers", _issues(self.blockers, "blockers"))
        object.__setattr__(self, "warnings", _issues(self.warnings, "warnings"))
        object.__setattr__(
            self,
            "claims_not_made",
            _unique_strings(self.claims_not_made, "claims_not_made"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "output_dir": self.output_dir,
            "schema_version": self.schema_version,
            "files": list(self.files),
            "hashes": dict(self.hashes),
            "sizes": dict(self.sizes),
            "blockers": [item.to_dict() for item in self.blockers],
            "warnings": [item.to_dict() for item in self.warnings],
            "claims_not_made": list(self.claims_not_made),
        }


@dataclass(frozen=True)
class RuntimeStateLoadReceipt:
    status: Literal["pass"]
    source_dir: str
    schema_version: str
    verified_files: tuple[str, ...]
    blockers: tuple[RuntimeIssue, ...] = ()
    warnings: tuple[RuntimeIssue, ...] = ()
    claims_not_made: tuple[str, ...] = RUNTIME_STATE_CLAIMS_NOT_MADE

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "source_dir": self.source_dir,
            "schema_version": self.schema_version,
            "verified_files": list(self.verified_files),
            "blockers": [item.to_dict() for item in self.blockers],
            "warnings": [item.to_dict() for item in self.warnings],
            "claims_not_made": list(self.claims_not_made),
        }


@dataclass(frozen=True)
class RuntimeResumeCompatibilityReport:
    status: Literal["pass", "fail"]
    blockers: tuple[RuntimeIssue, ...] = ()
    warnings: tuple[RuntimeIssue, ...] = ()
    claims_not_made: tuple[str, ...] = (
        "resume_compatibility_is_not_execution_equivalence",
        "resume_compatibility_does_not_compare_architecture_or_model_state",
    )

    def __post_init__(self) -> None:
        blockers = _issues(self.blockers, "blockers")
        warnings = _issues(self.warnings, "warnings")
        if self.status == "pass" and blockers:
            raise ValueError("passing resume compatibility cannot have blockers")
        if self.status == "fail" and not blockers:
            raise ValueError("failing resume compatibility requires blockers")
        object.__setattr__(self, "blockers", blockers)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(
            self,
            "claims_not_made",
            _unique_strings(self.claims_not_made, "claims_not_made"),
        )

    @property
    def ok(self) -> bool:
        return self.status == "pass"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "blockers": [item.to_dict() for item in self.blockers],
            "warnings": [item.to_dict() for item in self.warnings],
            "claims_not_made": list(self.claims_not_made),
        }


@dataclass(frozen=True)
class RuntimeStateSmokeReceipt:
    status: Literal["pass", "fail"]
    schema_version: str
    global_step: int | None
    seed_tree_verified: bool
    config_round_trip: bool
    digest_verified: bool
    topology_restored_as_metadata: bool
    continuation_execution_succeeds: bool
    model_state_included: bool = False
    optimizer_state_included: bool = False
    blockers: tuple[RuntimeIssue, ...] = ()
    warnings: tuple[RuntimeIssue, ...] = ()
    claims_not_made: tuple[str, ...] = RUNTIME_STATE_CLAIMS_NOT_MADE

    def __post_init__(self) -> None:
        _require_schema(self.schema_version)
        if self.global_step is not None and (
            isinstance(self.global_step, bool)
            or not isinstance(self.global_step, int)
            or self.global_step < 0
        ):
            raise ValueError("global_step must be a nonnegative integer or None")
        blockers = _issues(self.blockers, "blockers")
        if self.status == "pass" and blockers:
            raise ValueError("passing runtime-state smoke cannot have blockers")
        if self.status == "fail" and not blockers:
            raise ValueError("failing runtime-state smoke requires blockers")
        object.__setattr__(self, "blockers", blockers)
        object.__setattr__(self, "warnings", _issues(self.warnings, "warnings"))
        object.__setattr__(
            self,
            "claims_not_made",
            _unique_strings(self.claims_not_made, "claims_not_made"),
        )

    @property
    def ok(self) -> bool:
        return self.status == "pass"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "schema_version": self.schema_version,
            "global_step": self.global_step,
            "seed_tree_verified": self.seed_tree_verified,
            "config_round_trip": self.config_round_trip,
            "digest_verified": self.digest_verified,
            "topology_restored_as_metadata": self.topology_restored_as_metadata,
            "continuation_execution_succeeds": self.continuation_execution_succeeds,
            "model_state_included": self.model_state_included,
            "optimizer_state_included": self.optimizer_state_included,
            "blockers": [item.to_dict() for item in self.blockers],
            "warnings": [item.to_dict() for item in self.warnings],
            "claims_not_made": list(self.claims_not_made),
        }


def canonical_runtime_state_json(value: Mapping[str, Any]) -> bytes:
    """Encode finite JSON as deterministic UTF-8, sorted-key, newline JSON."""

    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RuntimeContractError(
            "runtime_state_save_failed",
            "runtime state cannot be represented as canonical JSON",
            details={"exception_type": type(exc).__name__},
        ) from exc


def runtime_state_from_context(
    context: ExecutionContext,
    config: RuntimeConfig,
    *,
    global_step: int,
    resume_metadata: Mapping[str, Any] | None = None,
) -> RuntimeState:
    """Create state metadata from initialized runtime facts, never backend handles."""

    if not isinstance(context, ExecutionContext):
        raise TypeError("context must be ExecutionContext")
    if not isinstance(config, RuntimeConfig):
        raise TypeError("config must be RuntimeConfig")
    if context.root_seed != config.seed:
        raise RuntimeContractError(
            "runtime_state_rng_invalid",
            "execution context root seed must match runtime configuration seed",
            details={
                "context_root_seed": context.root_seed,
                "config_seed": config.seed,
            },
        )
    environment = context.environment
    inventory = context.device_inventory
    device_kinds = tuple(
        sorted(
            {
                device.device_kind
                for device in inventory.devices
                if device.device_kind is not None
            }
        )
    )
    return RuntimeState(
        runtime_id=context.runtime_id,
        global_step=global_step,
        root_seed=context.root_seed,
        runtime_keys=context.runtime_keys,
        runtime_config=config,
        environment_summary=environment.to_dict(),
        topology_summary={
            "platform": environment.platform,
            "process_count": environment.process_count,
            "process_index": environment.process_index,
            "local_device_count": inventory.local_device_count,
            "global_device_count": inventory.global_device_count,
            "device_kinds": device_kinds,
            "topology_labels": json_value(inventory.topology_summary),
        },
        precision_policy=config.precision_policy,
        placement_policy=config.placement_policy,
        backend_id=context.backend_id,
        resume_metadata={} if resume_metadata is None else resume_metadata,
    )


def save_runtime_state(
    state: RuntimeState,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> RuntimeStateSaveReceipt:
    """Atomically write a small manifest-last runtime state artifact."""

    if not isinstance(state, RuntimeState):
        raise TypeError("state must be RuntimeState")
    destination = _safe_destination(output_dir)
    _prepare_destination(destination, overwrite=overwrite)
    state_bytes = canonical_runtime_state_json(state.to_dict())
    state_digest = _sha256(state_bytes)
    base_manifest = {
        "artifact_kind": RUNTIME_STATE_ARTIFACT_KIND,
        "schema_version": state.schema_version,
        "files": [RUNTIME_STATE_FILE],
        "hashes": {RUNTIME_STATE_FILE: state_digest},
        "sizes": {RUNTIME_STATE_FILE: len(state_bytes)},
        "created_by": "radjax_student",
        "claims_not_made": list(RUNTIME_STATE_CLAIMS_NOT_MADE),
    }
    integrity = RuntimeStateIntegrity(
        algorithm=RUNTIME_STATE_HASH_ALGORITHM,
        state_digest=state_digest,
        manifest_digest=_sha256(canonical_runtime_state_json(base_manifest)),
        verified=True,
    )
    manifest = RuntimeStateManifest(
        **base_manifest,
        integrity=integrity,
    )
    manifest_bytes = canonical_runtime_state_json(manifest.to_dict())
    try:
        destination.mkdir(parents=True, exist_ok=True)
        _atomic_write(destination / RUNTIME_STATE_FILE, state_bytes)
        _atomic_write(destination / RUNTIME_STATE_MANIFEST_FILE, manifest_bytes)
    except RuntimeContractError:
        raise
    except OSError as exc:
        raise RuntimeContractError(
            "runtime_state_save_failed",
            "runtime state could not be written",
            details={"exception_type": type(exc).__name__},
        ) from exc
    return RuntimeStateSaveReceipt(
        status="pass",
        output_dir=str(destination),
        schema_version=state.schema_version,
        files=(RUNTIME_STATE_FILE, RUNTIME_STATE_MANIFEST_FILE),
        hashes={RUNTIME_STATE_FILE: state_digest},
        sizes={RUNTIME_STATE_FILE: len(state_bytes)},
        warnings=(
            RuntimeIssue.create(
                "runtime_state_contains_runtime_only",
                "saved state contains runtime identity and policy only",
            ),
        ),
    )


def load_runtime_state(path: str | Path) -> RuntimeState:
    """Load only after path, manifest, size, hash, schema, and RNG validation."""

    return load_runtime_state_bundle(path).state


def load_runtime_state_bundle(path: str | Path) -> RuntimeStateBundle:
    source = _safe_source(path)
    manifest_path = source / RUNTIME_STATE_MANIFEST_FILE
    state_path = source / RUNTIME_STATE_FILE
    _require_regular_file(manifest_path, "runtime_state_manifest_invalid")
    _require_regular_file(state_path, "runtime_state_missing")
    manifest_payload = _read_json(manifest_path, "runtime_state_manifest_invalid")
    if manifest_payload.get("schema_version") != RUNTIME_STATE_SCHEMA_VERSION:
        raise RuntimeContractError(
            "runtime_state_schema_unsupported",
            "runtime-state manifest schema version is unsupported",
            details={"schema_version": manifest_payload.get("schema_version")},
        )
    try:
        manifest = RuntimeStateManifest.from_dict(manifest_payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeContractError(
            "runtime_state_manifest_invalid",
            "runtime-state manifest is invalid",
            details={"exception_type": type(exc).__name__},
        ) from exc
    _validate_manifest_integrity(manifest)
    state_bytes = _read_bytes(state_path, "runtime_state_missing")
    _verify_state_file(manifest, state_bytes)
    state_payload = _parse_json(state_bytes, "runtime_state_load_failed")
    _validate_state_payload(state_payload)
    try:
        state = RuntimeState.from_dict(state_payload)
    except (KeyError, TypeError, ValueError) as exc:
        code = _state_model_error_code(exc)
        raise RuntimeContractError(
            code,
            "runtime-state payload is invalid",
            details={"exception_type": type(exc).__name__},
        ) from exc
    if state.schema_version != manifest.schema_version:
        raise RuntimeContractError(
            "runtime_state_schema_unsupported",
            "runtime-state payload and manifest schema versions differ",
            details={
                "manifest_schema_version": manifest.schema_version,
                "state_schema_version": state.schema_version,
            },
        )
    return RuntimeStateBundle(
        state=state, manifest=manifest, integrity=manifest.integrity
    )


def load_runtime_state_with_receipt(
    path: str | Path,
) -> tuple[RuntimeState, RuntimeStateLoadReceipt]:
    bundle = load_runtime_state_bundle(path)
    return bundle.state, RuntimeStateLoadReceipt(
        status="pass",
        source_dir=str(_safe_source(path)),
        schema_version=bundle.state.schema_version,
        verified_files=(RUNTIME_STATE_FILE, RUNTIME_STATE_MANIFEST_FILE),
        warnings=(
            RuntimeIssue.create(
                "runtime_state_resume_not_execution_proof",
                "restore validates metadata continuity, not execution equivalence",
            ),
        ),
    )


def evaluate_runtime_resume_compatibility(
    saved_state: RuntimeState,
    current_config: RuntimeConfig,
    current_inspection: RuntimeInspection | None = None,
) -> RuntimeResumeCompatibilityReport:
    """Compare only runtime intent and observed topology metadata where meaningful."""

    blockers: list[RuntimeIssue] = []
    warnings: list[RuntimeIssue] = [
        RuntimeIssue.create(
            "runtime_state_resume_not_execution_proof",
            "compatibility does not prove equivalent execution or training resumption",
        )
    ]
    if saved_state.backend_id and current_config.backend_id:
        if saved_state.backend_id != current_config.backend_id:
            blockers.append(
                RuntimeIssue.create(
                    "runtime_state_resume_incompatible",
                    "saved and requested backend IDs differ",
                    saved_backend_id=saved_state.backend_id,
                    current_backend_id=current_config.backend_id,
                )
            )
    for field in ("precision_policy", "placement_policy", "distributed_policy"):
        saved = getattr(saved_state.runtime_config, field)
        current = getattr(current_config, field)
        if saved != current:
            blockers.append(
                RuntimeIssue.create(
                    "runtime_state_resume_incompatible",
                    "saved and requested runtime policy differs",
                    field=field,
                    saved=saved,
                    current=current,
                )
            )
    if saved_state.root_seed != current_config.seed:
        blockers.append(
            RuntimeIssue.create(
                "runtime_state_resume_incompatible",
                "saved root seed differs from current runtime configuration",
                saved_root_seed=saved_state.root_seed,
                current_root_seed=current_config.seed,
            )
        )
    if current_inspection is not None:
        topology = saved_state.topology_summary
        for key, current in (
            ("platform", current_inspection.environment.platform),
            ("process_count", current_inspection.environment.process_count),
            (
                "global_device_count",
                current_inspection.device_inventory.global_device_count,
            ),
        ):
            saved = topology.get(key)
            if saved is not None and current is not None and saved != current:
                warnings.append(
                    RuntimeIssue.create(
                        "runtime_state_topology_changed",
                        "restored topology is historical metadata and differs locally",
                        field=key,
                        saved=saved,
                        current=current,
                    )
                )
    return RuntimeResumeCompatibilityReport(
        status="fail" if blockers else "pass",
        blockers=tuple(blockers),
        warnings=tuple(warnings),
    )


def run_runtime_state_smoke() -> RuntimeStateSmokeReceipt:
    """Execute P2.4 before and after a temporary runtime-only state round trip."""

    from radjax_student.runtime.smoke import run_single_device_cpu_smoke

    first = run_single_device_cpu_smoke()
    if not first.ok:
        return RuntimeStateSmokeReceipt(
            status="fail",
            schema_version=RUNTIME_STATE_SCHEMA_VERSION,
            global_step=None,
            seed_tree_verified=False,
            config_round_trip=False,
            digest_verified=False,
            topology_restored_as_metadata=False,
            continuation_execution_succeeds=False,
            blockers=first.blockers
            or (
                RuntimeIssue.create(
                    "runtime_state_load_failed",
                    "runtime-state smoke requires a passing CPU runtime smoke",
                ),
            ),
        )
    try:
        state = _state_from_cpu_smoke(first, global_step=3)
        with tempfile.TemporaryDirectory(prefix="radjax-runtime-state-") as temp_dir:
            receipt = save_runtime_state(state, Path(temp_dir) / "runtime_state")
            loaded, load_receipt = load_runtime_state_with_receipt(
                Path(temp_dir) / "runtime_state"
            )
            compatibility = evaluate_runtime_resume_compatibility(
                loaded,
                first.config,
                inspect_runtime_environment(),
            )
        continuation = run_single_device_cpu_smoke(first.config)
    except RuntimeContractError as exc:
        return RuntimeStateSmokeReceipt(
            status="fail",
            schema_version=RUNTIME_STATE_SCHEMA_VERSION,
            global_step=None,
            seed_tree_verified=False,
            config_round_trip=False,
            digest_verified=False,
            topology_restored_as_metadata=False,
            continuation_execution_succeeds=False,
            blockers=(exc.issue,),
        )
    except Exception as exc:
        return RuntimeStateSmokeReceipt(
            status="fail",
            schema_version=RUNTIME_STATE_SCHEMA_VERSION,
            global_step=None,
            seed_tree_verified=False,
            config_round_trip=False,
            digest_verified=False,
            topology_restored_as_metadata=False,
            continuation_execution_succeeds=False,
            blockers=(
                RuntimeIssue.create(
                    "runtime_state_internal_error",
                    "runtime-state smoke failed before a coherent receipt was produced",
                    exception_type=type(exc).__name__,
                ),
            ),
        )
    return RuntimeStateSmokeReceipt(
        status="pass" if compatibility.ok and continuation.ok else "fail",
        schema_version=loaded.schema_version,
        global_step=loaded.global_step,
        seed_tree_verified=loaded.runtime_keys
        == RuntimeKeys.from_seed(loaded.root_seed),
        config_round_trip=loaded.runtime_config == state.runtime_config,
        digest_verified=(
            receipt.hashes[RUNTIME_STATE_FILE]
            == _sha256(canonical_runtime_state_json(loaded.to_dict()))
            and bool(load_receipt.verified_files)
        ),
        topology_restored_as_metadata=loaded.topology_summary == state.topology_summary,
        continuation_execution_succeeds=continuation.ok,
        blockers=(
            compatibility.blockers if not compatibility.ok else continuation.blockers
        ),
        warnings=(*compatibility.warnings, *load_receipt.warnings),
    )


def _state_from_cpu_smoke(receipt: Any, *, global_step: int) -> RuntimeState:
    report = receipt.runtime_report
    environment = report.environment
    inventory = report.device_inventory
    device_kinds = tuple(
        sorted(
            {
                item.device_kind
                for item in inventory.devices
                if item.device_kind is not None
            }
        )
    )
    return RuntimeState(
        runtime_id=receipt.runtime_id,
        global_step=global_step,
        root_seed=receipt.config.seed,
        runtime_config=receipt.config,
        environment_summary=environment.to_dict(),
        topology_summary={
            "platform": environment.platform,
            "process_count": environment.process_count,
            "process_index": environment.process_index,
            "local_device_count": inventory.local_device_count,
            "global_device_count": inventory.global_device_count,
            "device_kinds": device_kinds,
            "topology_labels": json_value(inventory.topology_summary),
        },
        precision_policy=receipt.config.precision_policy,
        placement_policy=receipt.config.placement_policy,
        backend_id=receipt.backend_id,
    )


def _safe_destination(path: str | Path) -> Path:
    destination = Path(path)
    if not destination.name or ".." in destination.parts:
        raise RuntimeContractError(
            "runtime_state_path_unsafe",
            "runtime-state output directory must not contain path traversal",
        )
    if destination.exists() and destination.is_symlink():
        raise RuntimeContractError(
            "runtime_state_path_unsafe",
            "runtime-state output directory must not be a symbolic link",
        )
    return destination


def _safe_source(path: str | Path) -> Path:
    source = Path(path)
    if ".." in source.parts or source.is_symlink() or not source.is_dir():
        raise RuntimeContractError(
            "runtime_state_path_unsafe" if source.exists() else "runtime_state_missing",
            "runtime-state source directory is missing or unsafe",
        )
    return source


def _prepare_destination(destination: Path, *, overwrite: bool) -> None:
    if not destination.exists():
        return
    if not destination.is_dir() or destination.is_symlink():
        raise RuntimeContractError(
            "runtime_state_path_unsafe",
            "runtime-state output path must be a real directory",
        )
    contents = {item.name for item in destination.iterdir()}
    if not overwrite:
        raise RuntimeContractError(
            "runtime_state_exists",
            "runtime-state output directory already exists; pass overwrite=True",
            details={"output_dir": str(destination)},
        )
    if contents - set(RUNTIME_STATE_FILES):
        raise RuntimeContractError(
            "runtime_state_path_unsafe",
            "refusing to overwrite a directory containing unrelated files",
        )
    for item in destination.iterdir():
        if item.is_symlink() or not item.is_file():
            raise RuntimeContractError(
                "runtime_state_path_unsafe",
                "runtime-state output directory contains an unsafe entry",
            )
        item.unlink()


def _atomic_write(path: Path, data: bytes) -> None:
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _validate_manifest_integrity(manifest: RuntimeStateManifest) -> None:
    _require_schema(manifest.schema_version)
    expected = _sha256(canonical_runtime_state_json(manifest.base_dict()))
    if expected != manifest.integrity.manifest_digest:
        raise RuntimeContractError(
            "runtime_state_hash_mismatch",
            "runtime-state manifest digest does not match its declared contents",
        )


def _verify_state_file(manifest: RuntimeStateManifest, state_bytes: bytes) -> None:
    expected_size = manifest.sizes[RUNTIME_STATE_FILE]
    if len(state_bytes) != expected_size:
        raise RuntimeContractError(
            "runtime_state_size_mismatch",
            "runtime-state file size does not match its manifest",
            details={"expected": expected_size, "actual": len(state_bytes)},
        )
    actual_digest = _sha256(state_bytes)
    if actual_digest != manifest.hashes[RUNTIME_STATE_FILE]:
        raise RuntimeContractError(
            "runtime_state_hash_mismatch",
            "runtime-state file digest does not match its manifest",
        )
    if actual_digest != manifest.integrity.state_digest:
        raise RuntimeContractError(
            "runtime_state_hash_mismatch",
            "runtime-state integrity digest does not match the state file",
        )


def _validate_state_payload(payload: Mapping[str, Any]) -> None:
    _reject_unknown_fields(payload, _STATE_FIELDS, "runtime-state payload")
    schema_version = payload.get("schema_version")
    try:
        _require_schema(schema_version)
    except ValueError as exc:
        raise RuntimeContractError(
            "runtime_state_schema_unsupported",
            "runtime-state schema version is unsupported",
            details={"schema_version": schema_version},
        ) from exc
    if (
        payload.get("global_step") is None
        or isinstance(payload["global_step"], bool)
        or not isinstance(payload["global_step"], int)
        or payload["global_step"] < 0
    ):
        raise RuntimeContractError(
            "runtime_state_step_invalid",
            "runtime-state global_step must be a nonnegative integer",
        )
    _reject_forbidden_payload_keys(payload)


def _reject_forbidden_payload_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        forbidden = sorted(_FORBIDDEN_STATE_KEYS.intersection(value))
        if forbidden:
            raise RuntimeContractError(
                "runtime_state_manifest_invalid",
                "runtime state contains a forbidden non-runtime field",
                details={"forbidden_fields": forbidden},
            )
        for child in value.values():
            _reject_forbidden_payload_keys(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_forbidden_payload_keys(child)


def _state_model_error_code(exc: Exception) -> str:
    message = str(exc).lower()
    if "runtime key" in message or "runtime keys" in message or "root seed" in message:
        return "runtime_state_rng_invalid"
    if "global_step" in message or "global step" in message:
        return "runtime_state_step_invalid"
    if (
        "runtime_config" in message
        or "precision_policy" in message
        or "placement_policy" in message
    ):
        return "runtime_state_config_invalid"
    return "runtime_state_load_failed"


def _read_json(path: Path, code: str) -> Mapping[str, Any]:
    return _parse_json(_read_bytes(path, code), code)


def _read_bytes(path: Path, code: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise RuntimeContractError(
            code,
            "runtime-state file could not be read",
            details={"file": path.name, "exception_type": type(exc).__name__},
        ) from exc


def _parse_json(data: bytes, code: str) -> Mapping[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeContractError(
            code,
            "runtime-state JSON is malformed",
            details={"exception_type": type(exc).__name__},
        ) from exc
    return _mapping(value, "runtime-state JSON")


def _require_regular_file(path: Path, code: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise RuntimeContractError(
            code,
            "required runtime-state file is missing or unsafe",
            details={"file": path.name},
        )


def _require_schema(value: Any) -> None:
    if value != RUNTIME_STATE_SCHEMA_VERSION:
        raise ValueError("unsupported runtime-state schema version")


def _validate_internal_file_names(files: tuple[str, ...]) -> None:
    for file_name in files:
        candidate = Path(file_name)
        if (
            candidate.is_absolute()
            or ".." in candidate.parts
            or candidate.name != file_name
        ):
            raise ValueError("runtime-state manifest contains an unsafe internal path")


def _reject_unknown_fields(
    payload: Mapping[str, Any], allowed: frozenset[str] | set[str], name: str
) -> None:
    unknown = sorted(set(payload) - set(allowed))
    if unknown:
        raise RuntimeContractError(
            "runtime_state_manifest_invalid",
            f"{name} contains unknown fields",
            details={"unknown_fields": unknown},
        )


def _hash_mapping(value: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError("hashes must be a mapping")
    result = {str(key): _string(item, "hash") for key, item in value.items()}
    if any(not _is_digest(item) for item in result.values()):
        raise ValueError("hashes must contain SHA-256 hexadecimal digests")
    return result


def _size_mapping(value: Mapping[str, int]) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise TypeError("sizes must be a mapping")
    result = {str(key): item for key, item in value.items()}
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in result.values()
    ):
        raise ValueError("sizes must contain nonnegative integers")
    return result


def _string_mapping(value: Any, name: str) -> Mapping[str, str]:
    mapping = _mapping(value, name)
    return {str(key): _string(item, name) for key, item in mapping.items()}


def _integer_mapping(value: Any, name: str) -> Mapping[str, int]:
    mapping = _mapping(value, name)
    result = {str(key): item for key, item in mapping.items()}
    if any(
        isinstance(item, bool) or not isinstance(item, int) for item in result.values()
    ):
        raise TypeError(f"{name} must contain integers")
    return result


def _issues(value: Any, name: str) -> tuple[RuntimeIssue, ...]:
    result = tuple(value)
    if any(not isinstance(item, RuntimeIssue) for item in result):
        raise TypeError(f"{name} must contain RuntimeIssue values")
    return result


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _strings(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple")
    result = tuple(value)
    if any(not isinstance(item, str) or not item for item in result):
        raise ValueError(f"{name} must contain nonempty strings")
    return result


def _unique_strings(value: Any, name: str) -> tuple[str, ...]:
    result = _strings(value, name)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def _is_digest(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
