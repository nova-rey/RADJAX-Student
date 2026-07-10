from __future__ import annotations

import hashlib
import importlib
import json
import platform
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from types import MappingProxyType
from typing import Any

from radjax_student.artifacts import open_tome_artifact
from radjax_student.runtime import (
    PLACEMENT_INTENTS,
    CpuRuntimeSmokeReceipt,
    RuntimeBackendDescriptor,
    RuntimeConfig,
    RuntimeInspection,
    RuntimeSelectionResult,
    RuntimeStateSmokeReceipt,
    build_default_runtime_registry,
    inspect_runtime_environment,
    run_runtime_state_smoke,
    run_single_device_cpu_smoke,
    select_runtime_backend,
)
from radjax_student.validation import (
    evaluate_student_compatibility,
    infer_run_defaults,
    metadata_inspection_only_profile,
)
from radjax_student.validation.profile_registry import available_profile_ids

ACCEPTED_FIXTURE_DIGEST = (
    "468a259d518a28a6f60af8c339b124b65fd52da0640544d186eb9609933608d1"
)
DOCTOR_CLAIMS_NOT_MADE: tuple[str, ...] = (
    "payload_loading_not_tested",
    "training_not_available",
    "runtime_execution_not_available",
    "architecture_not_instantiated",
    "checkpoint_execution_not_tested",
    "hf_export_not_available",
    "model_quality_not_claimed",
)


@dataclass(frozen=True)
class PackageStatus:
    package: str
    version: str | None
    commit: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "package": self.package,
            "version": self.version,
            "commit": self.commit,
        }


@dataclass(frozen=True)
class StudentDoctorReport:
    status: str
    python_version: str
    student_package: PackageStatus
    contract_package: PackageStatus
    available_profiles: tuple[str, ...]
    contract_apis_importable: bool
    canonical_fixture_helper_available: bool
    canonical_fixture_available: bool
    expected_fixture_digest: str
    actual_fixture_digest: str | None
    fixture_digest_matches: bool
    fixture_opens: bool
    defaults_inference_succeeds: bool
    compatibility_report_succeeds: bool
    expected_metadata_failure_recognized: bool
    report_serialization_succeeds: bool
    runtime_inspection: RuntimeInspection
    runtime_backend_descriptors: tuple[RuntimeBackendDescriptor, ...]
    runtime_selection: RuntimeSelectionResult
    runtime_smoke: CpuRuntimeSmokeReceipt | None
    runtime_state_smoke: RuntimeStateSmokeReceipt | None
    placement_intent: Mapping[str, Any]
    execution_boundary: Mapping[str, str]
    capability_state: Mapping[str, str]
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    claims_not_made: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.status == "pass"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "python_version": self.python_version,
            "student_package": self.student_package.to_dict(),
            "contract_package": self.contract_package.to_dict(),
            "available_profiles": list(self.available_profiles),
            "contract_apis_importable": self.contract_apis_importable,
            "canonical_fixture_helper_available": (
                self.canonical_fixture_helper_available
            ),
            "canonical_fixture_available": self.canonical_fixture_available,
            "expected_fixture_digest": self.expected_fixture_digest,
            "actual_fixture_digest": self.actual_fixture_digest,
            "fixture_digest_matches": self.fixture_digest_matches,
            "fixture_opens": self.fixture_opens,
            "defaults_inference_succeeds": self.defaults_inference_succeeds,
            "compatibility_report_succeeds": self.compatibility_report_succeeds,
            "expected_metadata_failure_recognized": (
                self.expected_metadata_failure_recognized
            ),
            "report_serialization_succeeds": self.report_serialization_succeeds,
            "runtime_inspection": self.runtime_inspection.to_dict(),
            "runtime_backend_descriptors": [
                item.to_dict() for item in self.runtime_backend_descriptors
            ],
            "runtime_selection": self.runtime_selection.to_dict(),
            "runtime_smoke": (
                None if self.runtime_smoke is None else self.runtime_smoke.to_dict()
            ),
            "runtime_state_smoke": (
                None
                if self.runtime_state_smoke is None
                else self.runtime_state_smoke.to_dict()
            ),
            "placement_intent": dict(self.placement_intent),
            "execution_boundary": dict(self.execution_boundary),
            "capability_state": dict(self.capability_state),
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "claims_not_made": list(self.claims_not_made),
        }


def build_doctor_report(
    *,
    run_runtime_smoke: bool = False,
    run_runtime_state_smoke_check: bool = False,
) -> StudentDoctorReport:
    blockers: list[str] = []
    warnings: list[str] = []
    contract_importable = _contract_import_health()
    if not contract_importable:
        blockers.append("contract_production_api_import_failed")

    fixture_helper_available, fixture = _canonical_fixture()
    if not fixture_helper_available:
        blockers.append("canonical_fixture_helper_unavailable")
    fixture_available = fixture is not None and fixture.is_dir()
    actual_digest: str | None = None
    digest_matches = False
    if not fixture_available:
        blockers.append("canonical_fixture_missing")
    else:
        try:
            actual_digest = artifact_tree_digest(fixture)
            digest_matches = actual_digest == ACCEPTED_FIXTURE_DIGEST
            if not digest_matches:
                blockers.append("canonical_fixture_digest_mismatch")
        except (OSError, ValueError) as exc:
            blockers.append(f"canonical_fixture_digest_failed: {exc}")

    fixture_opens = False
    defaults_succeed = False
    compatibility_succeeds = False
    expected_failure = False
    serialization_succeeds = False
    if fixture_available and digest_matches:
        try:
            view = open_tome_artifact(fixture)
            fixture_opens = True
            defaults = infer_run_defaults(view)
            defaults_succeed = True
            compatibility = evaluate_student_compatibility(
                view,
                defaults,
                metadata_inspection_only_profile(),
            )
            compatibility_succeeds = True
            expected_failure = compatibility.status == "fail" and bool(
                compatibility.missing_capabilities
            )
            if not expected_failure:
                blockers.append("metadata_profile_expected_failure_not_recognized")
            json.dumps(compatibility.to_dict())
            serialization_succeeds = True
        except Exception as exc:  # Doctor must convert self-check failures to data.
            blockers.append(f"phase_1_self_check_failed: {type(exc).__name__}: {exc}")
    if expected_failure:
        warnings.append(
            "metadata_inspection_only compatibility failure is expected and honest"
        )
    runtime_inspection = inspect_runtime_environment()
    if not runtime_inspection.ok:
        blockers.append("runtime_inspection_failed")
    runtime_registry = build_default_runtime_registry()
    runtime_backend_descriptors = runtime_registry.describe(runtime_inspection)
    runtime_selection = select_runtime_backend(
        config=RuntimeConfig(),
        inspection=runtime_inspection,
        registry=runtime_registry,
    )
    runtime_smoke = run_single_device_cpu_smoke() if run_runtime_smoke else None
    if runtime_smoke is not None and not runtime_smoke.ok:
        blockers.append("runtime_smoke_failed")
    runtime_state_smoke = (
        run_runtime_state_smoke() if run_runtime_state_smoke_check else None
    )
    if runtime_state_smoke is not None and not runtime_state_smoke.ok:
        blockers.append("runtime_state_smoke_failed")
    return StudentDoctorReport(
        status="pass" if not blockers else "fail",
        python_version=platform.python_version(),
        student_package=_package_status("radjax-student"),
        contract_package=_package_status("radjax-contract"),
        available_profiles=available_profile_ids(),
        contract_apis_importable=contract_importable,
        canonical_fixture_helper_available=fixture_helper_available,
        canonical_fixture_available=fixture_available,
        expected_fixture_digest=ACCEPTED_FIXTURE_DIGEST,
        actual_fixture_digest=actual_digest,
        fixture_digest_matches=digest_matches,
        fixture_opens=fixture_opens,
        defaults_inference_succeeds=defaults_succeed,
        compatibility_report_succeeds=compatibility_succeeds,
        expected_metadata_failure_recognized=expected_failure,
        report_serialization_succeeds=serialization_succeeds,
        runtime_inspection=runtime_inspection,
        runtime_backend_descriptors=runtime_backend_descriptors,
        runtime_selection=runtime_selection,
        runtime_smoke=runtime_smoke,
        runtime_state_smoke=runtime_state_smoke,
        placement_intent=MappingProxyType(
            {
                "supported_declarations": PLACEMENT_INTENTS,
                "concrete_resolution": ("single_device_cpu_smoke_only",),
                "unresolved_declarations": (
                    "replicated",
                    "data_sharded",
                    "model_sharded",
                    "automatic",
                    "unspecified",
                ),
                "claims_not_made": (
                    "mesh_not_created",
                    "concrete_sharding_not_implemented",
                    "multi_device_placement_not_tested",
                ),
            }
        ),
        execution_boundary=MappingProxyType(
            {
                "eager": "available_on_explicit_request",
                "jit": "available_on_explicit_request_when_jax_available",
                "automatic": "resolves_to_eager_with_warning",
                "default_execution": "not_run",
            }
        ),
        capability_state=MappingProxyType(
            {
                "metadata_inspection": "available",
                "run_default_inference": "available",
                "compatibility_reporting": "available",
                "runtime_inspection": "available",
                "runtime_backend_registry": "available",
                "runtime_backend_selection": "available",
                "runtime_cpu_smoke": "available_on_explicit_request",
                "placement_intent": "available",
                "execution_boundary": "available",
                "runtime_state": "available_on_explicit_request",
                "payload_loading": "unavailable",
                "training": "unavailable",
                "jax_execution": "unavailable",
                "runtime_execution": "unavailable",
                "hf_export": "unavailable",
            }
        ),
        blockers=tuple(blockers),
        warnings=tuple(warnings),
        claims_not_made=DOCTOR_CLAIMS_NOT_MADE,
    )


def artifact_tree_digest(artifact_dir: str | Path) -> str:
    root = Path(artifact_dir)
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.is_symlink():
            raise ValueError(f"fixture contains a symbolic link: {path.name}")
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _contract_import_health() -> bool:
    try:
        module = importlib.import_module("radjax_contract.tome.production")
    except ImportError:
        return False
    return all(
        hasattr(module, name)
        for name in (
            "inspect_production_tome",
            "load_production_tome",
            "validate_production_tome",
        )
    )


def _canonical_fixture() -> tuple[bool, Path | None]:
    try:
        module = importlib.import_module("radjax_contract.testing")
        helper = module.production_tome_fixture_path
    except (AttributeError, ImportError):
        return False, None
    if not callable(helper):
        return False, None
    try:
        return True, Path(helper())
    except (OSError, TypeError, ValueError):
        return True, None


def _package_status(distribution_name: str) -> PackageStatus:
    try:
        distribution = metadata.distribution(distribution_name)
    except metadata.PackageNotFoundError:
        return PackageStatus(distribution_name, None, None)
    commit: str | None = None
    direct_url = distribution.read_text("direct_url.json")
    if direct_url:
        try:
            payload = json.loads(direct_url)
            commit = payload.get("vcs_info", {}).get("commit_id")
        except (AttributeError, json.JSONDecodeError):
            commit = None
    return PackageStatus(distribution_name, distribution.version, commit)
