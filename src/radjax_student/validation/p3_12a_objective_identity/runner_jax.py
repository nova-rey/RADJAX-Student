"""Executed P3.12A objective-identity proof over the accepted public conveyor."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from radjax_student.checkpoints import (
    load_learning_checkpoint_v3,
    save_learning_checkpoint_v3,
)
from radjax_student.contracts import (
    OBJECTIVE_CAPABILITY_SCHEMA_VERSION,
    ObjectiveCapabilityProfile,
    ObjectiveConfig,
    ObjectiveContractError,
    ObjectiveExecutionDescriptor,
    ObjectiveIdentity,
    ResolvedObjectiveSelection,
    objective_digest,
)
from radjax_student.learning.run_report import build_learning_run_report
from radjax_student.objectives import (
    CANONICAL_MSE_IDENTITY,
    ObjectiveRegistry,
    ObjectiveRegistrySelection,
    build_default_objective_registry,
)
from radjax_student.objectives.jax import MeanSquaredErrorObjective
from radjax_student.objectives.legacy import resolve_historical_objective_alias
from radjax_student.validation.architecture_audit import build_architecture_audit
from radjax_student.validation.p3_11_9_replay.models import (
    ReplayRunEvidence,
)
from radjax_student.validation.p3_11_9_replay.runner_jax import (
    _new_lifecycle,
    _run_arm,
    execute_stateful_replays,
)
from radjax_student.validation.p3_11_9_replay.verifier import verify_replay_proof
from radjax_student.validation.p3_12a_objective_identity.models import (
    ObjectiveIdentityProof,
    ObjectiveProofCase,
    digest,
)

NON_CLAIMS = (
    "no_production_architecture",
    "no_tome_payload_consumption",
    "no_distillation",
    "no_hugging_face_export",
    "no_accelerator_scale_training",
    "no_multi_device_proof",
    "no_model_quality_claim",
    "no_phase4_implementation",
)


def _code(error: BaseException) -> str:
    value = getattr(error, "code", None)
    return value if isinstance(value, str) else type(error).__name__


def _passed(
    case_id: str, boundary: str, evidence: dict[str, Any]
) -> ObjectiveProofCase:
    return ObjectiveProofCase(
        case_id,
        "positive",
        "pass",
        None,
        None,
        boundary,
        digest(evidence),
    )


def _rejected(
    case_id: str,
    boundary: str,
    expected_code: str,
    invoke: Callable[[], Any],
) -> ObjectiveProofCase:
    """Capture a public boundary's real exception identity.

    The expected code participates only after the exception has been observed.
    No observer receives an inventory object or expected case metadata.
    """

    try:
        invoke()
    except BaseException as error:
        observed = _code(error)
        if observed != expected_code:
            raise RuntimeError(
                f"P3.12A {case_id} emitted {observed}, expected {expected_code}"
            ) from error
        return ObjectiveProofCase(
            case_id,
            "adversarial",
            "reject",
            expected_code,
            observed,
            boundary,
            digest(
                {
                    "exception_type": type(error).__qualname__,
                    "observed_code": observed,
                    "message": str(error),
                }
            ),
        )
    raise RuntimeError(f"P3.12A adversary unexpectedly succeeded: {case_id}")


def _rejected_blocker(
    case_id: str,
    boundary: str,
    expected_code: str,
    invoke: Callable[[], Any],
) -> ObjectiveProofCase:
    """Capture an actual typed replay blocker without fabricating an exception."""

    result = invoke()
    blockers = tuple(getattr(result, "blockers", ()))
    if not blockers:
        raise RuntimeError(f"P3.12A adversary unexpectedly succeeded: {case_id}")
    observed = blockers[0].code
    if observed != expected_code:
        raise RuntimeError(
            f"P3.12A {case_id} emitted {observed}, expected {expected_code}"
        )
    return ObjectiveProofCase(
        case_id,
        "adversarial",
        "reject",
        expected_code,
        observed,
        boundary,
        digest(blockers[0].to_dict()),
    )


class _EvaluateOnlyObjective:
    def evaluate(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs


class _ApplyOnlyObjective:
    def apply_jax(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs


class _CompleteObjectiveWithoutJax:
    objective_id = CANONICAL_MSE_IDENTITY.objective_id
    objective_version = CANONICAL_MSE_IDENTITY.objective_version

    def objective_identity(self) -> ObjectiveIdentity:
        return CANONICAL_MSE_IDENTITY

    def capability_profile(self) -> ObjectiveCapabilityProfile:
        return _profile(CANONICAL_MSE_IDENTITY, ("objective.jax_execution_v1",))

    def validate_config(self, config: ObjectiveConfig) -> None:
        del config

    def validate_resolved_surface(self, selection: ResolvedObjectiveSelection) -> None:
        del selection

    def validate_targets(self, targets: Any) -> None:
        del targets

    def validate_metrics(self, metrics: Any) -> None:
        del metrics

    def execution_contract_version(self) -> str:
        return "objective.jax_execution.v1"


def _profile(
    identity: ObjectiveIdentity,
    capabilities: tuple[str, ...] = ("objective.jax_execution_v1",),
    *,
    metric_names: tuple[str, ...] = ("objective.mse",),
    schema: str = OBJECTIVE_CAPABILITY_SCHEMA_VERSION,
) -> ObjectiveCapabilityProfile:
    return ObjectiveCapabilityProfile(
        identity=identity,
        supported_execution_capabilities=capabilities,
        required_surface_roles=("prediction",),
        target_requirements=("targets.y",),
        metric_schema_id="radjax.objective.mean_squared_error.metrics.v1",
        metric_names=metric_names,
        capability_schema_version=schema,
    )


class _ProfileIdentityMismatchObjective(MeanSquaredErrorObjective):
    def capability_profile(self) -> ObjectiveCapabilityProfile:
        return _profile(ObjectiveIdentity("radjax.objective.foreign", "1"))


class _ProfileVersionMismatchObjective(MeanSquaredErrorObjective):
    def capability_profile(self) -> ObjectiveCapabilityProfile:
        return _profile(ObjectiveIdentity(CANONICAL_MSE_IDENTITY.objective_id, "2"))


class _UndeclaredJaxObjective(MeanSquaredErrorObjective):
    def capability_profile(self) -> ObjectiveCapabilityProfile:
        return _profile(CANONICAL_MSE_IDENTITY, ("objective.scalar_execution_v1",))


class _AlternativeMseObjective(MeanSquaredErrorObjective):
    """A separately implemented but deliberately same-shaped test objective."""


class _SecondAlternativeMseObjective(MeanSquaredErrorObjective):
    """Distinct class used to prove implementation identity cannot collide."""


class _RawParameterAssumingObjective(MeanSquaredErrorObjective):
    def evaluate_jax(
        self,
        *,
        surface: Any,
        targets: Any,
        weights: Any,
        config: ObjectiveConfig,
    ) -> tuple[Any, dict[str, Any]]:
        del targets, weights, config
        if not isinstance(surface, dict):
            raise ObjectiveContractError(
                "objective_plugin_invalid",
                "objective attempted to treat its declared surface as parameters",
            )
        raise AssertionError("raw-parameter objective unexpectedly received a mapping")


def _foreign_config(*, version: str = "1") -> ObjectiveConfig:
    return ObjectiveConfig(
        ObjectiveIdentity("radjax.objective.foreign", version), {"reduction": "mean"}
    )


def _descriptor_with(
    descriptor: ObjectiveExecutionDescriptor, **changes: Any
) -> ObjectiveExecutionDescriptor:
    return replace(descriptor, **changes)


def _checkpoint_loader(lifecycle, directory: Path, **changes: Any):
    values = {
        "optimizer": lifecycle.optimizer,
        "parameter_layout": lifecycle.parameter_layout,
        "runtime_reference": lifecycle.runtime_reference,
        "expected_hf_reference": lifecycle.hf_reference,
        "expected_architecture_config_digest": lifecycle.config_digest,
        "expected_parameter_catalog_digest": lifecycle.catalog_digest,
        "expected_architecture_state_id": lifecycle.architecture_state.state_id,
        "expected_architecture_carry_descriptor": (
            lifecycle.architecture_carry_descriptor
        ),
        "expected_objective_descriptor": lifecycle.objective_descriptor,
        "expected_objective_config": lifecycle.objective_config,
        "expected_resolved_objective_selection": lifecycle.resolved_objective_selection,
        "expected_objective_selection": lifecycle.objective_selection,
    }
    values.update(changes)
    return load_learning_checkpoint_v3(directory, **values)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _rewrite_manifest(directory: Path) -> None:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in manifest["files"]:
        data = (directory / name).read_bytes()
        manifest["hashes"][name] = hashlib.sha256(data).hexdigest()
        manifest["sizes"][name] = len(data)
    unsigned = {key: value for key, value in manifest.items() if key != "integrity"}
    manifest["integrity"] = {
        "algorithm": "sha256",
        "manifest_digest": hashlib.sha256(_json_bytes(unsigned)).hexdigest(),
    }
    manifest_path.write_bytes(_json_bytes(manifest))


def _objective_checkpoint_mutation(
    source: Path,
    destination: Path,
    mutate: Callable[[dict[str, Any], dict[str, Any]], None],
) -> Path:
    shutil.copytree(source, destination)
    learning_path = destination / "learning.json"
    manifest_path = destination / "manifest.json"
    learning = json.loads(learning_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(learning, manifest)
    learning_path.write_bytes(_json_bytes(learning))
    # The objective manifest mirrors exactly the semantic block after a
    # deliberate rehash; this reaches identity validation instead of hash I/O.
    descriptor = learning["objective_descriptor"]
    manifest["objective"]["descriptor"] = descriptor
    manifest["objective"]["descriptor_digest"] = objective_digest(descriptor)
    manifest["objective"]["config_digest"] = descriptor["config_digest"]
    manifest["objective"]["resolved_surface_identity"] = descriptor[
        "resolved_surface_identity"
    ]
    manifest["objective"]["registry_selection"] = learning[
        "objective_registry_selection"
    ]
    manifest_path.write_bytes(_json_bytes(manifest))
    _rewrite_manifest(destination)
    return destination


def _mutated_replay_objective(proof):
    replay = proof.modes["eager"]["replay_b"]
    arm = replay.uninterrupted
    step = arm.steps[0]
    objective = replace(step.objective, objective_id="radjax.objective.foreign")
    changed_step = replace(step, objective=objective)
    changed_arm = replace(arm, steps=(changed_step, *arm.steps[1:]))
    changed_run = ReplayRunEvidence(changed_arm, replay.resumed)
    modes = {mode: dict(replays) for mode, replays in proof.modes.items()}
    modes["eager"]["replay_b"] = changed_run
    return replace(proof, modes=modes)


def execute_objective_identity_proof(
    root: Path | None = None,
) -> ObjectiveIdentityProof:
    """Execute real identity checks from registry through replay and restore."""

    directory = Path(root) if root is not None else Path(tempfile.mkdtemp())
    directory.mkdir(parents=True, exist_ok=True)
    objects: list[object] = []
    lifecycle = _new_lifecycle("eager", objects)
    selection = lifecycle.objective_selection
    descriptor = lifecycle.objective_descriptor
    registry = build_default_objective_registry()
    if selection.identity != CANONICAL_MSE_IDENTITY:
        raise RuntimeError("stateful conveyor did not select canonical MSE objective")
    checkpoint_directory = directory / "checkpoint"
    saved = save_learning_checkpoint_v3(
        lifecycle.checkpoint(), checkpoint_directory, optimizer=lifecycle.optimizer
    )
    restored = _new_lifecycle("eager", []).restore_from_checkpoint(checkpoint_directory)
    replay = execute_stateful_replays(directory / "replay")
    replay_arm = replay.modes["eager"]["replay_a"].uninterrupted
    report_arm = _run_arm("eager", "uninterrupted", directory / "report", [])
    audit = build_architecture_audit(Path.cwd())
    if audit["status"] != "pass":
        raise RuntimeError("dependency audit blocked P3.12A proof")

    positives = (
        _passed(
            "registered_plugin_selection",
            "objective_registry",
            {"selection": selection.to_dict(), "descriptor": descriptor.to_dict()},
        ),
        _passed(
            "descriptor_binds_config_and_architecture_surface",
            "objective_registry.execution_descriptor",
            {
                "config_digest": lifecycle.objective_config.digest,
                "surface_digest": lifecycle.resolved_objective_selection.digest,
                "descriptor": descriptor.to_dict(),
            },
        ),
        _passed(
            "checkpoint_caller_bound_restore",
            "checkpoint_v3_restore",
            {
                "saved_descriptor": saved.objective_descriptor.to_dict(),
                "restored_descriptor": restored.objective_descriptor.to_dict(),
            },
        ),
        _passed(
            "replay_records_canonical_objective",
            "p3_11_9_replay",
            replay_arm.steps[0].objective.to_dict(),
        ),
        _passed(
            "report_preserves_executed_objective_descriptor",
            "learning_run_report",
            report_arm.report.objective.to_dict(),
        ),
        _passed(
            "legacy_alias_resolves_to_registry_selection",
            "objective_compatibility_adapter",
            resolve_historical_objective_alias(
                source_alias="mse",
                registry=registry,
                resolved_selection=lifecycle.resolved_objective_selection,
            ).descriptor.to_dict(),
        ),
    )

    jnp = __import__("jax.numpy", fromlist=["array"])
    plugin = selection.plugin
    foreign_surface = replace(
        lifecycle.resolved_objective_selection,
        surface_id="undeclared_surface",
    )
    foreign_surface_descriptor = registry.execution_descriptor(
        selection=selection,
        config=lifecycle.objective_config,
        resolved_selection=foreign_surface,
    )
    alternative_registry = ObjectiveRegistry()
    alternative_selection = alternative_registry.register(_AlternativeMseObjective())
    alternative_descriptor = alternative_registry.execution_descriptor(
        selection=alternative_selection,
        config=lifecycle.objective_config,
        resolved_selection=lifecycle.resolved_objective_selection,
    )

    def checkpoint_case(
        name: str, mutate: Callable[[dict[str, Any], dict[str, Any]], None]
    ):
        target = directory / name
        return _objective_checkpoint_mutation(checkpoint_directory, target, mutate)

    adversarial = (
        _rejected(
            "evaluate_only_object",
            "objective_registry",
            "objective_plugin_invalid",
            lambda: ObjectiveRegistry().register(_EvaluateOnlyObjective()),
        ),
        _rejected(
            "apply_jax_only_object",
            "objective_registry",
            "objective_plugin_invalid",
            lambda: ObjectiveRegistry().register(_ApplyOnlyObjective()),
        ),
        _rejected(
            "plugin_profile_id_mismatch",
            "objective_registry",
            "objective_identity_mismatch",
            lambda: ObjectiveRegistry().register(_ProfileIdentityMismatchObjective()),
        ),
        _rejected(
            "plugin_profile_version_mismatch",
            "objective_registry",
            "objective_identity_mismatch",
            lambda: ObjectiveRegistry().register(_ProfileVersionMismatchObjective()),
        ),
        _rejected(
            "jax_declared_without_implementation",
            "objective_registry",
            "objective_capability_missing",
            lambda: ObjectiveRegistry().register(_CompleteObjectiveWithoutJax()),
        ),
        _rejected(
            "jax_implemented_without_declaration",
            "objective_registry",
            "objective_capability_missing",
            lambda: ObjectiveRegistry().register(_UndeclaredJaxObjective()),
        ),
        _rejected(
            "config_objective_id_mismatch",
            "objective_registry.execution_descriptor",
            "objective_config_identity_mismatch",
            lambda: registry.execution_descriptor(
                selection=selection,
                config=_foreign_config(),
                resolved_selection=lifecycle.resolved_objective_selection,
            ),
        ),
        _rejected(
            "config_objective_version_mismatch",
            "objective_registry.execution_descriptor",
            "objective_config_identity_mismatch",
            lambda: registry.execution_descriptor(
                selection=selection,
                config=ObjectiveConfig(
                    ObjectiveIdentity(CANONICAL_MSE_IDENTITY.objective_id, "2"),
                    {"reduction": "mean"},
                ),
                resolved_selection=lifecycle.resolved_objective_selection,
            ),
        ),
        _rejected(
            "config_digest_mismatch",
            "jax_lifecycle",
            "objective_config_identity_mismatch",
            lambda: replace(
                lifecycle,
                objective_descriptor=_descriptor_with(
                    descriptor, config_digest="0" * 64
                ),
            ),
        ),
        _rejected(
            "unsupported_capability_schema",
            "objective_contract",
            "objective_capability_mismatch",
            lambda: _profile(
                CANONICAL_MSE_IDENTITY, schema="objective.capability.legacy"
            ),
        ),
        _rejected(
            "missing_required_architecture_surface",
            "objective_plugin_surface_validation",
            "objective_surface_identity_mismatch",
            lambda: plugin.validate_resolved_surface(
                replace(lifecycle.resolved_objective_selection, surface_role="hidden")
            ),
        ),
        _rejected(
            "undeclared_architecture_surface",
            "jax_lifecycle",
            "objective_surface_identity_mismatch",
            lambda: replace(
                lifecycle,
                resolved_objective_selection=foreign_surface,
                objective_descriptor=foreign_surface_descriptor,
            ),
        ),
        _rejected(
            "resolved_surface_identity_drift",
            "jax_lifecycle",
            "objective_config_identity_mismatch",
            lambda: replace(
                lifecycle,
                objective_descriptor=_descriptor_with(
                    descriptor, resolved_surface_identity="1" * 64
                ),
            ),
        ),
        _rejected(
            "missing_required_target",
            "objective_target_validation",
            "objective_target_invalid",
            lambda: plugin.validate_targets({}),
        ),
        _rejected(
            "malformed_target_shape",
            "objective_execution",
            "objective_target_invalid",
            lambda: plugin.evaluate_jax(
                surface=jnp.ones((1,)),
                targets={"y": jnp.ones((2,))},
                weights=None,
                config=lifecycle.objective_config,
            ),
        ),
        _rejected(
            "nonfinite_target",
            "objective_target_validation",
            "objective_target_invalid",
            lambda: plugin.validate_targets({"y": jnp.asarray([float("nan")])}),
        ),
        _rejected(
            "objective_consumes_raw_parameters",
            "objective_execution",
            "objective_plugin_invalid",
            lambda: _RawParameterAssumingObjective().evaluate_jax(
                surface=jnp.asarray([1.0]),
                targets={"y": jnp.asarray([1.0])},
                weights=None,
                config=lifecycle.objective_config,
            ),
        ),
        _rejected(
            "unknown_objective_metric",
            "objective_metric_validation",
            "objective_metric_invalid",
            lambda: plugin.validate_metrics(
                {"objective.mse": 1.0, "objective.extra": 1.0}
            ),
        ),
        _rejected(
            "duplicate_objective_metric",
            "objective_contract",
            "objective_metric_invalid",
            lambda: _profile(
                CANONICAL_MSE_IDENTITY, metric_names=("objective.mse", "objective.mse")
            ),
        ),
        _rejected(
            "nonfinite_objective_metric",
            "objective_metric_validation",
            "objective_metric_invalid",
            lambda: plugin.validate_metrics({"objective.mse": float("nan")}),
        ),
        _rejected(
            "checkpoint_objective_id_tampering",
            "checkpoint_v3_restore",
            "checkpoint_objective_identity_mismatch",
            lambda: _checkpoint_loader(
                lifecycle,
                checkpoint_case(
                    "checkpoint-objective-id",
                    lambda learning, manifest: (
                        learning["objective_descriptor"]["identity"].__setitem__(
                            "objective_id", "radjax.objective.foreign"
                        ),
                        learning["objective_registry_selection"].__setitem__(
                            "objective_id", "radjax.objective.foreign"
                        ),
                    ),
                ),
            ),
        ),
        _rejected(
            "checkpoint_objective_version_tampering",
            "checkpoint_v3_restore",
            "checkpoint_objective_identity_mismatch",
            lambda: _checkpoint_loader(
                lifecycle,
                checkpoint_case(
                    "checkpoint-objective-version",
                    lambda learning, manifest: (
                        learning["objective_descriptor"]["identity"].__setitem__(
                            "objective_version", "2"
                        ),
                        learning["objective_registry_selection"].__setitem__(
                            "objective_version", "2"
                        ),
                    ),
                ),
            ),
        ),
        _rejected(
            "checkpoint_profile_tampering",
            "checkpoint_v3_restore",
            "checkpoint_objective_identity_mismatch",
            lambda: _checkpoint_loader(
                lifecycle,
                checkpoint_case(
                    "checkpoint-profile",
                    lambda learning, manifest: (
                        learning["objective_descriptor"].__setitem__(
                            "capability_profile_digest", "2" * 64
                        ),
                        learning["objective_registry_selection"].__setitem__(
                            "capability_profile_digest", "2" * 64
                        ),
                    ),
                ),
            ),
        ),
        _rejected(
            "checkpoint_config_digest_tampering",
            "checkpoint_v3_restore",
            "checkpoint_objective_identity_mismatch",
            lambda: _checkpoint_loader(
                lifecycle,
                checkpoint_case(
                    "checkpoint-config",
                    lambda learning, manifest: learning[
                        "objective_descriptor"
                    ].__setitem__("config_digest", "3" * 64),
                ),
            ),
        ),
        _rejected(
            "checkpoint_resolved_surface_tampering",
            "checkpoint_v3_restore",
            "checkpoint_objective_identity_mismatch",
            lambda: _checkpoint_loader(
                lifecycle,
                checkpoint_case(
                    "checkpoint-surface",
                    lambda learning, manifest: (
                        learning["resolved_objective_selection"].__setitem__(
                            "surface_id", "foreign_surface"
                        ),
                        learning["objective_descriptor"].__setitem__(
                            "resolved_surface_identity", "4" * 64
                        ),
                    ),
                ),
            ),
        ),
        _rejected(
            "checkpoint_metric_schema_tampering",
            "checkpoint_v3_restore",
            "checkpoint_objective_identity_mismatch",
            lambda: _checkpoint_loader(
                lifecycle,
                checkpoint_case(
                    "checkpoint-metric",
                    lambda learning, manifest: learning[
                        "objective_descriptor"
                    ].__setitem__(
                        "metric_schema_id", "radjax.objective.foreign.metrics.v1"
                    ),
                ),
            ),
        ),
        _rejected(
            "checkpoint_implementation_identity_tampering",
            "checkpoint_v3_restore",
            "checkpoint_objective_identity_mismatch",
            lambda: _checkpoint_loader(
                lifecycle,
                checkpoint_case(
                    "checkpoint-implementation",
                    lambda learning, manifest: (
                        learning["objective_descriptor"].__setitem__(
                            "implementation_identity",
                            "radjax.objective_impl.impl_foreign",
                        ),
                        learning["objective_registry_selection"].__setitem__(
                            "implementation_identity",
                            "radjax.objective_impl.impl_foreign",
                        ),
                    ),
                ),
            ),
        ),
        _rejected(
            "restore_foreign_objective_plugin",
            "checkpoint_v3_restore",
            "checkpoint_objective_identity_mismatch",
            lambda: _checkpoint_loader(
                lifecycle,
                checkpoint_directory,
                expected_objective_descriptor=alternative_descriptor,
                expected_objective_selection=alternative_selection,
            ),
        ),
        _rejected(
            "restore_missing_expected_objective",
            "checkpoint_v3_restore",
            "checkpoint_objective_identity_missing",
            lambda: load_learning_checkpoint_v3(
                checkpoint_directory,
                optimizer=lifecycle.optimizer,
                parameter_layout=lifecycle.parameter_layout,
            ),
        ),
        _rejected(
            "resumed_execution_different_objective",
            "jax_lifecycle",
            "objective_config_identity_mismatch",
            lambda: replace(lifecycle, objective_selection=alternative_selection),
        ),
        _rejected_blocker(
            "replay_objective_identity_drift",
            "replay_verifier",
            "replay_objective_identity_mismatch",
            lambda: verify_replay_proof(_mutated_replay_objective(replay)),
        ),
        _rejected(
            "report_claims_different_objective",
            "learning_run_report",
            "ValueError",
            lambda: build_learning_run_report(
                loop_result=report_arm.loop_result,
                run_id="p312a",
                update_scope="whole_student",
                objective_scope="final_output",
                objective_descriptor=alternative_descriptor,
            ),
        ),
        _rejected(
            "registry_duplicate_objective_identity",
            "objective_registry",
            "objective_plugin_invalid",
            lambda: (
                lambda duplicate: (
                    duplicate.register(MeanSquaredErrorObjective()),
                    duplicate.register(MeanSquaredErrorObjective()),
                )
            )(ObjectiveRegistry()),
        ),
        _rejected(
            "registry_incompatible_implementation_same_identity",
            "objective_registry",
            "objective_implementation_identity_mismatch",
            lambda: (
                lambda duplicate: (
                    duplicate.register(_AlternativeMseObjective()),
                    duplicate.register(_SecondAlternativeMseObjective()),
                )
            )(ObjectiveRegistry()),
        ),
        _rejected(
            "objective_supplied_without_registry_selection",
            "jax_lifecycle",
            "objective_identity_mismatch",
            lambda: replace(
                lifecycle,
                objective_selection=ObjectiveRegistrySelection(
                    selection.identity,
                    selection.profile,
                    selection.implementation_identity,
                    selection.registry_identity,
                    selection.plugin,
                ),
            ),
        ),
        _rejected(
            "legacy_adapter_bypass_complete_plugin",
            "objective_compatibility_adapter",
            "objective_identity_mismatch",
            lambda: resolve_historical_objective_alias(
                source_alias="foreign.mse",
                registry=registry,
                resolved_selection=lifecycle.resolved_objective_selection,
            ),
        ),
        _rejected(
            "historical_alias_cannot_supply_implementation",
            "objective_compatibility_adapter",
            "TypeError",
            lambda: resolve_historical_objective_alias(
                source_alias="mse",
                registry=registry,
                resolved_selection=lifecycle.resolved_objective_selection,
                implementation=MeanSquaredErrorObjective(),
            ),
        ),
    )
    return ObjectiveIdentityProof(
        descriptor=descriptor,
        positive_cases=positives,
        adversarial_cases=adversarial,
        checkpoint_objective_identity_digest=digest(
            saved.objective_descriptor.to_dict()
        ),
        replay_objective_evidence_digest=replay_arm.steps[0].objective.digest,
        report_objective_evidence_digest=digest(report_arm.report.objective.to_dict()),
        dependency_audit_digest=digest(audit),
        non_claims=NON_CLAIMS,
    )


__all__ = ["NON_CLAIMS", "execute_objective_identity_proof"]
