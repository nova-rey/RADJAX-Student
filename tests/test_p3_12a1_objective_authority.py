"""P3.12A.1 closure for the deprecated split JAX-objective authority."""

from __future__ import annotations

from pathlib import Path

import pytest

from radjax_student.objectives import ObjectiveRegistry
from radjax_student.validation.architecture_audit import build_architecture_audit

ROOT = Path(__file__).resolve().parents[1]


def test_public_namespaces_do_not_export_legacy_objective_authority():
    import radjax_student
    import radjax_student.learning as learning

    legacy = {
        "JaxObjective",
        "JaxObjectiveConfig",
        "build_jax_loss_fn",
        "build_resolved_jax_loss_fn",
    }
    assert legacy.isdisjoint(radjax_student.__dict__)
    assert legacy.isdisjoint(learning.__dict__)
    assert legacy.isdisjoint(learning.__all__)


def test_architecture_audit_rejects_legacy_and_split_core_objective_authority(
    tmp_path: Path,
):
    source = tmp_path / "src" / "radjax_student" / "learning"
    source.mkdir(parents=True)
    (source.parent / "__init__.py").write_text("", encoding="utf-8")
    (source / "__init__.py").write_text("", encoding="utf-8")
    (source / "jax_core.py").write_text(
        "class JaxObjectiveConfig: pass\n"
        "def build_jax_loss_fn(objective, objective_id): pass\n"
        "__all__ = ['JaxObjectiveConfig', 'build_jax_loss_fn']\n",
        encoding="utf-8",
    )
    audit = build_architecture_audit(tmp_path)
    codes = {item["code"] for item in audit["blockers"]}
    assert {
        "legacy_objective_config_in_core",
        "legacy_objective_exported_from_core",
        "split_objective_authority_signature",
        "unregistered_objective_builder_in_core",
    }.issubset(codes)


@pytest.mark.jax
def test_jax_core_exports_only_the_registered_objective_builder():
    import radjax_student.learning.jax_core as core

    assert core.__all__ == [
        "JaxBatch",
        "JaxLossAuxiliary",
        "build_registered_jax_loss_fn",
        "build_value_and_grad_fn",
        "validate_finite_loss_and_gradients",
    ]
    for name in (
        "JaxObjective",
        "JaxObjectiveConfig",
        "build_jax_loss_fn",
        "build_resolved_jax_loss_fn",
    ):
        assert not hasattr(core, name)
        with pytest.raises(ImportError):
            exec(f"from radjax_student.learning.jax_core import {name}", {})


@pytest.mark.jax
def test_legacy_jax_objective_builder_is_explicitly_deprecated_and_not_selectable():
    from dataclasses import replace

    from radjax_student.legacy.objectives_jax import (
        LegacyJaxObjectiveConfig,
        build_legacy_jax_loss_fn,
    )
    from radjax_student.validation.p3_11_9_replay.runner_jax import _new_lifecycle

    with pytest.warns(DeprecationWarning):
        LegacyJaxObjectiveConfig("mse")
    with pytest.warns(DeprecationWarning), pytest.raises(TypeError):
        build_legacy_jax_loss_fn(object(), object())
    with pytest.raises(Exception) as error:
        ObjectiveRegistry().register(_EvaluateOnlyLegacyObjective())
    assert getattr(error.value, "code", None) == "objective_plugin_invalid"
    lifecycle = _new_lifecycle("eager", [])
    with pytest.raises(TypeError, match="ObjectiveRegistrySelection"):
        replace(lifecycle, objective_selection=_EvaluateOnlyLegacyObjective())


class _EvaluateOnlyLegacyObjective:
    def evaluate(self, *args, **kwargs):
        del args, kwargs
