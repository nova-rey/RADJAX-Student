from __future__ import annotations

from dataclasses import dataclass

import pytest

from radjax_student.architecture import (
    ArchitectureCapabilityProfile,
    ArchitectureContractError,
    ArchitectureRegistry,
    ForwardResult,
)
from radjax_student.architecture.testing import FakeArchitecturePlugin


class JaxOnly:
    def apply_jax(self, *args, **kwargs):
        del args, kwargs
        return ForwardResult()


@dataclass(frozen=True)
class FalseClaim(FakeArchitecturePlugin):
    def capability_profile(self):
        return ArchitectureCapabilityProfile(
            self.architecture_id,
            self.architecture_version,
            (
                *super().capability_profile().capabilities,
                "architecture.jax_execution_v1",
            ),
        )


@dataclass(frozen=True)
class UndeclaredJax(FakeArchitecturePlugin):
    def apply_jax(self, *args, **kwargs):
        del args, kwargs
        return ForwardResult()


@dataclass(frozen=True)
class FullJax(FakeArchitecturePlugin):
    def capability_profile(self):
        return ArchitectureCapabilityProfile(
            self.architecture_id,
            self.architecture_version,
            (
                *super().capability_profile().capabilities,
                "architecture.jax_execution_v1",
            ),
        )

    def apply_jax(self, *args, **kwargs):
        del args, kwargs
        return ForwardResult()


def test_registry_rejects_apply_jax_only_object():
    with pytest.raises(ArchitectureContractError, match="complete ArchitecturePlugin"):
        ArchitectureRegistry().register(JaxOnly())


def test_registry_rejects_false_jax_capability_claim():
    with pytest.raises(ArchitectureContractError, match="must agree"):
        ArchitectureRegistry().register(FalseClaim())


def test_registry_rejects_undeclared_jax_execution():
    with pytest.raises(ArchitectureContractError, match="must agree"):
        ArchitectureRegistry().register(UndeclaredJax())


def test_registry_accepts_full_jax_architecture_plugin():
    registry = ArchitectureRegistry()
    registry.register(FullJax())
    assert registry.get("test.architecture.v1") is not None
