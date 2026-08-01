"""Artifact inspection and loading helpers."""

from importlib import import_module

from radjax_student.artifacts.loaders import inspect_teacher_tome
from radjax_student.artifacts.models import (
    TomeArtifactError,
    TomeArtifactIdentity,
    TomeArtifactProvenance,
    TomeArtifactValidation,
    TomeArtifactView,
    TomeBehavioralSurface,
    TomeCorridorView,
    TomeExemplarView,
    TomeInferredDefaults,
    TomePayloadSummary,
)
from radjax_student.artifacts.view import open_tome_artifact

_NATIVE_V3_EXPORTS = frozenset(
    {
        "NATIVE_V3_STUDENT_PROFILE",
        "NativeV3ContractAssets",
        "NativeV3StudentConsumptionError",
        "NativeV3StudentConsumptionView",
        "load_native_v3_contract_assets",
        "open_native_v3_student_consumption",
    }
)


def __getattr__(name: str):
    """Keep the historical metadata-only import path Contract-version tolerant."""

    if name in _NATIVE_V3_EXPORTS:
        module = import_module("radjax_student.artifacts.native_v3")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "TomeArtifactError",
    "TomeArtifactIdentity",
    "TomeArtifactProvenance",
    "TomeArtifactValidation",
    "TomeArtifactView",
    "TomeBehavioralSurface",
    "TomeCorridorView",
    "TomeExemplarView",
    "TomeInferredDefaults",
    "TomePayloadSummary",
    "NATIVE_V3_STUDENT_PROFILE",
    "NativeV3ContractAssets",
    "NativeV3StudentConsumptionError",
    "NativeV3StudentConsumptionView",
    "inspect_teacher_tome",
    "load_native_v3_contract_assets",
    "open_native_v3_student_consumption",
    "open_tome_artifact",
]
