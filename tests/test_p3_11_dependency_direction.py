"""P3.11.6A dependency-direction regression checks."""

from __future__ import annotations

import ast
from pathlib import Path

from radjax_student.contracts import (
    LearningBatch as ContractLearningBatch,
)
from radjax_student.contracts import (
    MetricRecord as ContractMetricRecord,
)
from radjax_student.contracts import (
    ObjectiveScope as ContractObjectiveScope,
)
from radjax_student.contracts.errors import LearningContractError as ContractError
from radjax_student.learning import (
    LearningBatch as PublicLearningBatch,
)
from radjax_student.learning import (
    LearningContractError as PublicError,
)
from radjax_student.learning import (
    MetricRecord as PublicMetricRecord,
)
from radjax_student.learning import (
    ObjectiveScope as PublicObjectiveScope,
)

ROOT = Path("src/radjax_student")


def test_public_scope_reexports_preserve_exact_contract_identity():
    assert ContractObjectiveScope is PublicObjectiveScope
    assert ContractLearningBatch is PublicLearningBatch
    assert ContractMetricRecord is PublicMetricRecord
    assert ContractError is PublicError


def test_architecture_and_optimizer_have_no_direct_public_learning_imports():
    offenders = []
    for package in ("architecture", "optimizers"):
        for path in (ROOT / package).glob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.module == "radjax_student.learning"
                ):
                    offenders.append(str(path))
    assert offenders == []


def test_dependency_direction_has_no_architecture_optimizer_cycle():
    edges = {}
    for package in ("architecture", "optimizers"):
        imports = set()
        for path in (ROOT / package).glob("*.py"):
            tree = ast.parse(path.read_text())
            imports.update(
                node.module.split(".")[1]
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("radjax_student.")
                and len(node.module.split(".")) > 1
            )
        edges[package] = imports
    assert not (
        "optimizers" in edges["architecture"] and "architecture" in edges["optimizers"]
    )


def test_contract_package_has_no_sibling_package_dependencies():
    offenders = []
    for path in (ROOT / "contracts").glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith(
                    (
                        "radjax_student.learning",
                        "radjax_student.architecture",
                        "radjax_student.optimizers",
                        "radjax_student.runtime",
                        "radjax_student.hf",
                    )
                )
            ):
                offenders.append(str(path))
    assert offenders == []


def test_handwritten_jax_update_is_legacy_only():
    production = (ROOT / "learning" / "jax_core.py").read_text()
    legacy = (ROOT / "legacy" / "jax_learning.py").read_text()
    assert "def apply_scoped_gradient_update" not in production
    assert "def apply_scoped_gradient_update" in legacy
