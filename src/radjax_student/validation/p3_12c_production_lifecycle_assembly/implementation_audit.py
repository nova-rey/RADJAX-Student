"""JAX-free AST/source audit for P3.12C assembly authority.

The audit intentionally reads source files instead of importing production
assembly code.  It is therefore safe to use from the base suite and checks
that the receipt path cannot turn expected metadata or a parallel construction
recipe into observed execution evidence.
"""

from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .inventory import ADVERSARIAL_CASE_IDS, POSITIVE_CASE_IDS

SCHEMA_VERSION = "radjax.p3_12c_one_authority_audit.v1"
_CANONICAL_ASSEMBLER = "assemble_jax_learning_lifecycle"
_ASSEMBLY_MODULE = "radjax_student.learning.assembly"
_PRODUCTION_OWNERS = (
    "architecture",
    "contracts",
    "checkpoints",
    "learning",
    "runtime",
    "objectives",
    "optimizers",
    "steps",
    "reports",
    "hf",
)
_EXPECTED_METADATA = {
    "case",
    "case_id",
    "case_name",
    "expected",
    "expected_code",
    "expected_outcome",
    "outcome",
    "category",
    "spec",
    "intended_boundary",
    "canonical_result",
}
_SUCCESS_NAMES = {"passed", "accepted", "success", "valid", "deterministic"}
_LOWER_LEVEL_VALIDATION_EXCEPTIONS = {
    "src/radjax_student/validation/p3_11_10_gate/implementations/section_a_contracts.py",
}


def _digest(value: object) -> str:
    return hashlib.sha256(
        (
            json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode()
    ).hexdigest()


def _sha(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or set(value) - set("0123456789abcdef")
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class AssemblyAuditBlocker:
    """One stable source-audit blocker with no runtime result semantics."""

    code: str
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code:
            raise ValueError("assembly audit blocker code must be nonempty")
        if not isinstance(self.detail, str) or not self.detail:
            raise ValueError("assembly audit blocker detail must be nonempty")

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "detail": self.detail}

    @classmethod
    def from_dict(cls, payload: object) -> AssemblyAuditBlocker:
        if not isinstance(payload, Mapping) or set(payload) != {"code", "detail"}:
            raise ValueError("assembly audit blocker fields are invalid")
        return cls(payload["code"], payload["detail"])


@dataclass(frozen=True)
class AssemblyAuditSourceEntry:
    """A deterministic source segment participating in audit evidence."""

    path: str
    source_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("assembly audit source path must be nonempty")
        _sha(self.source_digest, "assembly audit source digest")

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "source_digest": self.source_digest}

    @classmethod
    def from_dict(cls, payload: object) -> AssemblyAuditSourceEntry:
        if not isinstance(payload, Mapping) or set(payload) != {
            "path",
            "source_digest",
        }:
            raise ValueError("assembly audit source entry fields are invalid")
        return cls(payload["path"], payload["source_digest"])


class AssemblyAuditFailure(Exception):
    """Actual public rejection used by source-adversary invocations."""

    def __init__(self, blocker: AssemblyAuditBlocker) -> None:
        self.code = blocker.code
        self.blocker = blocker
        super().__init__(f"{blocker.code}: {blocker.detail}")


@dataclass(frozen=True)
class AssemblyAuthorityAudit:
    """Typed result whose status derives only from source blockers."""

    source_evidence_digest: str
    source_entries: tuple[AssemblyAuditSourceEntry, ...]
    positive_case_ids: tuple[str, ...]
    adversarial_case_ids: tuple[str, ...]
    blockers: tuple[AssemblyAuditBlocker, ...]

    def __post_init__(self) -> None:
        _sha(self.source_evidence_digest, "assembly audit source evidence digest")
        entries = tuple(self.source_entries)
        positives = tuple(self.positive_case_ids)
        adversarial = tuple(self.adversarial_case_ids)
        blockers = tuple(self.blockers)
        if not all(isinstance(item, AssemblyAuditSourceEntry) for item in entries):
            raise TypeError("assembly audit source entries must be typed")
        if not all(isinstance(item, str) and item for item in positives + adversarial):
            raise ValueError("assembly audit inventory IDs must be nonempty strings")
        if not all(isinstance(item, AssemblyAuditBlocker) for item in blockers):
            raise TypeError("assembly audit blockers must be typed")
        object.__setattr__(self, "source_entries", entries)
        object.__setattr__(self, "positive_case_ids", positives)
        object.__setattr__(self, "adversarial_case_ids", adversarial)
        object.__setattr__(self, "blockers", blockers)

    @property
    def status(self) -> str:
        return "pass" if not self.blockers else "blocked"

    @property
    def positive_inventory_count(self) -> int:
        return len(self.positive_case_ids)

    @property
    def adversarial_inventory_count(self) -> int:
        return len(self.adversarial_case_ids)

    @property
    def implementation_audit_digest(self) -> str:
        return _digest(
            {
                "source_evidence_digest": self.source_evidence_digest,
                "source_entries": [item.to_dict() for item in self.source_entries],
                "positive_case_ids": list(self.positive_case_ids),
                "adversarial_case_ids": list(self.adversarial_case_ids),
                "blockers": [item.to_dict() for item in self.blockers],
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": self.status,
            "blockers": [item.to_dict() for item in self.blockers],
            "positive_inventory_count": self.positive_inventory_count,
            "adversarial_inventory_count": self.adversarial_inventory_count,
            "positive_case_ids": list(self.positive_case_ids),
            "adversarial_case_ids": list(self.adversarial_case_ids),
            "source_entries": [item.to_dict() for item in self.source_entries],
            "source_evidence_digest": self.source_evidence_digest,
            "implementation_audit_digest": self.implementation_audit_digest,
        }

    @classmethod
    def from_dict(cls, payload: object) -> AssemblyAuthorityAudit:
        required = {
            "schema_version",
            "status",
            "blockers",
            "positive_inventory_count",
            "adversarial_inventory_count",
            "positive_case_ids",
            "adversarial_case_ids",
            "source_entries",
            "source_evidence_digest",
            "implementation_audit_digest",
        }
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise ValueError("assembly audit fields are missing or unknown")
        if payload["schema_version"] != SCHEMA_VERSION:
            raise ValueError("assembly audit schema is invalid")
        try:
            audit = cls(
                payload["source_evidence_digest"],
                tuple(
                    AssemblyAuditSourceEntry.from_dict(item)
                    for item in payload["source_entries"]
                ),
                tuple(payload["positive_case_ids"]),
                tuple(payload["adversarial_case_ids"]),
                tuple(
                    AssemblyAuditBlocker.from_dict(item) for item in payload["blockers"]
                ),
            )
        except (KeyError, TypeError) as exc:
            raise ValueError("assembly audit payload is malformed") from exc
        if payload["status"] != audit.status:
            raise ValueError("assembly audit status is invalid")
        if (
            payload["positive_inventory_count"] != audit.positive_inventory_count
            or payload["adversarial_inventory_count"]
            != audit.adversarial_inventory_count
        ):
            raise ValueError("assembly audit inventory count is invalid")
        if payload["implementation_audit_digest"] != audit.implementation_audit_digest:
            raise ValueError("assembly audit digest is invalid")
        return audit


def _add(blockers: list[AssemblyAuditBlocker], code: str, detail: str) -> None:
    item = AssemblyAuditBlocker(code, detail)
    if item not in blockers:
        blockers.append(item)


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def _imports(tree: ast.AST) -> tuple[str, ...]:
    values: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            values.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            values.append(module)
            values.extend(f"{module}.{alias.name}" for alias in node.names)
    return tuple(values)


def _functions(tree: ast.AST) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, ...]:
    return tuple(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )


def _calls(function: ast.AST) -> set[str]:
    return {
        name
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        if (name := _call_name(node.func)) is not None
    }


def _literal_tuple(tree: ast.Module, name: str) -> tuple[str, ...] | None:
    candidates: list[ast.AST] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(
                isinstance(target, ast.Name) and target.id == name
                for target in node.targets
            ):
                candidates.append(node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == name:
                candidates.append(node.value)
    if len(candidates) != 1 or not isinstance(candidates[0], (ast.Tuple, ast.List)):
        return None
    values = candidates[0].elts
    if not all(
        isinstance(item, ast.Constant) and isinstance(item.value, str)
        for item in values
    ):
        return None
    return tuple(item.value for item in values)


def _contains_constructor_call(tree: ast.AST, name: str) -> bool:
    return any(
        isinstance(node, ast.Call)
        and (_call_name(node.func) or "").split(".")[-1] == name
        for node in ast.walk(tree)
    )


def _annotation_names(tree: ast.AST) -> set[str]:
    return {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }


def _class_annotation_names(tree: ast.AST, class_name: str) -> set[str]:
    target = next(
        (
            node
            for node in getattr(tree, "body", ())
            if isinstance(node, ast.ClassDef) and node.name == class_name
        ),
        None,
    )
    return set() if target is None else _annotation_names(target)


def _source_entries(root: Path) -> tuple[AssemblyAuditSourceEntry, ...]:
    source_root = root / "src" / "radjax_student"
    relevant: list[Path] = []
    for owner in _PRODUCTION_OWNERS:
        directory = source_root / owner
        if directory.is_dir():
            relevant.extend(sorted(directory.rglob("*.py")))
    validation = source_root / "validation" / "p3_12c_production_lifecycle_assembly"
    if validation.is_dir():
        relevant.extend(sorted(validation.glob("*.py")))
    return tuple(
        AssemblyAuditSourceEntry(
            str(path.relative_to(root)), hashlib.sha256(path.read_bytes()).hexdigest()
        )
        for path in sorted(set(relevant))
    )


def _audit_inventory(
    inventory_path: Path, blockers: list[AssemblyAuditBlocker]
) -> None:
    if not inventory_path.is_file():
        _add(blockers, "assembly_audit_inventory_missing", "inventory.py")
        return
    tree = ast.parse(
        inventory_path.read_text(encoding="utf-8"), filename=str(inventory_path)
    )
    positives = _literal_tuple(tree, "POSITIVE_CASE_IDS")
    adversarial = _literal_tuple(tree, "ADVERSARIAL_CASE_IDS")
    if positives != POSITIVE_CASE_IDS:
        _add(blockers, "assembly_audit_positive_inventory", "POSITIVE_CASE_IDS")
    if adversarial != ADVERSARIAL_CASE_IDS:
        _add(blockers, "assembly_audit_adversarial_inventory", "ADVERSARIAL_CASE_IDS")


def _audit_assembly_source(path: Path, blockers: list[AssemblyAuditBlocker]) -> None:
    if not path.is_file():
        _add(blockers, "assembly_audit_assembler_missing", str(path))
        return
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    if any(
        item == "radjax_student.validation"
        or item.startswith("radjax_student.validation.")
        for item in _imports(tree)
    ):
        _add(blockers, "assembly_audit_assembler_imports_validation", str(path))
    public = [
        function.name
        for function in tree.body
        if isinstance(function, ast.FunctionDef)
        and function.name.startswith("assemble_")
    ]
    if public != [_CANONICAL_ASSEMBLER]:
        _add(blockers, "assembly_audit_public_assembler", repr(public))
    forbidden_fields = {
        "JaxArchitecturePlugin",
        "ObjectivePlugin",
        "ObjectiveRegistrySelection",
        "JaxOptimizerBackend",
        "ExecutionBackend",
        "ExecutionContext",
        "RuntimeKeyStream",
        "JaxLearningLifecycle",
        "JaxLoopExecutor",
        "HFCompatibilityDescriptor",
    }
    request = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "JaxLearningAssemblyRequest"
        ),
        None,
    )
    if request is None:
        _add(blockers, "assembly_audit_request_missing", "JaxLearningAssemblyRequest")
    else:
        names = _annotation_names(request)
        if names & forbidden_fields or "Callable" in names:
            _add(
                blockers,
                "assembly_audit_request_executable_injection",
                ",".join(
                    sorted(
                        names & forbidden_fields
                        | ({"Callable"} if "Callable" in names else set())
                    )
                ),
            )
    result = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "JaxLearningAssemblyResult"
        ),
        None,
    )
    if result is None:
        _add(blockers, "assembly_audit_result_missing", "JaxLearningAssemblyResult")
    elif _annotation_names(result) & _SUCCESS_NAMES:
        _add(blockers, "assembly_audit_result_caller_success", "result fields")
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            if node.value.id in {"parameters", "optimizer_state"}:
                _add(
                    blockers,
                    "assembly_audit_assembler_leaf_inspection",
                    node.value.id,
                )
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "optimizer_state" and node.attr in {
                "arrays",
                "envelope",
            }:
                _add(
                    blockers,
                    "assembly_audit_assembler_leaf_inspection",
                    "optimizer_state." + node.attr,
                )
        if isinstance(node, ast.Call):
            name = _call_name(node.func) or ""
            if name in {"jax.devices", "jax.device_put"}:
                _add(blockers, "assembly_audit_raw_device_selection", name)
            if name == "HFCompatibilityDescriptor":
                _add(blockers, "assembly_audit_assembler_hf_fabrication", name)
    if any(name in source for name in ("case_id", "expected_code", "expected_outcome")):
        _add(blockers, "assembly_audit_validation_metadata_in_production", "metadata")


def _audit_production_dependencies(
    root: Path, blockers: list[AssemblyAuditBlocker]
) -> None:
    source_root = root / "src" / "radjax_student"
    for owner in _PRODUCTION_OWNERS:
        directory = source_root / owner
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imports = _imports(tree)
            relative = str(path.relative_to(root))
            if any(
                "p3_12c_production_lifecycle_assembly" in item
                or item.endswith(".assembly")
                and item.startswith("radjax_student.validation")
                for item in imports
            ):
                _add(blockers, "assembly_audit_production_imports_validation", relative)
            if owner != "learning" and any(
                item == _ASSEMBLY_MODULE
                or item.startswith(_ASSEMBLY_MODULE + ".")
                or item.endswith("." + _CANONICAL_ASSEMBLER)
                for item in imports
            ):
                _add(blockers, "assembly_audit_owner_imports_assembler", relative)
            for function in _functions(tree):
                calls = _calls(function)
                if {
                    "JaxLearningLifecycle",
                    "JaxLoopExecutor",
                } <= {name.split(".")[-1] for name in calls}:
                    if path.name != "assembly.py":
                        _add(
                            blockers,
                            "assembly_audit_competing_production_assembler",
                            f"{relative}:{function.name}",
                        )


def _audit_validation_migration(
    root: Path, blockers: list[AssemblyAuditBlocker]
) -> None:
    validation_root = root / "src" / "radjax_student" / "validation"
    if not validation_root.is_dir():
        return
    direct_runner = validation_root / "p3_11_9_replay/runner_jax.py"
    if direct_runner.is_file() and _CANONICAL_ASSEMBLER not in direct_runner.read_text(
        encoding="utf-8"
    ):
        _add(
            blockers,
            "assembly_audit_validation_migration_missing",
            "p3_11_9_replay/runner_jax.py",
        )
    for relative in (
        "p3_12a_objective_identity/runner_jax.py",
        "p3_12b_hf_descriptor_authority/runner_jax.py",
    ):
        path = validation_root / relative
        if path.is_file() and "_new_lifecycle" not in path.read_text(encoding="utf-8"):
            _add(blockers, "assembly_audit_validation_migration_missing", relative)
    for path in sorted(validation_root.rglob("*.py")):
        relative = str(path.relative_to(root))
        if relative in _LOWER_LEVEL_VALIDATION_EXCEPTIONS:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if _contains_constructor_call(tree, "JaxLearningLifecycle"):
            _add(blockers, "assembly_audit_manual_lifecycle", relative)
        if _contains_constructor_call(tree, "JaxLoopExecutor"):
            _add(blockers, "assembly_audit_manual_executor", relative)


def _audit_receipt_and_observer(
    root: Path, blockers: list[AssemblyAuditBlocker]
) -> None:
    package = (
        root / "src/radjax_student/validation/p3_12c_production_lifecycle_assembly"
    )
    models = package / "models.py"
    diagnostic = package / "diagnostic.py"
    runner = package / "runner_jax.py"
    experiments = package / "experiments.py"
    required = (models, diagnostic, runner, experiments)
    if any(not path.is_file() for path in required):
        _add(
            blockers,
            "assembly_audit_evidence_path_missing",
            "runner/models/diagnostic/experiments",
        )
        return
    model_source = models.read_text(encoding="utf-8")
    runner_source = runner.read_text(encoding="utf-8")
    if any(
        item not in model_source
        for item in (
            "LifecycleAssemblyProof",
            "ADVERSARIAL_CASE_IDS",
            "POSITIVE_CASE_IDS",
        )
    ):
        _add(blockers, "assembly_audit_receipt_typed_proof", "models.py")
    model_tree = ast.parse(model_source, filename=str(models))
    for function in _functions(model_tree):
        if function.name != "build_receipt":
            continue
        names = {
            arg.arg
            for arg in (
                *function.args.posonlyargs,
                *function.args.args,
                *function.args.kwonlyargs,
            )
        }
        if names & _SUCCESS_NAMES:
            _add(blockers, "assembly_audit_receipt_success_flag", "build_receipt")
    observer_tree = ast.parse(
        diagnostic.read_text(encoding="utf-8"), filename=str(diagnostic)
    )
    observer = next(
        (
            item
            for item in observer_tree.body
            if isinstance(item, ast.FunctionDef) and item.name == "observe"
        ),
        None,
    )
    if observer is None:
        _add(blockers, "assembly_audit_observer_missing", "observe")
    else:
        parameters = {
            arg.arg
            for arg in (
                *observer.args.posonlyargs,
                *observer.args.args,
                *observer.args.kwonlyargs,
            )
        }
        if parameters != {"invocation"} or observer.args.vararg or observer.args.kwarg:
            _add(
                blockers,
                "assembly_audit_observer_expected_metadata",
                "observe parameters",
            )
        if any(
            isinstance(node, ast.Name) and node.id in _EXPECTED_METADATA
            for node in ast.walk(observer)
        ):
            _add(blockers, "assembly_audit_observer_expected_metadata", "observe body")
        if not any(
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "invocation"
            and node.attr == "callable"
            for node in ast.walk(observer)
        ):
            _add(blockers, "assembly_audit_observed_boundary_not_callable", "observe")
    invocation = next(
        (
            item
            for item in observer_tree.body
            if isinstance(item, ast.ClassDef) and item.name == "Invocation"
        ),
        None,
    )
    if invocation is None:
        _add(blockers, "assembly_audit_invocation_missing", "Invocation")
    else:
        fields = {
            item.target.id
            for item in invocation.body
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
        }
        if "callable" not in fields or "boundary" in fields:
            _add(blockers, "assembly_audit_free_boundary", "Invocation")
    for source_name, source in (
        ("runner_jax.py", runner_source),
        ("models.py", model_source),
    ):
        if "_matches_expected" in source or "matches_expected" in source:
            _add(blockers, "assembly_audit_permissive_matcher", source_name)
        tree = ast.parse(source, filename=source_name)
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "startswith"
            for node in ast.walk(tree)
        ):
            _add(blockers, "assembly_audit_permissive_matcher", source_name)
    if not all(
        item in runner_source for item in ("ADVERSARIAL_CASE_IDS", "POSITIVE_CASE_IDS")
    ):
        _add(blockers, "assembly_audit_runner_inventory_authority", "runner_jax.py")
    runner_tree = ast.parse(runner_source, filename=str(runner))
    experiments_mapping = next(
        (
            node.value
            for node in runner_tree.body
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "_EXPERIMENTS"
        ),
        None,
    )
    if not isinstance(experiments_mapping, ast.Dict):
        _add(blockers, "assembly_audit_adversarial_mapping", "_EXPERIMENTS")
    else:
        keys = tuple(
            item.value
            for item in experiments_mapping.keys
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        )
        values = tuple(
            _call_name(item) if item is not None else None
            for item in experiments_mapping.values
        )
        expected_values = tuple(
            f"experiments.experiment_{case_id}" for case_id in ADVERSARIAL_CASE_IDS
        )
        if keys != ADVERSARIAL_CASE_IDS or values != expected_values:
            _add(blockers, "assembly_audit_adversarial_mapping", "_EXPERIMENTS")
        if len(set(values)) != len(values):
            _add(
                blockers, "assembly_audit_reused_adversarial_experiment", "_EXPERIMENTS"
            )
        if any(isinstance(item, ast.Lambda) for item in experiments_mapping.values):
            _add(blockers, "assembly_audit_lambda_adversary", "_EXPERIMENTS")
    experiments_source = experiments.read_text(encoding="utf-8")
    experiments_tree = ast.parse(experiments_source, filename=str(experiments))
    functions = tuple(
        node
        for node in experiments_tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("experiment_")
    )
    names = tuple(node.name.removeprefix("experiment_") for node in functions)
    if names != ADVERSARIAL_CASE_IDS:
        _add(blockers, "assembly_audit_experiment_inventory", "experiments.py")
    if any(isinstance(node, ast.Lambda) for node in ast.walk(experiments_tree)):
        _add(blockers, "assembly_audit_lambda_adversary", "experiments.py")
    if any(
        isinstance(node, ast.For)
        and any(isinstance(item, ast.FunctionDef) for item in node.body)
        for node in ast.walk(experiments_tree)
    ):
        _add(blockers, "assembly_audit_dynamic_adversary", "experiments.py")
    if any(
        isinstance(node, ast.Call)
        and (_call_name(node.func) or "").split(".")[-1]
        in {"partial", "run_case", "generic_adversary"}
        for node in ast.walk(experiments_tree)
    ):
        _add(blockers, "assembly_audit_generic_adversary", "experiments.py")


def audit_assembly_authority(root: Path | None = None) -> AssemblyAuthorityAudit:
    """Audit the real repository without importing any JAX-bearing module."""

    root = Path.cwd() if root is None else Path(root)
    package = (
        root / "src/radjax_student/validation/p3_12c_production_lifecycle_assembly"
    )
    blockers: list[AssemblyAuditBlocker] = []
    _audit_inventory(package / "inventory.py", blockers)
    _audit_assembly_source(root / "src/radjax_student/learning/assembly.py", blockers)
    _audit_production_dependencies(root, blockers)
    _audit_validation_migration(root, blockers)
    _audit_receipt_and_observer(root, blockers)
    entries = _source_entries(root)
    evidence = _digest([item.to_dict() for item in entries])
    return AssemblyAuthorityAudit(
        evidence,
        entries,
        POSITIVE_CASE_IDS,
        ADVERSARIAL_CASE_IDS,
        tuple(sorted(blockers, key=lambda item: (item.code, item.detail))),
    )


def audit_synthetic_source(
    source: str, *, path: str = "src/radjax_student/learning/assembly.py"
) -> tuple[AssemblyAuditBlocker, ...]:
    """Execute source/AST checks against one focused JAX-free bad fixture."""

    if not isinstance(source, str):
        raise TypeError("synthetic source must be text")
    if not isinstance(path, str) or not path:
        raise TypeError("synthetic source path must be nonempty")
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as exc:
        return (AssemblyAuditBlocker("assembly_audit_fixture_syntax", str(exc)),)
    blockers: list[AssemblyAuditBlocker] = []
    functions = _functions(tree)
    assembler_functions = [
        item for item in functions if item.name.startswith("assemble_")
    ]
    if "validation" in Path(path).parts and assembler_functions:
        _add(blockers, "assembly_audit_fixture_assembler_under_validation", path)
    elif len(assembler_functions) > 1 or (
        assembler_functions and assembler_functions[0].name != _CANONICAL_ASSEMBLER
    ):
        _add(blockers, "assembly_audit_fixture_second_public_assembler", path)
    imports = _imports(tree)
    if any(item.startswith("radjax_student.validation") for item in imports):
        _add(blockers, "assembly_audit_fixture_production_validation_import", path)
    competing = any(
        {"JaxLearningLifecycle", "JaxLoopExecutor"}
        <= {item.split(".")[-1] for item in _calls(function)}
        and "learning" not in Path(path).parts
        for function in functions
    )
    if competing:
        _add(blockers, "assembly_audit_fixture_competing_factory", path)
    elif _contains_constructor_call(tree, "JaxLearningLifecycle"):
        _add(blockers, "assembly_audit_fixture_manual_lifecycle", path)
    elif _contains_constructor_call(tree, "JaxLoopExecutor"):
        _add(blockers, "assembly_audit_fixture_manual_executor", path)
    names = _class_annotation_names(tree, "JaxLearningAssemblyRequest")
    field_codes = {
        "JaxArchitecturePlugin": "assembly_audit_fixture_request_architecture_plugin",
        "ObjectivePlugin": "assembly_audit_fixture_request_objective_plugin",
        "JaxOptimizerBackend": "assembly_audit_fixture_request_optimizer_backend",
        "ExecutionBackend": "assembly_audit_fixture_request_runtime_backend",
        "Callable": "assembly_audit_fixture_request_raw_loss_callable",
        "HFCompatibilityDescriptor": "assembly_audit_fixture_request_hf_descriptor",
    }
    for name, code in field_codes.items():
        if name in names:
            _add(blockers, code, name)
    if "HFCompatibilityDescriptor(" in source:
        _add(
            blockers,
            "assembly_audit_fixture_hf_fabrication",
            "HFCompatibilityDescriptor",
        )
    if "parameters[" in source:
        _add(blockers, "assembly_audit_fixture_parameter_leaves", "parameters")
    if "optimizer_state.arrays" in source or "optimizer_state[" in source:
        _add(blockers, "assembly_audit_fixture_optimizer_leaves", "optimizer_state")
    if "jax.devices(" in source or "jax.device_put(" in source:
        _add(blockers, "assembly_audit_fixture_raw_device", "jax device")
    replacement_codes = {
        "lifecycle.architecture =": "assembly_audit_fixture_lifecycle_replacement",
        "loop.architecture =": "assembly_audit_fixture_loop_architecture",
        "loop.optimizer =": "assembly_audit_fixture_loop_optimizer",
        "loop.objective =": "assembly_audit_fixture_loop_objective",
        "runtime_key_stream =": "assembly_audit_fixture_runtime_key_injection",
        "resolved_surface =": "assembly_audit_fixture_surface_fabrication",
    }
    for token, code in replacement_codes.items():
        if token in source:
            _add(blockers, code, token)
    if "passed=True" in source:
        _add(blockers, "assembly_audit_fixture_caller_passed", "passed")
    observer = next((item for item in functions if item.name == "observe"), None)
    if observer is not None:
        parameters = {
            arg.arg
            for arg in (
                *observer.args.posonlyargs,
                *observer.args.args,
                *observer.args.kwonlyargs,
            )
        }
        if parameters & _EXPECTED_METADATA:
            _add(blockers, "assembly_audit_fixture_observer_expected_code", "observe")
    if "_matches_expected" in source:
        _add(blockers, "assembly_audit_fixture_permissive_matcher", "_matches_expected")
    if 'startswith("assembly_")' in source or "startswith('assembly_')" in source:
        _add(blockers, "assembly_audit_fixture_prefix_matcher", "assembly prefix")
    positives = _literal_tuple(tree, "POSITIVE_CASE_IDS")
    if positives is not None and positives != POSITIVE_CASE_IDS:
        _add(blockers, "assembly_audit_fixture_positive_inventory", "POSITIVE_CASE_IDS")
    adversarial = _literal_tuple(tree, "ADVERSARIAL_CASE_IDS")
    if adversarial is not None and adversarial != ADVERSARIAL_CASE_IDS:
        _add(
            blockers,
            "assembly_audit_fixture_adversarial_inventory",
            "ADVERSARIAL_CASE_IDS",
        )
    for function in functions:
        if function.name == "build_receipt":
            parameters = {
                arg.arg
                for arg in (
                    *function.args.posonlyargs,
                    *function.args.args,
                    *function.args.kwonlyargs,
                )
            }
            if parameters & _SUCCESS_NAMES or "success=" in source:
                _add(
                    blockers,
                    "assembly_audit_fixture_receipt_success_flag",
                    "build_receipt",
                )
    for function in assembler_functions:
        if any(
            isinstance(node, ast.Name) and node.id == "case_id"
            for node in ast.walk(function)
        ):
            _add(blockers, "assembly_audit_fixture_validation_case_id", function.name)
    return tuple(sorted(blockers, key=lambda item: (item.code, item.detail)))


def require_clean_synthetic_source(source: str) -> None:
    blockers = audit_synthetic_source(source)
    if blockers:
        raise AssemblyAuditFailure(blockers[0])


def require_clean_assembly_authority(audit: AssemblyAuthorityAudit) -> None:
    if not isinstance(audit, AssemblyAuthorityAudit):
        raise TypeError("assembly authority audit must be typed")
    if audit.status != "pass":
        raise AssemblyAuditFailure(audit.blockers[0])


__all__ = [
    "AssemblyAuditBlocker",
    "AssemblyAuditFailure",
    "AssemblyAuditSourceEntry",
    "AssemblyAuthorityAudit",
    "SCHEMA_VERSION",
    "audit_assembly_authority",
    "audit_synthetic_source",
    "require_clean_assembly_authority",
    "require_clean_synthetic_source",
]
