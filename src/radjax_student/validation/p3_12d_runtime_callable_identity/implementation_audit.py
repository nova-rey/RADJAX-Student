"""JAX-free AST/source audit for P3.12D runtime callable identity."""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .inventory import ADVERSARIAL_CASE_IDS, POSITIVE_CASE_IDS

SCHEMA_VERSION = "radjax.p3_12d_callable_identity_audit.v1"


class CallableAuditError(ValueError):
    """Stable source-audit rejection exposed at the audit boundary."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _digest(value: object) -> str:
    return hashlib.sha256(
        (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()


@dataclass(frozen=True)
class CallableAuditBlocker:
    code: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "detail": self.detail}


@dataclass(frozen=True)
class CallableAuditSourceEntry:
    path: str
    source_digest: str

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "source_digest": self.source_digest}


@dataclass(frozen=True)
class RuntimeCallableIdentityAudit:
    source_entries: tuple[CallableAuditSourceEntry, ...]
    blockers: tuple[CallableAuditBlocker, ...]

    @property
    def status(self) -> str:
        return "pass" if not self.blockers else "blocked"

    @property
    def source_evidence_digest(self) -> str:
        return _digest([item.to_dict() for item in self.source_entries])

    @property
    def implementation_audit_digest(self) -> str:
        return _digest(
            {
                "source_evidence_digest": self.source_evidence_digest,
                "positive_case_ids": list(POSITIVE_CASE_IDS),
                "adversarial_case_ids": list(ADVERSARIAL_CASE_IDS),
                "blockers": [item.to_dict() for item in self.blockers],
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": self.status,
            "blockers": [item.to_dict() for item in self.blockers],
            "positive_inventory_count": len(POSITIVE_CASE_IDS),
            "adversarial_inventory_count": len(ADVERSARIAL_CASE_IDS),
            "positive_case_ids": list(POSITIVE_CASE_IDS),
            "adversarial_case_ids": list(ADVERSARIAL_CASE_IDS),
            "source_entries": [item.to_dict() for item in self.source_entries],
            "source_evidence_digest": self.source_evidence_digest,
            "implementation_audit_digest": self.implementation_audit_digest,
        }


def audit_runtime_callable_identity(
    root: Path | None = None,
) -> RuntimeCallableIdentityAudit:
    """Audit the authoritative sources without importing JAX-bearing modules."""
    repository = Path.cwd() if root is None else root
    source_root = repository / "src/radjax_student"
    production_sources = tuple(
        path
        for path in sorted(source_root.rglob("*.py"))
        if "validation" not in path.parts and "__pycache__" not in path.parts
    )
    validation_sources = (
        repository
        / "src/radjax_student/validation/p3_12d_runtime_callable_identity/inventory.py",
        repository
        / "src/radjax_student/validation/p3_12d_runtime_callable_identity"
        / "implementation_audit.py",
        repository
        / "src/radjax_student/validation/p3_12d_runtime_callable_identity"
        / "experiments.py",
        repository
        / "src/radjax_student/validation/p3_12d_runtime_callable_identity"
        / "runner_jax.py",
        repository
        / "src/radjax_student/validation/p3_12d_runtime_callable_identity"
        / "diagnostic.py",
    )
    sources = (*production_sources, *validation_sources)
    entries: list[CallableAuditSourceEntry] = []
    blockers: list[CallableAuditBlocker] = []
    for path in sources:
        text = path.read_text()
        entries.append(
            CallableAuditSourceEntry(
                str(path.relative_to(repository)),
                hashlib.sha256(text.encode()).hexdigest(),
            )
        )
        tree = ast.parse(text)
        _inspect_tree(tree, str(path.relative_to(repository)), blockers)
    callable_path = repository / "src/radjax_student/runtime/callables.py"
    callable_source = callable_path.read_text()
    if callable_source.count("def bind_runtime_callable(") != 1:
        blockers.append(
            CallableAuditBlocker(
                "callable_audit_competing_authority",
                "runtime binding authority is not unique",
            )
        )
    if "inspect.getsource" not in callable_source or "ast.parse" not in callable_source:
        blockers.append(
            CallableAuditBlocker(
                "callable_audit_source_not_actual",
                "callable source identity is not AST-derived",
            )
        )
    if len(POSITIVE_CASE_IDS) != 18 or len(ADVERSARIAL_CASE_IDS) != 40:
        blockers.append(
            CallableAuditBlocker(
                "callable_audit_inventory_invalid", "P3.12D inventory is not exact"
            )
        )
    _audit_single_runtime_authority(production_sources, repository, blockers)
    _audit_literal_experiments(repository, blockers)
    return RuntimeCallableIdentityAudit(tuple(entries), tuple(blockers))


def _audit_single_runtime_authority(
    sources: tuple[Path, ...], repository: Path, blockers: list[CallableAuditBlocker]
) -> None:
    """Reject parallel production source-identity derivation outside runtime."""
    binder_definitions = 0
    identity_constructors: list[str] = []
    for path in sources:
        tree = ast.parse(path.read_text())
        relative = str(path.relative_to(repository))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.FunctionDef)
                and node.name == "bind_runtime_callable"
            ):
                binder_definitions += 1
                if relative != "src/radjax_student/runtime/callables.py":
                    blockers.append(
                        CallableAuditBlocker(
                            "callable_audit_competing_authority", relative
                        )
                    )
            if (
                isinstance(node, ast.Call)
                and _call_name(node) == "RuntimeCallableIdentity"
            ):
                identity_constructors.append(relative)
            if (
                _contains_validation_d_reference(node)
                and "/validation/" not in relative
            ):
                blockers.append(
                    CallableAuditBlocker(
                        "callable_audit_production_validation_import", relative
                    )
                )
    if binder_definitions != 1 or set(identity_constructors) != {
        "src/radjax_student/runtime/callables.py"
    }:
        blockers.append(
            CallableAuditBlocker(
                "callable_audit_competing_authority",
                "runtime must be the only callable identity authority",
            )
        )


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _contains_validation_d_reference(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return "p3_12d_runtime_callable_identity" in node.value
    return False


def _audit_literal_experiments(
    repository: Path, blockers: list[CallableAuditBlocker]
) -> None:
    """Require one named, no-control-parameter experiment per frozen case."""
    experiment_path = (
        repository
        / "src/radjax_student/validation/p3_12d_runtime_callable_identity"
        / "experiments.py"
    )
    runner_path = experiment_path.with_name("runner_jax.py")
    experiment_tree = ast.parse(experiment_path.read_text())
    functions = {
        node.name: node
        for node in experiment_tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("experiment_")
    }
    expected = {f"experiment_{case_id}" for case_id in ADVERSARIAL_CASE_IDS}
    if set(functions) != expected:
        blockers.append(
            CallableAuditBlocker(
                "callable_audit_literal_experiment_inventory", "experiments.py"
            )
        )
    prohibited = {
        "case",
        "case_id",
        "expected_code",
        "expected_outcome",
        "category",
        "spec",
        "intended_boundary",
    }
    for name, function in functions.items():
        arguments = (*function.args.posonlyargs, *function.args.args)
        if (
            function.args.vararg is not None
            or function.args.kwarg is not None
            or function.args.kwonlyargs
            or any(argument.arg in prohibited for argument in arguments)
            or not any(isinstance(node, ast.Return) for node in ast.walk(function))
        ):
            blockers.append(
                CallableAuditBlocker("callable_audit_experiment_signature", name)
            )
    runner_tree = ast.parse(runner_path.read_text())
    mapping_keys: list[str] = []
    mapping_values: list[str] = []
    for node in ast.walk(runner_tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    if key.value in ADVERSARIAL_CASE_IDS:
                        mapping_keys.append(key.value)
                        mapping_values.append(ast.dump(value, include_attributes=False))
    if tuple(mapping_keys) != ADVERSARIAL_CASE_IDS or len(set(mapping_values)) != len(
        mapping_values
    ):
        blockers.append(
            CallableAuditBlocker(
                "callable_audit_literal_experiment_mapping", "runner_jax.py"
            )
        )
    for function in (
        node for node in runner_tree.body if isinstance(node, ast.FunctionDef)
    ):
        if function.name in {"observe", "_matches_expected", "matches_expected"}:
            blockers.append(
                CallableAuditBlocker("callable_audit_permissive_matcher", function.name)
            )


def _inspect_tree(
    tree: ast.AST, path: str, blockers: list[CallableAuditBlocker]
) -> None:
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and "p3_12d_runtime_callable_identity" in node.module
        ):
            if "/validation/" not in path:
                blockers.append(
                    CallableAuditBlocker(
                        "callable_audit_production_validation_import", path
                    )
                )
        if (
            path.endswith("runtime/callables.py")
            and isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
        ):
            if node.func.id in {"id", "repr", "hash"}:
                blockers.append(
                    CallableAuditBlocker(
                        "callable_audit_unsafe_identity", f"{path}:{node.func.id}"
                    )
                )
        if isinstance(node, ast.FunctionDef) and node.name in {
            "_matches_expected",
            "matches_expected",
        }:
            blockers.append(
                CallableAuditBlocker("callable_audit_permissive_matcher", path)
            )
        if (
            not path.endswith("implementation_audit.py")
            and isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "startswith"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and node.args[0].value.startswith("execution_callable_")
        ):
            blockers.append(
                CallableAuditBlocker("callable_audit_permissive_matcher", path)
            )


def require_clean_runtime_callable_identity_audit(root: Path | None = None) -> None:
    audit = audit_runtime_callable_identity(root)
    if audit.blockers:
        raise ValueError(audit.blockers[0].code)


def audit_synthetic_source(
    source: str, *, path: str = "fixture.py"
) -> tuple[CallableAuditBlocker, ...]:
    """Execute the same narrow source rules against one principal-defect fixture."""
    tree = ast.parse(source)
    blockers: list[CallableAuditBlocker] = []
    _inspect_tree(tree, path, blockers)
    rules = (
        ("id(function)", "callable_audit_unsafe_identity"),
        ("repr(function)", "callable_audit_unsafe_identity"),
        ("hash(function)", "callable_audit_unsafe_identity"),
        ("def bind_runtime_callable_two", "callable_audit_competing_authority"),
        ("validation.assembly", "callable_audit_validation_binder"),
        (
            "from radjax_student.validation",
            "callable_audit_production_validation_import",
        ),
        ("raw_callable:", "callable_audit_request_raw_callable"),
        ("function:", "callable_audit_request_function"),
        ("caller_identity_digest", "callable_audit_caller_digest"),
        ("caller_source_digest", "callable_audit_caller_source_digest"),
        (
            "module_qualname_only_identity",
            "callable_audit_module_qualname_only_identity",
        ),
        ("module_only_identity", "callable_audit_module_only_identity"),
        ("qualname_only_identity", "callable_audit_qualname_only_identity"),
        ("filename_only_identity", "callable_audit_filename_only_identity"),
        ("mtime_identity", "callable_audit_mtime_identity"),
        ("validation_observed_identity", "callable_audit_validation_identity"),
        ("expected_code", "callable_audit_observer_expected_metadata"),
        ("_matches_expected", "callable_audit_permissive_matcher"),
        ("missing_positive_inventory", "callable_audit_positive_inventory"),
        ("missing_adversarial_inventory", "callable_audit_adversarial_inventory"),
        ("dynamic_adversary_generator", "callable_audit_dynamic_adversary"),
        ("reused_adversarial_callable", "callable_audit_reused_adversary"),
        ("assembler_source_identity", "callable_audit_assembler_identity"),
        ("backend_source_identity", "callable_audit_backend_identity"),
        ("static_argument_omitted", "callable_audit_static_digest_missing"),
        ("compilation_options_omitted", "callable_audit_compilation_digest_missing"),
        ("input_signature_omitted", "callable_audit_input_signature_missing"),
    )
    for needle, code in rules:
        if needle in source:
            blockers.append(CallableAuditBlocker(code, path))
            break
    result: list[CallableAuditBlocker] = []
    for blocker in blockers:
        if blocker.code not in {item.code for item in result}:
            result.append(blocker)
    return tuple(result)


def require_clean_synthetic_source(source: str) -> None:
    """Raise the first stable source-audit blocker for a synthetic mutation."""
    blockers = audit_synthetic_source(source)
    if blockers:
        raise CallableAuditError(blockers[0].code)
