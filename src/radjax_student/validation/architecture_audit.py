"""Installed AST audit for Phase 3.5 architecture boundaries."""

from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

SCHEMA = "radjax.p3_5_architecture_audit.v2"
FORBIDDEN_IMPORTS = (
    "accelerate",
    "datasets",
    "jax",
    "jaxlib",
    "radjax_tome",
    "torch",
    "transformers",
)
_JAX_EXCEPTIONS = {
    "radjax_student.learning.jax_core",
    "radjax_student.learning.p3_5_acceptance",
    "radjax_student.steps.jax_step",
    "radjax_student.validation.p3_11_9_replay.runner_jax",
    "radjax_student.validation.p3_11_10_gate.runner_jax",
}
_COMPATIBILITY_MODULES = {
    "radjax_student.learning.p3_5_acceptance",
    "radjax_student.learning.p3_10_acceptance",
}
_CLASSIFICATIONS = {
    "artifacts.targets": "smoke_debug",
    "cli.train_student": "deprecated",
    "debug": "smoke_debug",
    "hf": "optional_integration",
    "legacy": "deprecated",
    "learning.observability_acceptance": "transitional",
    "learning.p3_10_acceptance": "transitional",
    "learning.p3_5_acceptance": "transitional",
    "learning.synthetic_smoke": "smoke_debug",
    "validation.p3_11_9_replay.runner_jax": "optional_integration",
    "validation.p3_11_10_gate.runner_jax": "optional_integration",
    "losses": "research",
    "losses.dense_kl": "smoke_debug",
    "losses.sparse_topk": "research",
    "schedules": "research",
    "steps.single": "deprecated",
    "students": "deprecated",
    "training": "transitional",
}
_OWNERS = {
    "architecture": "architecture_plugin",
    "artifacts": "artifact_consumption",
    "checkpoints": "checkpoint_persistence",
    "cli": "product_cli",
    "debug": "debug_implementations",
    "hf": "hugging_face_integration",
    "learning": "learning_contracts",
    "legacy": "legacy_compatibility",
    "losses": "objective_research",
    "optimizers": "optimizer_contracts",
    "reports": "reporting",
    "runtime": "runtime_execution",
    "schedules": "training_policy",
    "steps": "learning_execution",
    "students": "legacy_architecture_compatibility",
    "training": "legacy_training_smoke",
    "validation": "compatibility_validation",
}


class ArchitectureAuditError(ValueError):
    """Stable public rejection for a blocked dependency audit."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def _module_name(path: Path, source_root: Path) -> str:
    parts = path.relative_to(source_root).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(("radjax_student", *parts))


def classify_module(module_name: str) -> str:
    key = module_name.removeprefix("radjax_student.")
    for prefix, classification in sorted(
        _CLASSIFICATIONS.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if key == prefix or key.startswith(prefix + "."):
            return classification
    return "core"


def _owner(module_name: str) -> str:
    key = module_name.removeprefix("radjax_student.")
    return _OWNERS.get(
        key.split(".", 1)[0] if key else "radjax_student", "package_boundary"
    )


def _literal_exports(tree: ast.Module) -> tuple[str, ...]:
    exports: set[str] = set()
    for node in tree.body:
        value: ast.AST | None = None
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            value, targets = node.value, node.targets
        elif isinstance(node, ast.AnnAssign):
            value, targets = node.value, [node.target]
        if value is None or not any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in targets
        ):
            continue
        if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            exports.update(
                item.value
                for item in value.elts
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            )
    return tuple(sorted(exports))


def _imports(tree: ast.Module) -> tuple[str, ...]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add("." * node.level + (node.module or ""))
    return tuple(sorted(names))


def _top_level_internal_edges(tree: ast.Module) -> tuple[str, ...]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(
                alias.name
                for alias in node.names
                if alias.name.startswith("radjax_student")
            )
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module.startswith("radjax_student"):
                names.add(node.module)
    return tuple(sorted(names))


def collect_module_record(path: Path, source_root: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    module = _module_name(path, source_root)
    imports = _imports(tree)
    return {
        "module": module,
        "path": str(path.relative_to(source_root.parent.parent)),
        "owner": _owner(module),
        "classification": classify_module(module),
        "imports": list(imports),
        "internal_edges": list(_top_level_internal_edges(tree)),
        "forbidden_imports": sorted(
            name for name in imports if name.split(".", 1)[0] in FORBIDDEN_IMPORTS
        ),
        "public_exports": list(_literal_exports(tree)),
        "has_dynamic_exports": any(
            isinstance(node, ast.FunctionDef) and node.name == "__getattr__"
            for node in tree.body
        ),
    }


def find_dependency_cycles(records: Iterable[Mapping[str, Any]]) -> list[list[str]]:
    record_list = list(records)
    modules = {str(record["module"]) for record in record_list}
    graph = {
        str(record["module"]): {
            target for target in record["internal_edges"] if target in modules
        }
        for record in record_list
    }
    cycles: set[tuple[str, ...]] = set()

    def visit(node: str, chain: tuple[str, ...]) -> None:
        for target in graph.get(node, set()):
            if target in chain:
                cycle = chain[chain.index(target) :]
                cycles.add(
                    min(
                        tuple(cycle[index:] + cycle[:index])
                        for index in range(len(cycle))
                    )
                )
            elif target not in chain:
                visit(target, chain + (target,))

    for node in graph:
        visit(node, (node,))
    return [list(cycle) for cycle in sorted(cycles)]


def _call_is_parameter_objective(call: ast.Call) -> bool:
    return (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "evaluate"
        and bool(call.args)
        and isinstance(call.args[0], ast.Name)
        and call.args[0].id == "parameters"
    )


def _forward_is_discarded(tree: ast.Module) -> bool:
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    for call in ast.walk(tree):
        if not (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr in {"forward", "apply_jax"}
        ):
            continue
        parent = parents.get(call)
        if isinstance(parent, ast.Expr):
            return True
    return False


def find_architecture_blockers(
    records: list[Mapping[str, Any]], source_root: Path
) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []

    def add(code: str, path: str, message: str, **details: Any) -> None:
        blockers.append({"code": code, "path": path, "message": message, **details})

    by_module = {str(record["module"]): record for record in records}
    root = by_module.get("radjax_student", {})
    if any(
        "students" in item or "debug" in item for item in root.get("public_exports", ())
    ):
        add(
            "root_legacy_export",
            "src/radjax_student/__init__.py",
            "root exports a legacy namespace",
        )
    for record in records:
        module = str(record["module"])
        path = str(record["path"])
        imports = tuple(str(item) for item in record["imports"])
        forbidden = tuple(str(item) for item in record["forbidden_imports"])
        if module in _JAX_EXCEPTIONS:
            forbidden = tuple(item for item in forbidden if not item.startswith("jax"))
        if forbidden:
            add(
                "forbidden_import", path, "forbidden optional import", imports=forbidden
            )
        if module.startswith("radjax_student") and any(
            item == "scripts" or item.startswith("scripts.") for item in imports
        ):
            add(
                "installed_package_imports_scripts",
                path,
                "installed package imports scripts",
            )
        core = (
            classify_module(module) == "core" and module not in _COMPATIBILITY_MODULES
        )
        if core and any(
            item.startswith(
                (
                    "radjax_student.legacy",
                    "radjax_student.debug",
                    "radjax_student.students",
                )
            )
            for item in imports
        ):
            add(
                "core_imports_legacy",
                path,
                "core module imports legacy/debug compatibility",
            )
        if module.startswith("radjax_student.architecture") and any(
            item.startswith("radjax_student.runtime") for item in imports
        ):
            add("architecture_imports_runtime", path, "architecture imports runtime")
        if module.startswith("radjax_student.runtime") and any(
            item.startswith("radjax_student.architecture") for item in imports
        ):
            add("runtime_imports_architecture", path, "runtime imports architecture")
    for path in sorted(source_root.rglob("*.py")):
        module = _module_name(path, source_root)
        if classify_module(module) != "core":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if any(_call_is_parameter_objective(call) for call in ast.walk(tree)):
            add(
                "objective_receives_raw_parameters",
                str(path.relative_to(source_root.parent.parent)),
                "objective receives raw parameters",
            )
        if _forward_is_discarded(tree):
            add(
                "forward_result_discarded",
                str(path.relative_to(source_root.parent.parent)),
                "architecture forward result is discarded",
            )
    artifacts = by_module.get("radjax_student.artifacts", {})
    if "load_dense_tome_targets" in artifacts.get("public_exports", ()):
        add(
            "dense_targets_public_export",
            "src/radjax_student/artifacts/__init__.py",
            "dense targets are exported",
        )
    steps = by_module.get("radjax_student.steps", {})
    if {"learning_step", "ScalarObjective"} & set(steps.get("public_exports", ())):
        add(
            "scalar_step_public_export",
            "src/radjax_student/steps/__init__.py",
            "scalar step is publicly exported",
        )
    return sorted(blockers, key=lambda item: (item["code"], item["path"]))


def build_architecture_audit(
    repo_root: Path | None = None, *, accepted_commit: str | None = None
) -> dict[str, Any]:
    root = Path(repo_root or Path.cwd()).resolve()
    source_root = root / "src" / "radjax_student"
    records = [
        collect_module_record(path, source_root)
        for path in sorted(source_root.rglob("*.py"))
    ]
    cycles = find_dependency_cycles(records)
    blockers = find_architecture_blockers(records, source_root)
    if cycles:
        blockers.extend(
            {
                "code": "dependency_cycle",
                "path": " -> ".join(cycle),
                "message": "internal dependency cycle",
            }
            for cycle in cycles
        )
    return {
        "schema_version": SCHEMA,
        "repository": "RADJAX-Student",
        "accepted_commit": accepted_commit,
        "source_root": "src/radjax_student",
        "forbidden_imports": list(FORBIDDEN_IMPORTS),
        "module_count": len(records),
        "modules": records,
        "dependency_edges": [
            {"source": record["module"], "target": target}
            for record in records
            for target in record["internal_edges"]
        ],
        "cycles": cycles,
        "blockers": sorted(blockers, key=lambda item: (item["code"], item["path"])),
        "status": "pass" if not blockers else "blocked",
    }


build_audit = build_architecture_audit


def require_clean_architecture_audit(audit: Mapping[str, Any]) -> None:
    """Reject a real audit result without consumers inventing failure codes."""

    if audit.get("status") != "pass":
        blockers = audit.get("blockers", ())
        first = blockers[0] if isinstance(blockers, list) and blockers else {}
        code = (
            str(first.get("code", "dependency_audit_blocked"))
            if isinstance(first, Mapping)
            else "dependency_audit_blocked"
        )
        raise ArchitectureAuditError(code, "dependency audit contains blockers")


__all__ = [
    "SCHEMA",
    "build_architecture_audit",
    "build_audit",
    "ArchitectureAuditError",
    "classify_module",
    "collect_module_record",
    "find_architecture_blockers",
    "find_dependency_cycles",
    "require_clean_architecture_audit",
]
