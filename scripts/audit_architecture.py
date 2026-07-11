"""Build the Phase 3.5 repository boundary and dependency audit."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

SCHEMA = "radjax.p3_5_architecture_audit.v1"
FORBIDDEN_IMPORTS = (
    "accelerate",
    "datasets",
    "jax",
    "jaxlib",
    "radjax_tome",
    "torch",
    "transformers",
)
CLASSIFICATIONS = {
    "artifacts.targets": "smoke_debug",
    "cli.train_student": "deprecated",
    "hf": "optional_integration",
    "losses": "research",
    "losses.dense_kl": "smoke_debug",
    "losses.sparse_topk": "research",
    "schedules": "research",
    "students": "transitional",
    "students.tiny_debug": "smoke_debug",
    "debug": "smoke_debug",
    "debug.tiny_debug": "smoke_debug",
    "training": "transitional",
    "training.distill": "smoke_debug",
}
OWNERS = {
    "architecture": "architecture_plugin",
    "artifacts": "artifact_consumption",
    "checkpoints": "checkpoint_persistence",
    "cli": "product_cli",
    "hf": "hugging_face_integration",
    "learning": "learning_contracts",
    "losses": "objective_research",
    "optimizers": "optimizer_contracts",
    "reports": "reporting",
    "runtime": "runtime_execution",
    "schedules": "training_policy",
    "steps": "learning_execution",
    "students": "legacy_architecture_compatibility",
    "debug": "debug_implementations",
    "training": "legacy_training_smoke",
    "validation": "compatibility_validation",
}


def _module_name(path: Path, source_root: Path) -> str:
    relative = path.relative_to(source_root).with_suffix("")
    parts = relative.parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(("radjax_student", *parts))


def _package_key(module_name: str) -> str:
    return module_name.removeprefix("radjax_student.")


def _classification(module_name: str) -> str:
    key = _package_key(module_name)
    for prefix, classification in sorted(
        CLASSIFICATIONS.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if key == prefix or key.startswith(prefix + "."):
            return classification
    return "core"


def _owner(module_name: str) -> str:
    key = _package_key(module_name)
    package = key.split(".", 1)[0] if key else "radjax_student"
    return OWNERS.get(package, "package_boundary")


def _literal_strings(value: ast.AST) -> tuple[str, ...]:
    if not isinstance(value, (ast.List, ast.Tuple, ast.Set)):
        return ()
    result = []
    for item in value.elts:
        if isinstance(item, ast.Constant) and isinstance(item.value, str):
            result.append(item.value)
    return tuple(sorted(set(result)))


def _public_exports(tree: ast.Module) -> tuple[str, ...]:
    exports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    exports.update(_literal_strings(node.value))
        if isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "__all__":
                exports.update(_literal_strings(node.value))
    return tuple(sorted(exports))


def _import_names(tree: ast.Module) -> tuple[str, ...]:
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = "." * node.level + (node.module or "")
            imports.add(prefix)
    return tuple(sorted(imports))


def _internal_edges(imports: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted(name for name in imports if name.startswith("radjax_student")))


def _module_record(path: Path, source_root: Path) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    module = _module_name(path, source_root)
    imports = _import_names(tree)
    forbidden = tuple(
        sorted(name for name in imports if name.split(".", 1)[0] in FORBIDDEN_IMPORTS)
    )
    return {
        "module": module,
        "path": str(path.relative_to(source_root.parent.parent)),
        "owner": _owner(module),
        "classification": _classification(module),
        "imports": list(imports),
        "internal_edges": list(_internal_edges(imports)),
        "forbidden_imports": list(forbidden),
        "public_exports": list(_public_exports(tree)),
        "has_module_getattr": any(
            isinstance(node, ast.FunctionDef) and node.name == "__getattr__"
            for node in tree.body
        ),
    }


def _findings(modules: list[dict[str, Any]], source_root: Path) -> list[dict[str, str]]:
    by_module = {item["module"]: item for item in modules}
    blockers: list[dict[str, str]] = []

    def add(code: str, message: str, path: str) -> None:
        blockers.append({"code": code, "message": message, "path": path})

    root = by_module.get("radjax_student", {})
    if any("radjax_student.students" in item for item in root.get("imports", ())):
        add(
            "root_exports_transitional_students",
            "The package root imports the transitional students namespace.",
            "src/radjax_student/__init__.py",
        )
    if "radjax_student.students" in root.get("internal_edges", ()):
        add(
            "root_exports_transitional_students",
            "The package root exposes the transitional students namespace.",
            "src/radjax_student/__init__.py",
        )

    for item in modules:
        if item["forbidden_imports"]:
            add(
                "forbidden_import",
                "A forbidden optional or sibling implementation import exists.",
                item["path"],
            )

    students = [
        item for item in modules if item["module"].startswith("radjax_student.students")
    ]
    if students:
        add(
            "transitional_students_namespace",
            "The students namespace still contains implementation modules.",
            "src/radjax_student/students",
        )

    artifacts = by_module.get("radjax_student.artifacts", {})
    if "load_dense_tome_targets" in artifacts.get("public_exports", ()):
        add(
            "dense_targets_public_export",
            "Dense Tome target loading is exported from the production "
            "artifact namespace.",
            "src/radjax_student/artifacts/__init__.py",
        )

    training = by_module.get("radjax_student.training", {})
    if "run_tiny_train_step" in training.get("public_exports", ()):
        add(
            "tiny_training_public_export",
            "The tiny debug training shim is exported from the training namespace.",
            "src/radjax_student/training/__init__.py",
        )

    single_step = source_root / "steps" / "single.py"
    if single_step.is_file():
        source = single_step.read_text(encoding="utf-8")
        if "objective.evaluate(parameters, batch)" in source:
            add(
                "objective_receives_raw_parameters",
                "The learning step passes raw parameters directly to the objective.",
                "src/radjax_student/steps/single.py",
            )
        if (
            "architecture.forward(" in source
            and "forward = architecture.forward" not in source
        ):
            add(
                "forward_result_discarded",
                "The learning step invokes architecture.forward without "
                "consuming its result.",
                "src/radjax_student/steps/single.py",
            )

    registry_names = {
        path.name.removesuffix(".py") for path in source_root.rglob("*registry.py")
    }
    if (
        "registry" in registry_names
        and (source_root / "students" / "registry.py").is_file()
    ):
        add(
            "competing_architecture_registries",
            "Architecture and transitional student registry implementations coexist.",
            "src/radjax_student/architecture/registry.py;src/radjax_student/students/registry.py",
        )

    return sorted(blockers, key=lambda item: (item["code"], item["path"]))


def build_audit(repo_root: Path | None = None) -> dict[str, Any]:
    root = Path(repo_root or Path(__file__).resolve().parents[1])
    source_root = root / "src" / "radjax_student"
    paths = sorted(source_root.rglob("*.py"))
    modules = [_module_record(path, source_root) for path in paths]
    return {
        "schema_version": SCHEMA,
        "repository": "RADJAX-Student",
        "phase": "P3.5.1",
        "source_root": "src/radjax_student",
        "forbidden_imports": list(FORBIDDEN_IMPORTS),
        "module_count": len(modules),
        "modules": modules,
        "dependency_edges": [
            {"source": item["module"], "target": target}
            for item in modules
            for target in item["internal_edges"]
        ],
        "blockers": _findings(modules, source_root),
        "status": "blocked" if _findings(modules, source_root) else "pass",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/P3_5_DEPENDENCY_AUDIT.json"),
    )
    args = parser.parse_args()
    audit = build_audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"P3.5.1 audit: {audit['status']} "
        f"({audit['module_count']} modules, {len(audit['blockers'])} blockers)"
    )
    return 0 if audit["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
