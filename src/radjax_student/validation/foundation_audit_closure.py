"""Focused, JAX-free closure audit for the post-P3.12 foundation."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from radjax_student.validation.p3_12b_hf_descriptor_authority import (
    implementation_audit as p312b_implementation_audit,
)
from radjax_student.validation.p3_12b_hf_descriptor_authority import (
    models as p312b_models,
)

SCHEMA_VERSION = "radjax.foundation_audit_closure_report.v1"
REPORT_PATH = Path("docs/FOUNDATION_AUDIT_CLOSURE_REPORT.json")

# This is intentionally a reviewed literal contract, not import-graph discovery.
CANONICAL_TRAINING_PATHS = (
    "architecture/models.py",
    "architecture/protocols.py",
    "architecture/registry.py",
    "objectives/builtin.py",
    "objectives/jax.py",
    "objectives/registry.py",
    "optimizers/jax.py",
    "optimizers/registry.py",
    "runtime/callables.py",
    "runtime/execution.py",
    "runtime/keys.py",
    "runtime/lifecycle.py",
    "learning/assembly.py",
    "learning/composition.py",
    "learning/jax_batch.py",
    "learning/jax_core.py",
    "learning/jax_execution.py",
    "steps/jax_loop.py",
    "steps/jax_step.py",
)
# These owners are a reviewed production boundary, not an import-graph
# approximation.  A proof-shaped module beneath any of them is a policy
# violation unless it is one of the frozen paths below.
PRODUCTION_OWNER_ROOTS = frozenset(
    {
        "architecture",
        "checkpoints",
        "cli",
        "contracts",
        "hf",
        "learning",
        "objectives",
        "optimizers",
        "reports",
        "runtime",
        "steps",
    }
)
PROOF_OWNED_NAMESPACE = "validation"
HISTORICAL_PROOF_EXCEPTIONS = (
    "learning/observability_acceptance.py",
    "learning/p3_10_acceptance.py",
    "learning/p3_5_acceptance.py",
    "learning/synthetic_smoke.py",
)
# Dynamic imports are permitted only when their target is an explicit literal
# (for example the optional-JAX bridge).  A constructed target can conceal an
# otherwise forbidden production dependency from the normal import graph.
_DYNAMIC_IMPORT_CALLEES = frozenset(
    {"__import__", "import_module", "_import_module", "importlib.import_module"}
)
_PROOF_COMMAND_FLAGS = frozenset(
    {"--check-recorded", "--diagnostic", "--write-receipt"}
)
CLAIMS_NOT_MADE = (
    "historical_acceptance_modules_not_fully_relocated",
    "validation_namespace_not_fully_split",
    "arbitrary_architecture_ingestion_not_yet_proven",
    "rwkv_not_implemented",
    "phase4_not_started",
    "full_hf_export_not_implemented",
    "save_pretrained_not_implemented",
    "from_pretrained_not_implemented",
    "teacher_inference_not_implemented",
    "tome_training_not_started",
    "distributed_training_not_proven",
    "multi_device_training_not_proven",
    "tpu_training_not_proven",
    "pallas_optimization_not_started",
    "production_cli_not_implemented",
    "model_quality_not_measured",
)

# This frozen source-byte attestation binds the recorded P3.12B descriptor to
# the JAX-free contract and proof sources that define it.  Refreshing it is a
# deliberate receipt-review operation, never a schema-validity fallback.
P312B_SOURCE_ATTESTATION_PATHS = (
    "contracts/hf.py",
    "validation/p3_11_9_replay/runner_jax.py",
    "validation/p3_12b_hf_descriptor_authority/implementation_audit.py",
    "validation/p3_12b_hf_descriptor_authority/models.py",
    "validation/p3_12b_hf_descriptor_authority/runner_jax.py",
)
P312B_SOURCE_ATTESTATION_DIGEST = (
    "1ea5d7862d4d629deaeaa6143d70a85f465c44e630644cce62cbb9c5a91d7084"
)
P312B_ATTESTED_DESCRIPTOR_DIGEST = (
    "abf84ccc695458fdc857aac0afc2e645cad3d71ec98d6e6a81dbab0075849ff6"
)
P312B_ATTESTED_RECEIPT_SHA256 = (
    "79472ee02e17786922d00a3e019fdfd52c14b5516878699624783e94638ddf90"
)


def _digest(value: object) -> str:
    return hashlib.sha256(
        (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()


def _normalized_from_imports(
    node: ast.ImportFrom, *, relative_path: str
) -> tuple[str, ...]:
    """Resolve a relative import from a source-relative module path."""
    if not node.level:
        base = node.module
    else:
        package = ("radjax_student", *Path(relative_path).parent.parts)
        parent_count = node.level - 1
        base = ".".join(package[: max(1, len(package) - parent_count)])
        if node.module:
            base = f"{base}.{node.module}"
    if not base:
        return ()
    if node.module:
        return (base, *(f"{base}.{item.name}" for item in node.names))
    return tuple(f"{base}.{item.name}" for item in node.names)


def _importlib_aliases(tree: ast.Module) -> dict[str, str]:
    """Resolve local spellings of the standard library import primitives."""
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                if item.name == "importlib":
                    aliases[item.asname or item.name] = "importlib"
                elif item.name == "builtins":
                    aliases[item.asname or item.name] = "builtins"
        elif isinstance(node, ast.ImportFrom) and node.module == "importlib":
            for item in node.names:
                if item.name == "import_module":
                    aliases[item.asname or item.name] = "importlib.import_module"
        elif isinstance(node, ast.ImportFrom) and node.module == "builtins":
            for item in node.names:
                if item.name == "__import__":
                    aliases[item.asname or item.name] = "__import__"
    # Retain source-local bindings of a recognized import primitive.  This is
    # deliberately narrow (only exact importlib/builtins member expressions),
    # but it prevents a protected owner from hiding an import behind one
    # assignment before invoking it.
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = _canonical_import_callee(node.value, aliases)
            if value is None:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            for target in targets:
                if isinstance(target, ast.Name) and aliases.get(target.id) != value:
                    aliases[target.id] = value
                    changed = True
    return aliases


def _canonical_import_callee(node: ast.AST, aliases: dict[str, str]) -> str | None:
    def holder_name(value: ast.AST) -> str | None:
        if isinstance(value, ast.Call):
            raw = _call_name(value.func)
            raw = aliases.get(raw, raw) if raw is not None else None
            target = (
                value.args[0]
                if value.args
                else next(
                    (
                        keyword.value
                        for keyword in value.keywords
                        if keyword.arg == "name"
                    ),
                    None,
                )
            )
            imported = _literal_string(target) if target is not None else None
            if raw in {"__import__", "builtins.__import__"} and imported in {
                "importlib",
                "builtins",
            }:
                return imported
            return None
        if isinstance(value, ast.Attribute):
            parent = holder_name(value.value)
            return f"{parent}.{value.attr}" if parent else None
        raw = _call_name(value)
        if raw is None:
            return None
        head, separator, tail = raw.partition(".")
        resolved_head = aliases.get(head, head)
        return resolved_head if not separator else f"{resolved_head}.{tail}"

    def declared_import_member(value: ast.AST) -> str | None:
        """Resolve explicit ``getattr`` and ``__dict__`` import spellings.

        The audit deliberately recognizes only literal member names.  Any
        computed member name is retained as a possible dynamic import below and
        therefore fails closed at a protected production boundary.
        """
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "getattr"
            and len(value.args) >= 2
        ):
            owner = holder_name(value.args[0])
            member = _literal_string(value.args[1])
            if owner == "importlib" and member == "import_module":
                return "importlib.import_module"
            if owner in {"builtins", "__builtins__"} and member == "__import__":
                return "__import__"
            return None
        if isinstance(value, ast.Subscript):
            owner = holder_name(value.value)
            member = _literal_string(value.slice)
            if owner == "importlib.__dict__" and member == "import_module":
                return "importlib.import_module"
            if (
                owner in {"builtins.__dict__", "__builtins__.__dict__"}
                and member == "__import__"
            ):
                return "__import__"
        return None

    member = declared_import_member(node)
    if member is not None:
        return member
    raw = _call_name(node)
    if raw is None:
        return None
    if raw in _DYNAMIC_IMPORT_CALLEES:
        return "importlib.import_module" if raw != "__import__" else "__import__"
    head, separator, tail = raw.partition(".")
    replacement = aliases.get(head)
    if replacement is None:
        return None
    resolved = replacement if not separator else f"{replacement}.{tail}"
    if resolved in {"__import__", "builtins.__import__"}:
        return "__import__"
    if resolved == "importlib.import_module":
        return resolved
    return None


def _could_be_dynamic_import_callee(node: ast.AST, aliases: dict[str, str]) -> bool:
    """Recognize uncertain import indirection so protected owners fail closed."""
    if _canonical_import_callee(node, aliases) is not None:
        return True
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and node.args
    ):
        holder = _call_name(node.args[0])
        if holder is not None:
            head = holder.split(".", 1)[0]
            return aliases.get(head, head) in {"importlib", "builtins", "__builtins__"}
    if isinstance(node, ast.Subscript):
        holder = _call_name(node.value)
        if holder is not None:
            head = holder.split(".", 1)[0]
            return aliases.get(head, head) in {"importlib", "builtins", "__builtins__"}
    return False


def _resolved_dynamic_import_target(
    node: ast.Call, aliases: dict[str, str]
) -> str | None:
    """Return a literal importlib target, resolving a literal relative package."""
    if _canonical_import_callee(node.func, aliases) is None:
        return None
    target = (
        node.args[0]
        if node.args
        else next(
            (keyword.value for keyword in node.keywords if keyword.arg == "name"),
            None,
        )
    )
    target_value = _literal_string(target) if target is not None else None
    if target_value is None:
        return None
    if not target_value.startswith("."):
        return target_value
    package = next(
        (keyword.value for keyword in node.keywords if keyword.arg == "package"),
        None,
    )
    if not isinstance(package, ast.Constant) or not isinstance(package.value, str):
        return None
    dots = len(target_value) - len(target_value.lstrip("."))
    suffix = target_value[dots:]
    parts = package.value.split(".")
    if dots > len(parts) or not suffix:
        return None
    return ".".join((*parts[: len(parts) - dots + 1], suffix))


def _dynamic_import_calls(tree: ast.Module) -> tuple[ast.Call, ...]:
    aliases = _importlib_aliases(tree)
    return tuple(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _could_be_dynamic_import_callee(node.func, aliases)
    )


def _imports_from_tree(tree: ast.Module, *, relative_path: str) -> tuple[str, ...]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(item.name for item in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.update(_normalized_from_imports(node, relative_path=relative_path))
    aliases = _importlib_aliases(tree)
    for node in _dynamic_import_calls(tree):
        target = _resolved_dynamic_import_target(node, aliases)
        if target is not None:
            names.add(target)
    return tuple(sorted(names))


def _has_dynamic_import_target(tree: ast.Module) -> bool:
    """Detect a source-computed import target without importing the module."""
    aliases = _importlib_aliases(tree)
    return any(
        _resolved_dynamic_import_target(node, aliases) is None
        for node in _dynamic_import_calls(tree)
    )


def _has_import_segment(name: str, segment: str) -> bool:
    return any(part.lower() == segment for part in name.split("."))


def _has_trainable_host_conversion(tree: ast.AST) -> bool:
    """Reject scalar casts whose source names claim trainable numerical state."""
    trainable_tokens = (
        "gradient",
        "optimizer_state",
        "parameter",
        "trainable",
    )
    return any(
        isinstance(node, ast.Call)
        and _call_name(node.func) in {"float", "int"}
        and any(
            token in name.id.lower()
            for argument in node.args
            for name in ast.walk(argument)
            if isinstance(name, ast.Name)
            for token in trainable_tokens
        )
        for node in ast.walk(tree)
    )


def _imports(path: Path, *, relative_path: str) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return _imports_from_tree(tree, relative_path=relative_path)


def _source_root(root: Path) -> Path:
    return root / "src" / "radjax_student"


def _production_paths(root: Path) -> tuple[Path, ...]:
    source = _source_root(root)
    return tuple(
        path
        for path in sorted(source.rglob("*.py"))
        if path.relative_to(source).parts[0] in PRODUCTION_OWNER_ROOTS
    )


def _relative(root: Path, path: Path) -> str:
    return str(path.relative_to(_source_root(root)))


def _has_proof_shape(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return _has_proof_shape_from_tree(tree)


def _literal_string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _literal_string(node.left)
        right = _literal_string(node.right)
        if left is not None and right is not None:
            return left + right
    return None


def _has_proof_shape_from_tree(tree: ast.Module) -> bool:
    """Identify proof behavior even when the module uses a neutral filename."""
    return (
        any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and (
                node.name.startswith(("run_p3_", "execute_p3_"))
                or node.name in {"run_recorded_gate", "execute_recorded_gate"}
                or node.name.endswith("_acceptance")
                or node.name.endswith("_proof")
            )
            for node in ast.walk(tree)
        )
        or any(
            isinstance(node, ast.ClassDef)
            and ("Acceptance" in node.name or node.name.endswith("Proof"))
            for node in ast.walk(tree)
        )
        or any(
            (value := _literal_string(node)) is not None
            and (value in _PROOF_COMMAND_FLAGS or value.startswith("radjax.p3_"))
            for node in ast.walk(tree)
        )
    )


def _is_new_proof_path(relative: str) -> bool:
    """Reject proof/acceptance filenames under production owners exactly."""
    if relative in HISTORICAL_PROOF_EXCEPTIONS:
        return False
    filename = Path(relative).stem
    return any(token in filename for token in ("acceptance", "proof", "audit"))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef | None:
    return next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == name
        ),
        None,
    )


def _reachable_statements(statements: tuple[ast.stmt, ...]) -> tuple[ast.stmt, ...]:
    """Return AST statements not hidden behind a statically dead branch."""
    reachable: list[ast.stmt] = []

    def literal_truth_value(node: ast.AST) -> bool | None:
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            nested = literal_truth_value(node.operand)
            return None if nested is None else not nested
        try:
            return bool(ast.literal_eval(node))
        except (TypeError, ValueError):
            pass
        if not isinstance(node, ast.Compare):
            return None
        try:
            values = [ast.literal_eval(item) for item in (node.left, *node.comparators)]
        except (TypeError, ValueError):
            return None
        for left, operator, right in zip(values, node.ops, values[1:], strict=True):
            if isinstance(operator, ast.Eq):
                matched = left == right
            elif isinstance(operator, ast.NotEq):
                matched = left != right
            elif isinstance(operator, ast.Is):
                matched = left is right
            elif isinstance(operator, ast.IsNot):
                matched = left is not right
            elif isinstance(operator, ast.Lt):
                matched = left < right
            elif isinstance(operator, ast.LtE):
                matched = left <= right
            elif isinstance(operator, ast.Gt):
                matched = left > right
            elif isinstance(operator, ast.GtE):
                matched = left >= right
            else:
                return None
            if not matched:
                return False
        return True

    def visit(items: tuple[ast.stmt, ...]) -> None:
        for statement in items:
            reachable.append(statement)
            if isinstance(statement, ast.If):
                truth = literal_truth_value(statement.test)
                if truth is not None:
                    visit(tuple(statement.body if truth else statement.orelse))
                else:
                    visit(tuple(statement.body))
                    visit(tuple(statement.orelse))
            elif isinstance(statement, ast.Try):
                visit(tuple(statement.body))
                visit(tuple(statement.orelse))
                visit(tuple(statement.finalbody))
                for handler in statement.handlers:
                    visit(tuple(handler.body))
            elif isinstance(statement, (ast.With, ast.AsyncWith)):
                visit(tuple(statement.body))
            if isinstance(statement, (ast.Raise, ast.Return)):
                break

    visit(statements)
    return tuple(reachable)


def _has_reachable_raise(statements: tuple[ast.stmt, ...]) -> bool:
    return any(
        isinstance(statement, ast.Raise)
        for statement in _reachable_statements(statements)
    )


def _assigns_name(function: ast.FunctionDef, name: str) -> bool:
    return any(
        isinstance(target, ast.Name) and target.id == name
        for node in ast.walk(function)
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr))
        for target in (node.targets if isinstance(node, ast.Assign) else (node.target,))
    )


def _loads_hf_descriptor_once(function: ast.FunctionDef) -> bool:
    values = [
        node.value
        for node in ast.walk(function)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else (node.target,))
        if isinstance(target, ast.Name) and target.id == "hf_descriptor"
    ]
    return (
        len(values) == 1
        and isinstance(values[0], ast.Call)
        and _call_name(values[0].func) == "HFCompatibilityDescriptor.from_dict"
    )


def _call_keywords(function: ast.FunctionDef, callee: str) -> set[str]:
    keys: set[str] = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if name == callee:
            keys.update(keyword.arg for keyword in node.keywords if keyword.arg)
    return keys


def _call_keyword_attribute(
    function: ast.FunctionDef,
    callee: str,
    keyword_name: str,
    value: str,
    attribute: str,
) -> bool:
    return any(
        _call_name(node.func) == callee
        and any(
            keyword.arg == keyword_name
            and isinstance(keyword.value, ast.Attribute)
            and isinstance(keyword.value.value, ast.Name)
            and keyword.value.value.id == value
            and keyword.value.attr == attribute
            for keyword in node.keywords
        )
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
    )


def _call_arguments_are(node: ast.Call, callee: str, *arguments: str) -> bool:
    return (
        _call_name(node.func) == callee
        and tuple(_call_name(argument) for argument in node.args) == arguments
    )


def _has_operational_report_validation(function: ast.FunctionDef) -> bool:
    """Require the fixed direct try-path; dead calls do not prove validation."""
    for statement in function.body:
        if isinstance(statement, (ast.Return, ast.Raise)):
            return False
        if isinstance(statement, ast.Try):
            for item in statement.body:
                if isinstance(item, ast.Raise) or _may_return(item):
                    return False
                if (
                    isinstance(item, ast.Expr)
                    and isinstance(item.value, ast.Call)
                    and _call_arguments_are(
                        item.value,
                        "validate_hf_descriptor_match",
                        "executed_descriptor",
                        "summary.descriptor",
                    )
                ):
                    return True
            return False
    return False


def _may_return(statement: ast.stmt) -> bool:
    """Conservatively detect an outer-function return before the matcher."""
    if isinstance(statement, ast.Return):
        return True
    if isinstance(statement, ast.If):
        if isinstance(statement.test, ast.Constant) and isinstance(
            statement.test.value, bool
        ):
            branches = statement.body if statement.test.value else statement.orelse
        else:
            branches = (*statement.body, *statement.orelse)
        return any(_may_return(item) for item in branches)
    if isinstance(
        statement, (ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith)
    ):
        return any(_may_return(item) for item in statement.body)
    if isinstance(statement, ast.Try):
        return any(
            _may_return(item)
            for item in (*statement.body, *statement.orelse, *statement.finalbody)
        ) or any(
            _may_return(item) for handler in statement.handlers for item in handler.body
        )
    if isinstance(statement, ast.Match):
        return any(_may_return(item) for case in statement.cases for item in case.body)
    return False


def _handler_swallows_checkpoint_mismatch(handler: ast.ExceptHandler) -> bool:
    def catches_checkpoint_error(caught: ast.expr | None) -> bool:
        if caught is None:
            return True
        if isinstance(caught, ast.Name):
            return caught.id in {
                "CheckpointValidationError",
                "ValueError",
                "Exception",
                "BaseException",
            }
        if isinstance(caught, ast.Tuple):
            return any(catches_checkpoint_error(item) for item in caught.elts)
        return False

    if not catches_checkpoint_error(handler.type):
        return False
    # A ``finally`` return replaces even a syntactically reachable bare
    # rethrow.  Likewise a context manager can suppress the error through its
    # ``__exit__`` result, which is outside source-local proof.  The frozen
    # checkpoint path needs an unambiguous direct propagation route, so these
    # forms fail closed rather than treating a nested ``raise`` as evidence.
    if any(
        isinstance(node, ast.Try)
        and any(_may_return(item) for item in node.finalbody)
        or isinstance(node, (ast.With, ast.AsyncWith))
        for node in ast.walk(handler)
    ):
        return True
    return not _has_reachable_raise(tuple(handler.body))


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def _contains_attribute(tree: ast.AST, value: str, attribute: str) -> bool:
    return any(
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == value
        and node.attr == attribute
        for node in ast.walk(tree)
    )


def _authority_blockers(sources: dict[str, str]) -> tuple[str, ...]:
    """Prove the fixed HF descriptor route using source structure, not tokens."""
    blockers: list[str] = []

    def tree(path: str) -> ast.Module | None:
        source = sources.get(path)
        if source is None:
            blockers.append(f"hf_authority_source_missing:{path}")
            return None
        return ast.parse(source, filename=path)

    models = tree("architecture/models.py")
    if models is not None:
        result = next(
            (
                node
                for node in models.body
                if isinstance(node, ast.ClassDef)
                and node.name == "ArchitectureInitResult"
            ),
            None,
        )
        if result is None or not (
            _contains_attribute(result, "self", "hf_descriptor")
            and _contains_attribute(result, "self", "hf_reference")
            and any(
                isinstance(node, ast.Call)
                and _call_name(node.func) == "self.hf_descriptor.preservation_reference"
                for node in ast.walk(result)
            )
        ):
            blockers.append("hf_architecture_descriptor_reference_unproven")

    assembly = tree("learning/assembly.py")
    if assembly is not None:
        assemble = _function(assembly, "assemble_jax_learning_lifecycle")
        if assemble is None or not (
            _contains_attribute(assemble, "initialized", "hf_descriptor")
            and _contains_attribute(assemble, "initialized", "hf_reference")
            and _call_keyword_attribute(
                assemble,
                "JaxLearningLifecycle",
                "hf_descriptor",
                "initialized",
                "hf_descriptor",
            )
            and _call_keyword_attribute(
                assemble,
                "JaxLearningLifecycle",
                "hf_reference",
                "initialized",
                "hf_reference",
            )
        ):
            blockers.append("hf_assembly_descriptor_substitution")

    lifecycle = tree("steps/jax_loop.py")
    if lifecycle is not None:
        restore = _function(lifecycle, "restore_from_checkpoint")
        if restore is None or not (
            _call_keyword_attribute(
                restore,
                "load_learning_checkpoint_v3",
                "expected_hf_reference",
                "self",
                "hf_reference",
            )
            and _call_keyword_attribute(
                restore,
                "load_learning_checkpoint_v3",
                "expected_hf_descriptor",
                "self",
                "hf_descriptor",
            )
        ):
            blockers.append("hf_checkpoint_lifecycle_bypass")

    checkpoint = tree("checkpoints/v3.py")
    if checkpoint is not None:
        restore = _function(checkpoint, "load_learning_checkpoint_v3")
        reachable_statements = _reachable_statements(
            tuple(restore.body) if restore is not None else ()
        )

        def contains_expected_descriptor_mismatch(node: ast.AST) -> bool:
            return any(
                isinstance(descendant, ast.If)
                and isinstance(descendant.test, ast.Compare)
                and len(descendant.test.ops) == 1
                and isinstance(descendant.test.ops[0], ast.NotEq)
                and isinstance(descendant.test.left, ast.Name)
                and descendant.test.left.id == "hf_descriptor"
                and len(descendant.test.comparators) == 1
                and isinstance(descendant.test.comparators[0], ast.Name)
                and descendant.test.comparators[0].id == "expected_hf_descriptor"
                for descendant in ast.walk(node)
            )

        has_required_guard = any(
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and len(node.test.ops) == 1
            and isinstance(node.test.ops[0], ast.Is)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "expected_hf_descriptor"
            and len(node.test.comparators) == 1
            and isinstance(node.test.comparators[0], ast.Constant)
            and node.test.comparators[0].value is None
            and _has_reachable_raise(tuple(node.body))
            for node in reachable_statements
        )
        mismatch_rejects = any(
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and len(node.test.ops) == 1
            and isinstance(node.test.ops[0], ast.NotEq)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "hf_descriptor"
            and len(node.test.comparators) == 1
            and isinstance(node.test.comparators[0], ast.Name)
            and node.test.comparators[0].id == "expected_hf_descriptor"
            and _has_reachable_raise(tuple(node.body))
            # Required validation must be reachable from the public restore
            # function; a syntactically present ``if False`` branch is not a
            # validation path.
            for node in reachable_statements
        )
        mismatch_is_swallowed = any(
            any(
                _handler_swallows_checkpoint_mismatch(handler)
                for handler in node.handlers
            )
            and contains_expected_descriptor_mismatch(node)
            for node in ast.walk(restore or checkpoint)
            if isinstance(node, ast.Try)
        ) or any(
            contains_expected_descriptor_mismatch(node)
            and (
                isinstance(node, (ast.With, ast.AsyncWith))
                or isinstance(node, ast.Try)
                and any(_may_return(item) for item in node.finalbody)
            )
            for node in ast.walk(restore or checkpoint)
        )
        if (
            restore is None
            or not has_required_guard
            or not mismatch_rejects
            or mismatch_is_swallowed
            or _assigns_name(restore, "expected_hf_descriptor")
            or not _loads_hf_descriptor_once(restore)
        ):
            blockers.append("hf_checkpoint_descriptor_validation_bypassed")

    replay = tree("validation/p3_11_9_replay/runner_jax.py")
    if replay is not None:
        identity = _function(replay, "_identity")
        if identity is None or not (
            _contains_attribute(identity, "lifecycle", "hf_descriptor")
            and _contains_attribute(identity, "lifecycle", "hf_reference")
        ):
            blockers.append("hf_replay_non_lifecycle_descriptor")

    report = tree("learning/run_report.py")
    if report is not None:
        validate = _function(report, "validate_run_hf_summary")
        if validate is None or not _has_operational_report_validation(validate):
            blockers.append("hf_report_descriptor_validation_bypassed")
    return tuple(sorted(set(blockers)))


@dataclass(frozen=True)
class FoundationAuditReport:
    blockers: tuple[str, ...]
    runtime_steps_import_count: int
    runtime_learning_import_count: int
    production_validation_import_count: int
    canonical_numpy_loss_import_count: int
    test_support_hermetic: bool
    jax_purity_gate: bool
    hf_shape_gate: bool
    p312b_recorded_evidence_current: bool
    phase4_ingestion_policy_locked: bool

    @property
    def composition_root_corrected(self) -> bool:
        return (
            self.runtime_steps_import_count == 0
            and self.runtime_learning_import_count == 0
        )

    @property
    def status(self) -> str:
        return "pass" if not self.blockers else "fail"

    @property
    def evidence_digest(self) -> str:
        return _digest(
            {
                "status": self.status,
                "composition_root_corrected": self.composition_root_corrected,
                "runtime_steps_import_count": self.runtime_steps_import_count,
                "runtime_learning_import_count": self.runtime_learning_import_count,
                "production_validation_import_count": (
                    self.production_validation_import_count
                ),
                "canonical_numpy_loss_import_count": (
                    self.canonical_numpy_loss_import_count
                ),
                "test_support_hermetic": self.test_support_hermetic,
                "jax_purity_gate": self.jax_purity_gate,
                "hf_shape_gate": self.hf_shape_gate,
                "p312b_recorded_evidence_current": self.p312b_recorded_evidence_current,
                "phase4_ingestion_policy_locked": self.phase4_ingestion_policy_locked,
                "claims_not_made": list(CLAIMS_NOT_MADE),
                "blockers": list(self.blockers),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": self.status,
            "composition_root_corrected": self.composition_root_corrected,
            "runtime_steps_import_count": self.runtime_steps_import_count,
            "runtime_learning_import_count": self.runtime_learning_import_count,
            "production_validation_import_count": (
                self.production_validation_import_count
            ),
            "canonical_numpy_loss_import_count": self.canonical_numpy_loss_import_count,
            "test_support_hermetic": self.test_support_hermetic,
            "jax_purity_gate": self.jax_purity_gate,
            "hf_shape_gate": self.hf_shape_gate,
            "p312b_recorded_evidence_current": self.p312b_recorded_evidence_current,
            "phase4_ingestion_policy_locked": self.phase4_ingestion_policy_locked,
            "claims_not_made": list(CLAIMS_NOT_MADE),
            "blockers": list(self.blockers),
            "evidence_digest": self.evidence_digest,
        }


def build_foundation_audit(root: Path | None = None) -> FoundationAuditReport:
    """Measure the frozen closure boundaries without importing JAX surfaces."""
    repository = Path.cwd() if root is None else root
    source = _source_root(repository)
    blockers: list[str] = []
    runtime_sources = tuple(sorted((source / "runtime").rglob("*.py")))
    runtime_imports = tuple(
        name
        for path in runtime_sources
        for name in _imports(path, relative_path=_relative(repository, path))
    )
    runtime_steps = sum(
        name.startswith("radjax_student.steps") for name in runtime_imports
    )
    runtime_learning = sum(
        name.startswith("radjax_student.learning") for name in runtime_imports
    )
    if runtime_steps:
        blockers.append("runtime_steps_import")
    if runtime_learning:
        blockers.append("runtime_learning_import")
    if any(name.startswith("radjax_student.architecture") for name in runtime_imports):
        blockers.append("runtime_architecture_import")
    if any(_has_import_segment(name, "tome") for name in runtime_imports):
        blockers.append("runtime_tome_import")
    if any(_has_import_segment(name, "rwkv") for name in runtime_imports):
        blockers.append("runtime_rwkv_import")
    if any(
        _has_dynamic_import_target(
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        )
        for path in runtime_sources
    ):
        blockers.append("runtime_dynamic_import")

    validation_imports = 0
    for path in _production_paths(repository):
        relative = _relative(repository, path)
        imports = _imports(path, relative_path=relative)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if relative not in HISTORICAL_PROOF_EXCEPTIONS:
            validation_imports += sum(
                name.startswith("radjax_student.validation") for name in imports
            )
        if not relative.startswith("runtime/") and _has_dynamic_import_target(tree):
            blockers.append(f"production_dynamic_import:{relative}")
        if not relative.startswith(("architecture/", "runtime/")) and any(
            _has_import_segment(name, "rwkv") for name in imports
        ):
            blockers.append(f"production_rwkv_import:{relative}")
        if (
            _is_new_proof_path(relative)
            or _has_proof_shape(path)
            and (relative not in HISTORICAL_PROOF_EXCEPTIONS)
        ):
            blockers.append(f"new_production_proof_module:{relative}")
    if validation_imports:
        blockers.append("production_validation_import")

    canonical_loss_imports = 0
    purity_failures: list[str] = []
    forbidden = (
        "numpy",
        "torch",
        "tensorflow",
        "tensorflow_probability",
        "transformers",
    )
    for relative in CANONICAL_TRAINING_PATHS:
        path = source / relative
        imports = _imports(path, relative_path=relative)
        canonical_loss_imports += sum(
            name.startswith("radjax_student.legacy.losses") for name in imports
        )
        if any(name.split(".", 1)[0] in forbidden for name in imports):
            purity_failures.append(relative)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if _has_trainable_host_conversion(tree) or any(
            isinstance(node, ast.Attribute)
            and node.attr in {"device_get", "item", "numpy", "tolist"}
            for node in ast.walk(tree)
        ):
            purity_failures.append(relative)
    if canonical_loss_imports:
        blockers.append("canonical_numpy_loss_import")
    if purity_failures:
        blockers.append(
            "canonical_jax_purity:" + ",".join(sorted(set(purity_failures)))
        )

    test_support_hermetic = _test_support_is_hermetic(repository)
    if not test_support_hermetic:
        blockers.append("test_support_not_hermetic")

    hf_authority_blockers = _hf_shape_blockers(repository)
    if hf_authority_blockers:
        blockers.extend(hf_authority_blockers)
    p312b_current = _p312b_recorded_evidence_current(repository)
    if not p312b_current:
        blockers.append("p312b_recorded_evidence_stale")
    phase4_locked = _phase4_policy_locked(repository)
    if not phase4_locked:
        blockers.append("phase4_ingestion_policy_missing")
    return FoundationAuditReport(
        tuple(sorted(set(blockers))),
        runtime_steps,
        runtime_learning,
        validation_imports,
        canonical_loss_imports,
        test_support_hermetic,
        not purity_failures and canonical_loss_imports == 0,
        not hf_authority_blockers,
        p312b_current,
        phase4_locked,
    )


def _p312b_recorded_evidence_current(root: Path) -> bool:
    try:
        receipt_bytes = (
            root / "docs/P3_12B_HF_DESCRIPTOR_AUTHORITY_RECEIPT.json"
        ).read_bytes()
        payload = json.loads(receipt_bytes)
        p312b_models.validate_receipt(payload)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    source = _source_root(root)
    try:
        source_digests = {
            relative: hashlib.sha256((source / relative).read_bytes()).hexdigest()
            for relative in P312B_SOURCE_ATTESTATION_PATHS
        }
        audit = p312b_implementation_audit.audit_gate_source(
            source / "validation/p3_12b_hf_descriptor_authority/runner_jax.py",
            repository_root=root,
        )
        recorded_audit = payload["implementation_audit"]
    except (KeyError, OSError, TypeError, ValueError):
        return False
    return (
        hashlib.sha256(receipt_bytes).hexdigest() == P312B_ATTESTED_RECEIPT_SHA256
        and _digest(source_digests) == P312B_SOURCE_ATTESTATION_DIGEST
        and audit.status == "pass"
        and recorded_audit["source_evidence_digest"] == audit.source_evidence_digest
        and recorded_audit["implementation_audit_digest"]
        == audit.implementation_audit_digest
        and payload["descriptor_digest"] == P312B_ATTESTED_DESCRIPTOR_DIGEST
        and payload["checkpoint_hf_descriptor_digest"]
        == P312B_ATTESTED_DESCRIPTOR_DIGEST
    )


def _test_support_is_hermetic(repository: Path) -> bool:
    """Prove local ``tests.support`` wins over a competing PYTHONPATH package."""
    support = repository / "tests" / "support" / "linear_objective.py"
    if not support.is_file() or not (repository / "tests" / "__init__.py").is_file():
        return False
    with tempfile.TemporaryDirectory(prefix="radjax-foundation-tests-") as temp:
        competing = Path(temp) / "competing"
        competitor = competing / "tests" / "support"
        competitor.mkdir(parents=True)
        (competitor.parent / "__init__.py").write_text("", encoding="utf-8")
        (competitor / "__init__.py").write_text("", encoding="utf-8")
        (competitor / "linear_objective.py").write_text("", encoding="utf-8")
        expected = f"pathlib.Path({str(support.resolve())!r})"
        script = "\n".join(
            (
                "import importlib.util",
                "import pathlib",
                "spec = importlib.util.find_spec('tests.support.linear_objective')",
                "assert spec is not None and spec.origin is not None",
                "assert pathlib.Path(spec.origin).resolve() == " + expected,
            )
        )
        environment = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                filter(None, (str(competing), os.environ.get("PYTHONPATH")))
            ),
        }
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=repository,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
    return result.returncode == 0


HF_AUTHORITY_PATHS = (
    "architecture/models.py",
    "learning/assembly.py",
    "steps/jax_loop.py",
    "checkpoints/v3.py",
    "validation/p3_11_9_replay/runner_jax.py",
    "learning/run_report.py",
)


def _hf_shape_blockers(root: Path) -> tuple[str, ...]:
    source = _source_root(root)
    try:
        sources = {
            relative: (source / relative).read_text(encoding="utf-8")
            for relative in HF_AUTHORITY_PATHS
        }
    except OSError:
        return ("hf_authority_source_unreadable",)
    return _authority_blockers(sources)


def _phase4_policy_locked(root: Path) -> bool:
    roadmap = (root / "docs/RADJAX_DEVELOPMENT_ROADMAP.md").read_text(encoding="utf-8")
    return all(
        token in roadmap
        for token in (
            "Architecture Plugin Ingestion and First Real Architecture",
            "P4.1 — Architecture Ingestion Contract Freeze",
            "runtime RWKV mode",
            "Phase 4 remains unstarted",
        )
    )


def audit_source_fixture(source: str, *, relative_path: str) -> tuple[str, ...]:
    """Apply the closure's structural policy to one literal source fixture."""
    tree = ast.parse(source, filename=relative_path)
    imports = _imports_from_tree(tree, relative_path=relative_path)
    blockers: list[str] = []
    if relative_path.startswith("runtime/"):
        if any(name.startswith("radjax_student.steps") for name in imports):
            blockers.append("runtime_steps_import")
        if any(name.startswith("radjax_student.learning") for name in imports):
            blockers.append("runtime_learning_import")
        if any(name.startswith("radjax_student.architecture") for name in imports):
            blockers.append("runtime_architecture_import")
        if any(_has_import_segment(name, "tome") for name in imports):
            blockers.append("runtime_tome_import")
        if any(_has_import_segment(name, "rwkv") for name in imports):
            blockers.append("runtime_rwkv_import")
        if _has_dynamic_import_target(tree):
            blockers.append("runtime_dynamic_import")
    if relative_path.split("/", 1)[0] != PROOF_OWNED_NAMESPACE and any(
        name.startswith("radjax_student.validation") for name in imports
    ):
        blockers.append("production_validation_import")
    if (
        relative_path.split("/", 1)[0] in PRODUCTION_OWNER_ROOTS
        and not relative_path.startswith("runtime/")
        and _has_dynamic_import_target(tree)
    ):
        blockers.append(f"production_dynamic_import:{relative_path}")
    if relative_path in CANONICAL_TRAINING_PATHS and any(
        name.split(".", 1)[0]
        in {"numpy", "torch", "tensorflow", "tensorflow_probability", "transformers"}
        for name in imports
    ):
        blockers.append("canonical_jax_purity")
    if relative_path in CANONICAL_TRAINING_PATHS and any(
        name.startswith("radjax_student.legacy.losses") for name in imports
    ):
        blockers.append("canonical_numpy_loss_import")
    if relative_path in CANONICAL_TRAINING_PATHS and (
        _has_trainable_host_conversion(tree)
        or any(
            isinstance(node, ast.Attribute)
            and node.attr in {"device_get", "item", "numpy", "tolist"}
            for node in ast.walk(tree)
        )
    ):
        blockers.append("canonical_jax_purity")
    if (
        relative_path.split("/", 1)[0] in PRODUCTION_OWNER_ROOTS
        and not relative_path.startswith(("architecture/", "runtime/"))
        and any(_has_import_segment(name, "rwkv") for name in imports)
    ):
        blockers.append(f"production_rwkv_import:{relative_path}")
    if relative_path.split("/", 1)[0] in PRODUCTION_OWNER_ROOTS and (
        _is_new_proof_path(relative_path)
        or _has_proof_shape_from_tree(tree)
        and relative_path not in HISTORICAL_PROOF_EXCEPTIONS
    ):
        blockers.append(f"new_production_proof_module:{relative_path}")
    return tuple(sorted(set(blockers)))


def audit_hf_authority_fixture(sources: dict[str, str]) -> tuple[str, ...]:
    """Expose fixed-path HF provenance checks for independent breakage tests."""
    return _authority_blockers(sources)


def _bytes(report: FoundationAuditReport) -> bytes:
    return (
        json.dumps(report.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true")
    group.add_argument("--check-recorded", action="store_true")
    parser.add_argument("--recorded", type=Path, default=REPORT_PATH)
    args = parser.parse_args(argv)
    report = build_foundation_audit(Path.cwd())
    generated = _bytes(report)
    if args.write:
        args.recorded.write_bytes(generated)
        return 0 if report.status == "pass" else 1
    return (
        0 if report.status == "pass" and args.recorded.read_bytes() == generated else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
