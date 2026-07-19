"""JAX-free, source-derived anti-cheat audit for the frozen P3.12B gate."""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .inventory import ADVERSARIAL_CASE_IDS, POSITIVE_CASE_IDS

SCHEMA_VERSION = "radjax.p3_12b_implementation_audit.v3"
_SUSPICIOUS_NAMES = {
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
    "boundary_name",
    "mutation_name",
    "inventory_index",
    "function_name",
}
_TRANSLATION_NAMES = {
    "_matches_expected",
    "matches_expected",
    "normalize_expected",
    "normalize_failure",
    "expected_to_observed",
    "observed_to_expected",
    "blocker_aliases",
    "blocker_families",
    "accepted_codes",
    "accepted_prefixes",
    "failure_aliases",
    "map_expected_code",
    "canonicalize_observed_code",
}
_GENERIC_NAMES = {
    "run_case",
    "run_generic_adversary",
    "generic_adversary",
    "execute_case",
    "make_adversary",
    "blocker_for_case",
    "mutation_for_case",
}
_PRODUCTION_OWNERS = {
    "architecture",
    "contracts",
    "checkpoints",
    "cli",
    "learning",
    "runtime",
    "objectives",
    "optimizers",
    "steps",
    "reports",
    "hf",
}
_PROTECTED_GATE_IMPORT_MARKERS = (
    "p3_12b_hf_descriptor_authority",
    "implementation_audit",
)


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
class HFImplementationAuditBlocker:
    code: str
    detail: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.code, str)
            or not self.code
            or not isinstance(self.detail, str)
            or not self.detail
        ):
            raise ValueError(
                "implementation audit blocker fields must be nonempty strings"
            )

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "detail": self.detail}


@dataclass(frozen=True)
class HFImplementationAuditEntry:
    case_id: str
    function_name: str
    source_digest: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.case_id, str)
            or not self.case_id
            or not isinstance(self.function_name, str)
            or not self.function_name
        ):
            raise ValueError("implementation audit entry names must be nonempty")
        _sha(self.source_digest, "implementation audit entry source_digest")

    def to_dict(self) -> dict[str, str]:
        return {
            "case_id": self.case_id,
            "function_name": self.function_name,
            "source_digest": self.source_digest,
        }


@dataclass(frozen=True)
class HFDescriptorGateImplementationAudit:
    source_evidence_digest: str
    positive_case_ids: tuple[str, ...]
    adversarial_source_entries: tuple[HFImplementationAuditEntry, ...]
    blockers: tuple[HFImplementationAuditBlocker, ...]

    def __post_init__(self) -> None:
        _sha(self.source_evidence_digest, "implementation audit source_evidence_digest")
        positives = tuple(self.positive_case_ids)
        entries = tuple(self.adversarial_source_entries)
        blockers = tuple(self.blockers)
        if not all(isinstance(item, str) and item for item in positives):
            raise ValueError("implementation audit positive IDs must be nonempty")
        if not all(isinstance(item, HFImplementationAuditEntry) for item in entries):
            raise TypeError("implementation audit entries must be typed")
        if not all(isinstance(item, HFImplementationAuditBlocker) for item in blockers):
            raise TypeError("implementation audit blockers must be typed")
        object.__setattr__(self, "positive_case_ids", positives)
        object.__setattr__(self, "adversarial_source_entries", entries)
        object.__setattr__(self, "blockers", blockers)

    @property
    def status(self) -> str:
        return "pass" if not self.blockers else "blocked"

    @property
    def adversarial_case_ids(self) -> tuple[str, ...]:
        return tuple(item.case_id for item in self.adversarial_source_entries)

    @property
    def adversarial_inventory_count(self) -> int:
        return len(self.adversarial_source_entries)

    @property
    def positive_inventory_count(self) -> int:
        return len(self.positive_case_ids)

    @property
    def implementation_audit_digest(self) -> str:
        return _digest(
            {
                "source_evidence_digest": self.source_evidence_digest,
                "positive_case_ids": list(self.positive_case_ids),
                "adversarial_source_entries": [
                    item.to_dict() for item in self.adversarial_source_entries
                ],
                "blockers": [item.to_dict() for item in self.blockers],
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": self.status,
            "blockers": [item.to_dict() for item in self.blockers],
            "adversarial_inventory_count": self.adversarial_inventory_count,
            "positive_inventory_count": self.positive_inventory_count,
            "positive_case_ids": list(self.positive_case_ids),
            "adversarial_case_ids": list(self.adversarial_case_ids),
            "adversarial_source_entries": [
                item.to_dict() for item in self.adversarial_source_entries
            ],
            "source_evidence_digest": self.source_evidence_digest,
            "implementation_audit_digest": self.implementation_audit_digest,
        }

    @classmethod
    def from_dict(cls, payload: object) -> HFDescriptorGateImplementationAudit:
        required = {
            "schema_version",
            "status",
            "blockers",
            "adversarial_inventory_count",
            "positive_inventory_count",
            "positive_case_ids",
            "adversarial_case_ids",
            "adversarial_source_entries",
            "source_evidence_digest",
            "implementation_audit_digest",
        }
        if not isinstance(payload, dict) or set(payload) != required:
            raise ValueError("implementation audit fields are missing or unknown")
        if payload["schema_version"] != SCHEMA_VERSION:
            raise ValueError("implementation audit schema is invalid")
        try:
            audit = cls(
                payload["source_evidence_digest"],
                tuple(payload["positive_case_ids"]),
                tuple(
                    HFImplementationAuditEntry(
                        item["case_id"], item["function_name"], item["source_digest"]
                    )
                    for item in payload["adversarial_source_entries"]
                ),
                tuple(
                    HFImplementationAuditBlocker(item["code"], item["detail"])
                    for item in payload["blockers"]
                ),
            )
        except (KeyError, TypeError) as error:
            raise ValueError("implementation audit payload is malformed") from error
        if (
            payload["status"] != audit.status
            or payload["adversarial_inventory_count"]
            != audit.adversarial_inventory_count
            or payload["positive_inventory_count"] != audit.positive_inventory_count
            or tuple(payload["adversarial_case_ids"]) != audit.adversarial_case_ids
            or payload["implementation_audit_digest"]
            != audit.implementation_audit_digest
        ):
            raise ValueError("implementation audit evidence is invalid")
        return audit


def _assignment_index(
    nodes: tuple[ast.AST, ...],
) -> dict[str, tuple[ast.Assign | ast.AnnAssign, ...]]:
    matches: dict[str, list[ast.Assign | ast.AnnAssign]] = {}
    for node in nodes:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    matches.setdefault(target.id, []).append(node)
    return {name: tuple(items) for name, items in matches.items()}


def _assignment(
    assignments: dict[str, tuple[ast.Assign | ast.AnnAssign, ...]], name: str
) -> ast.Assign | ast.AnnAssign | None:
    matches = assignments.get(name, ())
    return matches[0] if len(matches) == 1 else None


def _tuple_values(node: ast.AST | None) -> tuple[ast.expr, ...] | None:
    if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(
        node.value, (ast.Tuple, ast.List)
    ):
        return tuple(node.value.elts)
    return None


def _literal_strings(node: ast.AST | None) -> tuple[str | None, ...] | None:
    values = _tuple_values(node)
    if values is None or not all(
        isinstance(item, ast.Constant)
        and (isinstance(item.value, str) or item.value is None)
        for item in values
    ):
        return None
    return tuple(item.value for item in values)  # type: ignore[union-attr]


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def _positive_ids(
    assignments: dict[str, tuple[ast.Assign | ast.AnnAssign, ...]],
) -> tuple[str, ...]:
    values = _tuple_values(_assignment(assignments, "positives"))
    if values is None:
        return ()
    result: list[str] = []
    for call in values:
        if not (
            isinstance(call, ast.Call)
            and _call_name(call.func) == "_positive"
            and call.args
            and isinstance(call.args[0], ast.Constant)
            and isinstance(call.args[0].value, str)
        ):
            return ()
        result.append(call.args[0].value)
    return tuple(result)


def _boundary_count(values: tuple[ast.expr, ...] | None) -> int | None:
    if values is None:
        return None
    total = 0
    for item in values:
        if not isinstance(item, ast.Starred):
            total += 1
            continue
        value = item.value
        if not (isinstance(value, ast.GeneratorExp) and len(value.generators) == 1):
            return None
        iterator = value.generators[0].iter
        if not (
            isinstance(iterator, ast.Call)
            and _call_name(iterator.func) == "range"
            and len(iterator.args) == 1
            and isinstance(iterator.args[0], ast.Constant)
            and isinstance(iterator.args[0].value, int)
        ):
            return None
        total += iterator.args[0].value
    return total


def _add(blockers: list[HFImplementationAuditBlocker], code: str, detail: str) -> None:
    item = HFImplementationAuditBlocker(code, detail)
    if item not in blockers:
        blockers.append(item)


def _audit_function(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    source: str,
    blockers: list[HFImplementationAuditBlocker],
) -> None:
    arguments = [
        *function.args.posonlyargs,
        *function.args.args,
        *function.args.kwonlyargs,
    ]
    names = {argument.arg for argument in arguments}
    if (
        len(arguments) != 1
        or function.args.vararg
        or function.args.kwarg
        or function.args.kwonlyargs
    ):
        _add(blockers, "adversarial_signature_invalid", function.name)
    if names & _SUSPICIOUS_NAMES:
        _add(blockers, "adversarial_signature_metadata", function.name)
    if not any(isinstance(node, ast.Return) for node in ast.walk(function)):
        _add(blockers, "adversarial_return_missing", function.name)
    for node in ast.walk(function):
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in _SUSPICIOUS_NAMES
        ):
            _add(blockers, "case_driven_semantics", function.name)
        if isinstance(node, ast.Call):
            name = _call_name(node.func) or ""
            if name.split(".")[-1] in _GENERIC_NAMES:
                _add(blockers, "generic_semantic_fallback", function.name)
            if name.split(".")[-1] in _TRANSLATION_NAMES:
                _add(blockers, "forbidden_expected_translation", function.name)
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.slice, ast.Name)
            and node.slice.id in _SUSPICIOUS_NAMES
        ):
            _add(blockers, "case_driven_semantics", function.name)


def _audit_observer(
    tree: ast.Module, blockers: list[HFImplementationAuditBlocker]
) -> None:
    observer = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_observe"
        ),
        None,
    )
    if observer is None:
        _add(blockers, "observer_missing", "_observe")
        return
    names = {
        argument.arg
        for argument in [
            *observer.args.posonlyargs,
            *observer.args.args,
            *observer.args.kwonlyargs,
        ]
    }
    if names != {"invocation"} or observer.args.vararg or observer.args.kwarg:
        _add(blockers, "observer_expected_metadata", "observer parameters")
    if any(
        isinstance(node, ast.Name) and node.id in _SUSPICIOUS_NAMES
        for node in ast.walk(observer)
    ):
        _add(blockers, "observer_expected_metadata", "observer body")


def _audit_translation_and_semantics(
    nodes: tuple[ast.AST, ...], blockers: list[HFImplementationAuditBlocker]
) -> None:
    for node in nodes:
        if isinstance(node, ast.FunctionDef) and node.name in _TRANSLATION_NAMES:
            _add(blockers, "forbidden_expected_translation", node.name)
        if isinstance(node, ast.Call):
            name = (_call_name(node.func) or "").split(".")[-1]
            if name in _TRANSLATION_NAMES:
                _add(blockers, "forbidden_expected_translation", name)
            if name == "partial":
                _add(blockers, "partial_canonical_experiment", "partial")
            if isinstance(node.func, ast.Attribute) and node.func.attr == "startswith":
                _add(blockers, "forbidden_prefix_family_match", "startswith")
        if isinstance(node, ast.For) and any(
            isinstance(item, ast.FunctionDef) for item in node.body
        ):
            _add(blockers, "loop_generated_experiment", "function in loop")


def _audit_observed_boundary(
    source: str, blockers: list[HFImplementationAuditBlocker]
) -> None:
    if "observed_boundary = _callable_identity(first.boundary)" not in source:
        _add(blockers, "observed_boundary_not_callable_derived", "_run")
    if (
        "observed_boundary = spec.intended_boundary" in source
        or 'Invocation(\n        "' in source
    ):
        _add(blockers, "free_standing_observed_boundary", "boundary source")


def _audit_receipt_authority(
    models_path: Path, blockers: list[HFImplementationAuditBlocker]
) -> None:
    source = models_path.read_text(encoding="utf-8") if models_path.is_file() else ""
    required = (
        "ADVERSARIAL_CASE_IDS",
        "POSITIVE_CASE_IDS",
        "HFDescriptorGateImplementationAudit",
        'implementation_audit.status != "pass"',
        "implementation_audit_digest",
    )
    if not source or any(marker not in source for marker in required):
        _add(blockers, "receipt_authority_incomplete", "models.py")
    if "passed=" in source or "accepted=" in source or "success=" in source:
        _add(blockers, "caller_supplied_receipt_success", "models.py")


def _source_literal_string(node: ast.AST | None) -> str | None:
    """Resolve the narrow literal forms allowed for a dynamic import target."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _source_literal_string(node.left)
        right = _source_literal_string(node.right)
        if left is not None and right is not None:
            return left + right
    return None


def _production_dynamic_gate_import(tree: ast.Module, source: str) -> bool:
    """Reject dynamic protected-gate imports, including literal construction.

    Production does not need a dynamic route into this validation gate.  This
    intentionally recognizes the standard import primitives and fails closed
    when source uses reflection to compute a protected target.
    """
    aliases: dict[str, str] = {}
    if any(
        isinstance(node, ast.Import)
        and any(
            item.name.split(".", 1)[0] in {"builtins", "operator"}
            for item in node.names
        )
        or isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.split(".", 1)[0] in {"builtins", "operator"}
        or isinstance(node, ast.Name)
        and node.id in {"__builtins__", "globals", "eval", "exec"}
        for node in ast.walk(tree)
    ):
        return True
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                if item.name in {"importlib", "builtins"}:
                    aliases[item.asname or item.name] = item.name
        elif isinstance(node, ast.ImportFrom):
            if node.module == "importlib":
                for item in node.names:
                    if item.name == "import_module":
                        aliases[item.asname or item.name] = "import_module"
            elif node.module == "builtins":
                for item in node.names:
                    if item.name == "__import__":
                        aliases[item.asname or item.name] = "__import__"

    if any(_source_literal_string(node) == "__builtins__" for node in ast.walk(tree)):
        return True
    if any(
        isinstance(node, ast.Call)
        and _call_name(node.func) in {"__import__", "builtins.__import__"}
        and node.args
        and _source_literal_string(node.args[0]) == "builtins"
        for node in ast.walk(tree)
    ):
        return True

    def member_callee(owner: str | None, member: str | None) -> str | None:
        if owner in {"importlib", "importlib.__dict__"} and member == "import_module":
            return "import_module"
        if (
            owner
            in {
                "builtins",
                "__builtins__",
                "builtins.__dict__",
                "__builtins__.__dict__",
            }
            and member == "__import__"
        ):
            return "__import__"
        if owner in {
            "importlib",
            "builtins",
            "__builtins__",
            "importlib.__dict__",
            "builtins.__dict__",
            "__builtins__.__dict__",
        }:
            return "__potential_dynamic_import__"
        return None

    def holder_name(value: ast.AST) -> str | None:
        if isinstance(value, ast.Name):
            return aliases.get(value.id, value.id)
        if isinstance(value, ast.Attribute):
            parent = holder_name(value.value)
            return f"{parent}.{value.attr}" if parent else None
        if isinstance(value, ast.Call):
            name = _call_name(value.func)
            name = aliases.get(name, name) if name is not None else None
            if name == "vars" and len(value.args) == 1:
                parent = holder_name(value.args[0])
                return f"{parent}.__dict__" if parent else None
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
            imported = _source_literal_string(target)
            if name in {"__import__", "builtins.__import__"} and imported in {
                "importlib",
                "builtins",
            }:
                return imported
        return None

    def import_callee_name(callable_node: ast.AST) -> str | None:
        if isinstance(callable_node, ast.Call):
            name = _call_name(callable_node.func)
            name = aliases.get(name, name) if name is not None else None
            if (
                name
                in {
                    "getattr",
                    "object.__getattribute__",
                    "__reflection_getattr__",
                }
                and len(callable_node.args) >= 2
            ):
                return member_callee(
                    holder_name(callable_node.args[0]),
                    _source_literal_string(callable_node.args[1]),
                )
            if (
                name in {"dict.__getitem__", "__reflection_getitem__"}
                and len(callable_node.args) >= 2
            ):
                return member_callee(
                    holder_name(callable_node.args[0]),
                    _source_literal_string(callable_node.args[1]),
                )
            target = (
                callable_node.args[0]
                if callable_node.args
                else next(
                    (
                        keyword.value
                        for keyword in callable_node.keywords
                        if keyword.arg == "name"
                    ),
                    None,
                )
            )
            imported = _source_literal_string(target)
            if name in {
                "importlib.__dict__",
                "builtins.__dict__",
                "__builtins__.__dict__",
            }:
                return member_callee(name, imported)
            if name in {"__import__", "builtins.__import__"} and imported in {
                "importlib",
                "builtins",
            }:
                return imported
        name = _call_name(callable_node)
        if name in {"import_module", "__import__", "importlib.import_module"}:
            return "__import__" if name == "__import__" else "import_module"
        if name is not None and aliases.get(name) in {"import_module", "__import__"}:
            return aliases[name]
        if name is not None and aliases.get(name) == "__potential_dynamic_import__":
            return aliases[name]
        if name is not None:
            head, separator, tail = name.partition(".")
            if (
                separator
                and aliases.get(head) == "importlib"
                and tail == "import_module"
            ):
                return "import_module"
            if separator and aliases.get(head) == "builtins" and tail == "__import__":
                return "__import__"
            if (
                separator
                and aliases.get(head) in {"importlib", "builtins"}
                and tail == "__dict__"
            ):
                return f"{aliases[head]}.__dict__"
            if (
                separator
                and aliases.get(head)
                in {
                    "importlib.__dict__",
                    "builtins.__dict__",
                    "__builtins__.__dict__",
                }
                and tail in {"get", "setdefault", "__getitem__"}
            ):
                return aliases[head]
            if (
                separator
                and aliases.get(head) == "object"
                and tail == "__getattribute__"
            ):
                return "__reflection_getattr__"
            if separator and aliases.get(head) == "dict" and tail == "__getitem__":
                return "__reflection_getitem__"
        if (
            isinstance(callable_node, ast.Call)
            and isinstance(callable_node.func, ast.Name)
            and callable_node.func.id == "getattr"
            and len(callable_node.args) >= 2
        ):
            holder = holder_name(callable_node.args[0])
            member = _source_literal_string(callable_node.args[1])
            return member_callee(holder, member)
        if (
            isinstance(callable_node, ast.Call)
            and isinstance(callable_node.func, ast.Attribute)
            and callable_node.func.attr == "get"
        ):
            holder = holder_name(callable_node.func.value)
            member = (
                _source_literal_string(callable_node.args[0])
                if callable_node.args
                else None
            )
            return member_callee(holder, member)
        if isinstance(callable_node, ast.Subscript):
            holder = holder_name(callable_node.value)
            member = _source_literal_string(callable_node.slice)
            return member_callee(holder, member)
        return None

    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            binding = import_callee_name(node.value)
            if binding is None:
                declared_name = _call_name(node.value)
                if declared_name in {"getattr", "object.__getattribute__"}:
                    binding = "__reflection_getattr__"
                elif declared_name == "dict.__getitem__":
                    binding = "__reflection_getitem__"
                elif declared_name in {"object", "dict"}:
                    binding = declared_name
            if binding is None:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            for target in targets:
                if isinstance(target, ast.Name) and aliases.get(target.id) != binding:
                    aliases[target.id] = binding
                    changed = True

    if any(
        isinstance(node, ast.Subscript)
        and any(
            _source_literal_string(descendant) in {"__import__", "__builtins__"}
            for descendant in ast.walk(node)
        )
        and import_callee_name(node) is None
        for node in ast.walk(tree)
    ):
        return True

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        callee = import_callee_name(node.func)
        if callee is None:
            # Reflection over an import primitive that this narrow resolver
            # cannot identify is an attempted dynamic gate path and must fail
            # closed.  Ordinary product methods remain outside this narrow
            # reflection family.
            import_markers = {
                "__import__",
                "importlib",
                "import_module",
                "builtins",
                "operator",
                "eval",
            }
            if any(
                isinstance(descendant, ast.Attribute)
                and descendant.attr
                in {
                    "get",
                    "getitem",
                    "__getitem__",
                    "attrgetter",
                    "__getattribute__",
                    "setdefault",
                    "__import__",
                    "import_module",
                }
                for descendant in ast.walk(node.func)
            ) and any(
                isinstance(descendant, ast.Name)
                and descendant.id
                in {
                    "importlib",
                    "builtins",
                    "__builtins__",
                    "operator",
                    "object",
                    "dict",
                }
                or _source_literal_string(descendant) in import_markers
                for descendant in ast.walk(node)
            ):
                return True
            continue
        target = (
            node.args[0]
            if node.args
            else next(
                (keyword.value for keyword in node.keywords if keyword.arg == "name"),
                None,
            )
        )
        target_value = _source_literal_string(target)
        if target_value is None or any(
            marker in target_value for marker in _PROTECTED_GATE_IMPORT_MARKERS
        ):
            return True
    return False


def _audit_production_imports(
    repository_root: Path, blockers: list[HFImplementationAuditBlocker]
) -> None:
    source_root = repository_root / "src" / "radjax_student"
    if not source_root.is_dir():
        return
    for path in sorted(source_root.rglob("*.py")):
        relative = path.relative_to(source_root)
        if not relative.parts or relative.parts[0] not in _PRODUCTION_OWNERS:
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        if _production_dynamic_gate_import(tree, source):
            _add(blockers, "production_imports_gate_code", str(relative))
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            else:
                module = node.module or ""
                names = [module]
                names.extend(
                    f"{module}.{alias.name}" if module else alias.name
                    for alias in node.names
                )
            if any(
                marker in name
                for name in names
                for marker in _PROTECTED_GATE_IMPORT_MARKERS
            ):
                _add(blockers, "production_imports_gate_code", str(relative))


def audit_gate_source(
    path: Path,
    *,
    expected_adversarial_case_ids: tuple[str, ...] = ADVERSARIAL_CASE_IDS,
    expected_positive_case_ids: tuple[str, ...] = POSITIVE_CASE_IDS,
    repository_root: Path | None = None,
) -> HFDescriptorGateImplementationAudit:
    """Audit literal source without importing, inspecting, or executing JAX."""

    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    nodes = tuple(ast.walk(tree))
    assignments = _assignment_index(nodes)
    definitions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    blockers: list[HFImplementationAuditBlocker] = []
    gate_source = expected_adversarial_case_ids == ADVERSARIAL_CASE_IDS
    functions = _tuple_values(_assignment(assignments, "_FUNCTIONS"))
    if functions is None or not all(isinstance(item, ast.Name) for item in functions):
        names: tuple[str, ...] = ()
        registry = _assignment(assignments, "_FUNCTIONS")
        flattened = tuple(ast.walk(registry)) if registry is not None else ()
        if any(isinstance(item, ast.Lambda) for item in flattened):
            _add(blockers, "lambda_canonical_experiment", "_FUNCTIONS")
        elif any(
            isinstance(item, ast.Call)
            and (_call_name(item.func) or "").split(".")[-1] == "partial"
            for item in flattened
        ):
            _add(blockers, "partial_canonical_experiment", "_FUNCTIONS")
        elif any(
            isinstance(item, ast.Call)
            and (_call_name(item.func) or "").split(".")[-1]
            in {"iterdir", "glob", "rglob", "listdir", "walk"}
            for item in flattened
        ):
            _add(blockers, "filesystem_discovered_inventory", "_FUNCTIONS")
        else:
            _add(blockers, "adversarial_registry_invalid", "_FUNCTIONS")
    else:
        names = tuple(item.id for item in functions if isinstance(item, ast.Name))
    case_ids = tuple(name.removeprefix("adversary_") for name in names)
    if case_ids != expected_adversarial_case_ids:
        if len(case_ids) != len(expected_adversarial_case_ids):
            _add(blockers, "wrong_adversarial_count", "canonical inventory")
        elif set(case_ids) == set(expected_adversarial_case_ids):
            _add(blockers, "reordered_adversarial_ids", "canonical inventory")
        else:
            _add(
                blockers, "adversarial_inventory_mismatch", "canonical ordered case IDs"
            )
    if len(set(names)) != len(names):
        _add(blockers, "duplicate_adversarial_function", "_FUNCTIONS")
    line_offsets: tuple[int, ...] = ()
    if source.isascii():
        offsets = [0]
        for line in source.splitlines(keepends=True):
            offsets.append(offsets[-1] + len(line))
        line_offsets = tuple(offsets)
    entries: list[HFImplementationAuditEntry] = []
    for case_id, name in zip(case_ids, names, strict=True):
        function = definitions.get(name)
        if function is None:
            _add(blockers, "missing_adversarial_function", name)
            continue
        _audit_function(function, source, blockers)
        if line_offsets and function.end_lineno is not None:
            segment = source[
                line_offsets[function.lineno - 1] + function.col_offset : line_offsets[
                    function.end_lineno - 1
                ]
                + function.end_col_offset
            ]
        else:
            segment = ast.get_source_segment(source, function) or ""
        entries.append(
            HFImplementationAuditEntry(
                case_id, name, hashlib.sha256(segment.encode()).hexdigest()
            )
        )
    if len({item.source_digest for item in entries}) != len(entries):
        _add(blockers, "reused_semantic_handler", "duplicate experiment source")
    if gate_source:
        codes = _literal_strings(_assignment(assignments, "_CODES"))
        boundaries = _boundary_count(
            _tuple_values(_assignment(assignments, "_BOUNDARIES"))
        )
        if codes is None or len(codes) != len(names):
            _add(blockers, "parallel_inventory_length_mismatch", "_CODES")
        if boundaries is None or boundaries != len(names):
            _add(blockers, "parallel_inventory_length_mismatch", "_BOUNDARIES")
        if "zip(_FUNCTIONS, _BOUNDARIES, _CODES, strict=True)" not in source:
            _add(blockers, "parallel_inventory_mapping_unproven", "SPECS")
    positives = _positive_ids(assignments)
    if positives != expected_positive_case_ids:
        _add(blockers, "positive_inventory_mismatch", "canonical ordered positive IDs")
    if gate_source or "def _observe" in source:
        _audit_observer(tree, blockers)
    if gate_source or "observed_boundary" in source:
        _audit_observed_boundary(source, blockers)
    _audit_translation_and_semantics(nodes, blockers)
    inferred_root = repository_root
    if inferred_root is None:
        parents = path.resolve().parents
        inferred_root = next(
            (
                parent
                for parent in parents
                if (parent / "src" / "radjax_student").is_dir()
            ),
            None,
        )
    if gate_source and inferred_root is not None:
        _audit_production_imports(inferred_root, blockers)
        _audit_receipt_authority(path.with_name("models.py"), blockers)
    return HFDescriptorGateImplementationAudit(
        hashlib.sha256(source.encode()).hexdigest(),
        positives,
        tuple(entries),
        tuple(sorted(blockers, key=lambda item: (item.code, item.detail))),
    )


def require_clean_implementation_audit(
    audit: HFDescriptorGateImplementationAudit,
) -> None:
    if not isinstance(audit, HFDescriptorGateImplementationAudit):
        raise TypeError("implementation audit must be typed")
    if audit.status != "pass":
        raise ValueError(audit.blockers[0].code)


__all__ = [
    "HFDescriptorGateImplementationAudit",
    "HFImplementationAuditBlocker",
    "HFImplementationAuditEntry",
    "SCHEMA_VERSION",
    "audit_gate_source",
    "require_clean_implementation_audit",
]
