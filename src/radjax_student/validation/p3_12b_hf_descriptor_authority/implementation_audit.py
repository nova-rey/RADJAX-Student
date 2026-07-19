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
_MODULE_EXECUTION_AUTHORITIES = frozenset(
    {
        "_frozen_importlib",
        "_frozen_importlib_external",
        "cffi",
        "code",
        "codeop",
        "cloudpickle",
        "ctypes",
        "dill",
        "doctest",
        "imp",
        "marshal",
        "pkgutil",
        "pickle",
        "pydoc",
        "runpy",
        "timeit",
        "unittest",
        "zipimport",
    }
)


def _is_module_execution_authority(name: str) -> bool:
    """Whether an import spelling exposes the standard module executor."""
    return (
        name.split(".", 1)[0] in _MODULE_EXECUTION_AUTHORITIES
        or name.startswith("importlib._")
        or name == "importlib.machinery"
        or name == "importlib.resources"
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
    # Module-execution authorities have no approved production use.  Reject the
    # capability itself before a source-computed protected target can be hidden
    # in its bootstrap, loader, or result carrier.
    if any(
        isinstance(node, ast.Import)
        and any(_is_module_execution_authority(item.name) for item in node.names)
        or isinstance(node, ast.ImportFrom)
        and _is_module_execution_authority(node.module or "")
        for node in ast.walk(tree)
    ):
        return True
    if any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"compile", "eval", "exec"}
        for node in ast.walk(tree)
    ):
        return True
    if any(
        isinstance(node, ast.Attribute)
        and node.attr == "modules"
        and isinstance(node.value, ast.Name)
        and node.value.id == "sys"
        for node in ast.walk(tree)
    ):
        return True
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func) or ""
        tail = name.rsplit(".", 1)[-1]
        if tail in {
            "_find_and_load",
            "_find_and_load_unlocked",
            "_gcd_import",
            "exec_module",
            "load_module",
            "module_from_spec",
            "resolve_name",
            "locate",
        }:
            return True
    if any(
        isinstance(node, ast.Attribute)
        and node.attr
        in {
            "exec_module",
            "load_module",
            "locate",
            "module_from_spec",
            "_find_and_load",
            "_find_and_load_unlocked",
            "_gcd_import",
            "resolve_name",
            "run_module",
            "run_path",
        }
        for node in ast.walk(tree)
    ):
        return True
    aliases: dict[str, str] = {}
    identity_names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and len(node.args.args) == 1
        and len(node.body) == 1
        and isinstance(node.body[0], ast.Return)
        and isinstance(node.body[0].value, ast.Name)
        and node.body[0].value.id == node.args.args[0].arg
    }
    identity_names.update(
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Lambda)
        and len(node.value.args.args) == 1
        and isinstance(node.value.body, ast.Name)
        and node.value.body.id == node.value.args.args[0].arg
        for target in node.targets
        if isinstance(target, ast.Name)
    )

    def carries_reflection_value(value: ast.AST | None) -> bool:
        if value is None:
            return False
        if isinstance(value, ast.Name) and _call_name(value) == "getattr":
            return True
        if isinstance(value, ast.Attribute):
            return value.attr in {"__getattribute__", "__getitem__"}
        if isinstance(value, ast.IfExp):
            return carries_reflection_value(value.body) or carries_reflection_value(
                value.orelse
            )
        if isinstance(value, ast.BoolOp):
            return any(carries_reflection_value(item) for item in value.values)
        if isinstance(value, ast.Lambda):
            return carries_reflection_value(value.body)
        if isinstance(value, (ast.Tuple, ast.List, ast.Set)):
            return any(carries_reflection_value(item) for item in value.elts)
        if isinstance(value, ast.Dict):
            return any(carries_reflection_value(item) for item in value.values)
        if isinstance(value, ast.Subscript):
            return carries_reflection_value(value.value)
        if isinstance(value, ast.GeneratorExp):
            return carries_reflection_value(value.elt)
        if isinstance(value, ast.Call):
            if _call_name(value.func) == "getattr":
                return False
            return any(carries_reflection_value(item) for item in value.args) or any(
                carries_reflection_value(item.value) for item in value.keywords
            )
        return False

    def carries_mapping_type(value: ast.AST | None) -> bool:
        if value is None:
            return False
        if isinstance(value, ast.Name) and _call_name(value) in {"dict", "object"}:
            return True
        if isinstance(value, ast.IfExp):
            return carries_mapping_type(value.body) or carries_mapping_type(
                value.orelse
            )
        if isinstance(value, ast.BoolOp):
            return any(carries_mapping_type(item) for item in value.values)
        if isinstance(value, ast.Lambda):
            return carries_mapping_type(value.body)
        if isinstance(value, (ast.Tuple, ast.List, ast.Set)):
            return any(carries_mapping_type(item) for item in value.elts)
        if isinstance(value, ast.Dict):
            return any(carries_mapping_type(item) for item in value.values)
        if isinstance(value, ast.Subscript):
            return carries_mapping_type(value.value)
        return False

    if (
        any(
            isinstance(node, (ast.Assign, ast.AnnAssign, ast.Return, ast.Lambda))
            and carries_reflection_value(
                node.value if not isinstance(node, ast.Lambda) else node.body
            )
            for node in ast.walk(tree)
            if not isinstance(node, ast.AnnAssign) or node.value is not None
        )
        or any(
            isinstance(node, ast.Call) and carries_reflection_value(node)
            for node in ast.walk(tree)
        )
        or any(
            isinstance(node, (ast.Assign, ast.AnnAssign))
            and node.value is not None
            and carries_mapping_type(node.value)
            for node in ast.walk(tree)
        )
    ):
        return True
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
        and node.id in {"__builtins__", "globals", "eval", "exec", "vars"}
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

    allowed_importlib_roots: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        raw = _call_name(node.func)
        head, separator, tail = (raw or "").partition(".")
        owner = aliases.get(head, head)
        canonical_name = owner if not separator else f"{owner}.{tail}"
        if canonical_name not in {
            "importlib.import_module",
            "importlib.util.find_spec",
        }:
            continue
        receiver: ast.AST = node.func
        while isinstance(receiver, ast.Attribute):
            receiver = receiver.value
        if isinstance(receiver, ast.Name):
            allowed_importlib_roots.add(id(receiver))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
            continue
        if not isinstance(node.value, ast.Attribute):
            continue
        raw = _call_name(node.value)
        head, separator, tail = (raw or "").partition(".")
        owner = aliases.get(head, head)
        canonical_name = owner if not separator else f"{owner}.{tail}"
        if canonical_name != "importlib.import_module":
            continue
        targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
        if not all(isinstance(target, ast.Name) for target in targets):
            continue
        receiver = node.value.value
        if isinstance(receiver, ast.Name):
            allowed_importlib_roots.add(id(receiver))
    if any(
        isinstance(node, ast.Name)
        and aliases.get(node.id) == "importlib"
        and id(node) not in allowed_importlib_roots
        for node in ast.walk(tree)
    ):
        return True

    primitive_names = {
        name for name, target in aliases.items() if target == "import_module"
    }
    partial_names = {"partial", "functools.partial"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                if item.name == "functools":
                    partial_names.add(f"{item.asname or item.name}.partial")
        elif isinstance(node, ast.ImportFrom) and node.module == "functools":
            for item in node.names:
                if item.name == "partial":
                    partial_names.add(item.asname or item.name)

    def primitive_value(value: ast.AST | None) -> bool:
        if value is None:
            return False
        if isinstance(value, ast.Name):
            return value.id in primitive_names
        if isinstance(value, ast.Attribute):
            raw = _call_name(value)
            head, separator, tail = (raw or "").partition(".")
            owner = aliases.get(head, head)
            return (
                owner if not separator else f"{owner}.{tail}"
            ) == "importlib.import_module"
        if isinstance(value, ast.NamedExpr):
            return primitive_value(value.value)
        if isinstance(value, ast.BinOp):
            return primitive_value(value.left) or primitive_value(value.right)
        if isinstance(value, (ast.Tuple, ast.List, ast.Set)):
            return any(primitive_value(item) for item in value.elts)
        if isinstance(value, ast.Dict):
            return any(primitive_value(item) for item in value.values)
        if isinstance(value, ast.Subscript):
            return primitive_value(value.value)
        if isinstance(value, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            return primitive_value(value.elt) or any(
                primitive_value(generator.iter) for generator in value.generators
            )
        if isinstance(value, ast.Call):
            name = _call_name(value.func)
            if name in primitive_names or name in partial_names:
                return any(primitive_value(item) for item in value.args)
            if isinstance(value.func, ast.Lambda) and len(value.args) == 1:
                argument = value.func.args.args[0] if value.func.args.args else None
                return (
                    argument is not None
                    and isinstance(value.func.body, ast.Name)
                    and value.func.body.id == argument.arg
                    and primitive_value(value.args[0])
                )
        return False

    def forbidden_primitive_target(value: ast.AST | None) -> bool:
        target = _source_literal_string(value) if value is not None else None
        return (
            target is None
            or target in {"pkgutil", "pydoc", "runpy"}
            or (target is not None and "validation" in target)
        )

    direct_callees = {
        id(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)
    }
    class_body_assignments = {
        id(item)
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        for item in node.body
        if isinstance(item, (ast.Assign, ast.AnnAssign))
    }
    allowed_primitive_values = {
        id(node.value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and node.value is not None
        and id(node) not in class_body_assignments
        and primitive_value(node.value)
        and all(
            isinstance(target, ast.Name)
            for target in (
                node.targets if isinstance(node, ast.Assign) else (node.target,)
            )
        )
    }
    if any(
        isinstance(node, ast.Attribute)
        and node.attr == "import_module"
        and id(node) not in direct_callees
        and id(node) not in allowed_primitive_values
        for node in ast.walk(tree)
    ):
        return True
    if any(
        primitive_value(node)
        and id(node) not in direct_callees
        and id(node) not in allowed_primitive_values
        and isinstance(node, (ast.Attribute, ast.Name))
        and (not isinstance(node, ast.Name) or isinstance(node.ctx, ast.Load))
        for node in ast.walk(tree)
    ):
        return True

    changed_primitives = True
    while changed_primitives:
        changed_primitives = False
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                positional = (
                    node.args.args[-len(node.args.defaults) :]
                    if node.args.defaults
                    else ()
                )
                for argument, default in zip(
                    positional, node.args.defaults, strict=True
                ):
                    if primitive_value(default) and argument.arg not in primitive_names:
                        primitive_names.add(argument.arg)
                        changed_primitives = True
                if (
                    any(
                        isinstance(item, ast.Return) and primitive_value(item.value)
                        for item in ast.walk(node)
                    )
                    and node.name not in primitive_names
                ):
                    primitive_names.add(node.name)
                    changed_primitives = True
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
                if not primitive_value(node.value):
                    continue
                targets = (
                    node.targets if isinstance(node, ast.Assign) else (node.target,)
                )
                for target in targets:
                    if (
                        isinstance(target, ast.Name)
                        and target.id not in primitive_names
                    ):
                        primitive_names.add(target.id)
                        changed_primitives = True
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Lambda) and len(node.func.args.args) == 1:
                argument = node.func.args.args[0]
                if (
                    node.args
                    and primitive_value(node.args[0])
                    and isinstance(node.func.body, ast.Call)
                    and isinstance(node.func.body.func, ast.Name)
                    and node.func.body.func.id == argument.arg
                    and forbidden_primitive_target(
                        node.func.body.args[0] if node.func.body.args else None
                    )
                ):
                    return True
            if not primitive_value(node.func):
                continue
            target = (
                node.args[0]
                if node.args
                else next(
                    (
                        keyword.value
                        for keyword in node.keywords
                        if keyword.arg == "name"
                    ),
                    None,
                )
            )
            if forbidden_primitive_target(target):
                return True

    holder_attribute_names: set[str] = set()

    def protected_holder(value: ast.AST | None) -> bool:
        """Follow local import-module carriers only at reflection boundaries."""
        if value is None:
            return False
        if isinstance(value, ast.Name):
            return value.id in holder_names or aliases.get(value.id, value.id) in {
                "importlib",
                "builtins",
                "__builtins__",
            }
        if isinstance(value, ast.Attribute):
            return (
                value.attr == "__dict__"
                and protected_holder(value.value)
                or ((_call_name(value) or "") in holder_attribute_names)
            )
        if isinstance(value, (ast.IfExp, ast.BoolOp)):
            values = (
                (value.body, value.orelse)
                if isinstance(value, ast.IfExp)
                else value.values
            )
            return any(protected_holder(item) for item in values)
        if isinstance(value, ast.Lambda):
            return protected_holder(value.body)
        if isinstance(value, ast.NamedExpr):
            return protected_holder(value.value)
        if isinstance(value, ast.BinOp):
            return protected_holder(value.left) or protected_holder(value.right)
        if isinstance(value, (ast.Tuple, ast.List, ast.Set)):
            return any(protected_holder(item) for item in value.elts)
        if isinstance(value, (ast.ListComp, ast.SetComp)):
            return protected_holder(value.elt) or any(
                protected_holder(generator.iter) for generator in value.generators
            )
        if isinstance(value, ast.Dict):
            return any(protected_holder(item) for item in value.values)
        if isinstance(value, ast.Subscript):
            owner = _call_name(value.value)
            if owner == "sys.modules" and _source_literal_string(value.slice) in {
                "importlib",
                "builtins",
            }:
                return True
            return protected_holder(value.value)
        if isinstance(value, ast.GeneratorExp):
            return protected_holder(value.elt) or any(
                protected_holder(generator.iter) for generator in value.generators
            )
        if isinstance(value, ast.Call):
            name = _call_name(value.func)
            head, separator, tail = (name or "").partition(".")
            resolved_name = aliases.get(head, head)
            canonical_name = (
                resolved_name if not separator else f"{resolved_name}.{tail}"
            )
            builtin_target = (
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
            if name in {"__import__", "builtins.__import__"} and builtin_target:
                return _source_literal_string(builtin_target) in {
                    "importlib",
                    "builtins",
                    "runpy",
                }
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
            if (
                canonical_name == "importlib.import_module"
                or aliases.get(name or "") == "import_module"
            ) and target is not None:
                return _source_literal_string(target) == "runpy"
            if name == "getattr" and len(value.args) >= 2:
                owner = _call_name(value.args[0])
                member = _source_literal_string(value.args[1])
                return (
                    owner is not None
                    and member is not None
                    and (f"{owner}.{member}" in holder_attribute_names)
                )
            if name in holder_names or aliases.get(name or "") in {
                "importlib",
                "builtins",
                "__builtins__",
            }:
                return True
            if isinstance(value.func, ast.Lambda):
                return protected_holder(value.func.body) or any(
                    protected_holder(item) for item in value.args
                )
            if isinstance(value.func, ast.Attribute) and value.func.attr in {
                "get",
                "setdefault",
                "pop",
                "values",
                "__getitem__",
                "copy",
                "__next__",
                "fromkeys",
                "__iter__",
                "__reversed__",
                "send",
                "update",
                "popleft",
            }:
                return protected_holder(value.func.value)
            return any(protected_holder(item) for item in value.args) or any(
                protected_holder(item.value) for item in value.keywords
            )
        return False

    def mapping_carrier(value: ast.AST | None, names: set[str]) -> bool:
        if value is None:
            return False
        if isinstance(value, ast.Name):
            return value.id in names
        if isinstance(value, ast.Call):
            if _call_name(value.func) == "type" and value.args:
                return (
                    isinstance(value.args[0], (ast.Dict, ast.List, ast.Set))
                    or (_call_name(value.args[0]) == "dict")
                    or (
                        isinstance(value.args[0], ast.Call)
                        and _call_name(value.args[0].func) == "dict"
                    )
                )
            if isinstance(value.func, ast.Lambda):
                return mapping_carrier(value.func.body, names) or any(
                    mapping_carrier(item, names) for item in value.args
                )
            name = _call_name(value.func)
            if name in {"deque", "collections.deque"}:
                return any(mapping_carrier(item, names) for item in value.args)
            if name in {*identity_names, "next", "iter"}:
                return any(mapping_carrier(item, names) for item in value.args)
            if isinstance(value.func, ast.Attribute) and value.func.attr in {
                "get",
                "setdefault",
                "pop",
                "values",
                "copy",
                "__next__",
                "fromkeys",
                "__reversed__",
                "send",
                "update",
                "mro",
                "popleft",
            }:
                return mapping_carrier(value.func.value, names)
            return name in names
        if isinstance(value, ast.IfExp):
            return mapping_carrier(value.body, names) or mapping_carrier(
                value.orelse, names
            )
        if isinstance(value, ast.BoolOp):
            return any(mapping_carrier(item, names) for item in value.values)
        if isinstance(value, ast.NamedExpr):
            return mapping_carrier(value.value, names)
        if isinstance(value, ast.BinOp):
            return mapping_carrier(value.left, names) or mapping_carrier(
                value.right, names
            )
        if isinstance(value, (ast.Tuple, ast.List, ast.Set)):
            return any(mapping_carrier(item, names) for item in value.elts)
        if isinstance(value, (ast.ListComp, ast.SetComp)):
            return mapping_carrier(value.elt, names) or any(
                mapping_carrier(generator.iter, names) for generator in value.generators
            )
        if isinstance(value, ast.Dict):
            return any(mapping_carrier(item, names) for item in value.values)
        if isinstance(value, ast.Subscript):
            return mapping_carrier(value.value, names)
        if isinstance(value, ast.Attribute) and value.attr in {
            "__class__",
            "__dict__",
            "__mro__",
            "mro",
        }:
            return (
                isinstance(value.value, (ast.Dict, ast.List, ast.Set))
                or (_call_name(value.value) == "dict")
                or mapping_carrier(value.value, names)
            )
        if isinstance(value, ast.GeneratorExp):
            return mapping_carrier(value.elt, names) or any(
                mapping_carrier(generator.iter, names) for generator in value.generators
            )
        return False

    def reflection_carrier(value: ast.AST | None, mapping_names: set[str]) -> bool:
        if value is None:
            return False
        if isinstance(value, ast.Name):
            return value.id == "getattr"
        if isinstance(value, ast.Attribute):
            return value.attr in {"__getattribute__", "__getitem__"} or (
                value.attr in {"get", "setdefault"}
                and mapping_carrier(value.value, mapping_names)
            )
        if isinstance(value, ast.Call):
            if isinstance(value.func, ast.Lambda):
                return reflection_carrier(value.func.body, mapping_names) or any(
                    reflection_carrier(item, mapping_names) for item in value.args
                )
            if isinstance(value.func, ast.Attribute) and value.func.attr in {
                "get",
                "setdefault",
                "pop",
                "values",
                "copy",
                "__next__",
                "__reversed__",
                "send",
                "update",
                "popleft",
            }:
                return reflection_carrier(value.func.value, mapping_names)
            if _call_name(value.func) in {"deque", "collections.deque"}:
                return any(
                    reflection_carrier(item, mapping_names) for item in value.args
                )
            return _call_name(value.func) in reflection_names
        if isinstance(value, ast.IfExp):
            return reflection_carrier(value.body, mapping_names) or reflection_carrier(
                value.orelse, mapping_names
            )
        if isinstance(value, ast.BoolOp):
            return any(reflection_carrier(item, mapping_names) for item in value.values)
        if isinstance(value, ast.NamedExpr):
            return reflection_carrier(value.value, mapping_names)
        if isinstance(value, ast.BinOp):
            return reflection_carrier(value.left, mapping_names) or reflection_carrier(
                value.right, mapping_names
            )
        if isinstance(value, (ast.Tuple, ast.List, ast.Set)):
            return any(reflection_carrier(item, mapping_names) for item in value.elts)
        if isinstance(value, (ast.ListComp, ast.SetComp)):
            return reflection_carrier(value.elt, mapping_names) or any(
                reflection_carrier(generator.iter, mapping_names)
                for generator in value.generators
            )
        if isinstance(value, ast.Dict):
            return any(reflection_carrier(item, mapping_names) for item in value.values)
        if isinstance(value, ast.Subscript):
            return reflection_carrier(value.value, mapping_names)
        if isinstance(value, ast.GeneratorExp):
            return reflection_carrier(value.elt, mapping_names) or any(
                reflection_carrier(generator.iter, mapping_names)
                for generator in value.generators
            )
        return False

    holder_names = {
        name
        for name, target in aliases.items()
        if target in {"importlib", "builtins", "__builtins__"}
    }
    mapping_names = {"dict", "object"}
    reflection_names: set[str] = set()
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            if protected_holder(node.value):
                for target in targets:
                    target_name = _call_name(target)
                    if target_name is not None and target_name not in holder_names:
                        holder_names.add(target_name)
                        if isinstance(target, ast.Name):
                            aliases[target.id] = "importlib"
                        changed = True
                    if isinstance(target, ast.Attribute):
                        key = _call_name(target) or ""
                        if key not in holder_attribute_names:
                            holder_attribute_names.add(key)
                            changed = True
                    if (
                        isinstance(target, ast.Subscript)
                        and isinstance(target.value, ast.Attribute)
                        and target.value.attr == "__dict__"
                    ):
                        owner = _call_name(target.value.value)
                        member = _source_literal_string(target.slice)
                        if owner is not None and member is not None:
                            key = f"{owner}.{member}"
                            if key not in holder_attribute_names:
                                holder_attribute_names.add(key)
                                changed = True
            if mapping_carrier(node.value, mapping_names):
                for target in targets:
                    if isinstance(target, ast.Name) and target.id not in mapping_names:
                        mapping_names.add(target.id)
                        changed = True
            if reflection_carrier(node.value, mapping_names):
                for target in targets:
                    if (
                        isinstance(target, ast.Name)
                        and target.id not in reflection_names
                    ):
                        reflection_names.add(target.id)
                        changed = True
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or _call_name(node.func) != "setattr":
                continue
            if len(node.args) < 3 or not protected_holder(node.args[2]):
                continue
            owner = _call_name(node.args[0])
            member = _source_literal_string(node.args[1])
            if owner is not None and member is not None:
                key = f"{owner}.{member}"
                if key not in holder_attribute_names:
                    holder_attribute_names.add(key)
                    changed = True
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            returns = [
                item.value
                for item in ast.walk(node)
                if isinstance(item, ast.Return) and item.value is not None
            ]
            if any(protected_holder(value) for value in returns):
                if node.name not in holder_names:
                    aliases[node.name] = "importlib"
                    holder_names.add(node.name)
                    changed = True
            if any(mapping_carrier(value, mapping_names) for value in returns):
                if node.name not in mapping_names:
                    mapping_names.add(node.name)
                    changed = True
            if any(reflection_carrier(value, mapping_names) for value in returns):
                if node.name not in reflection_names:
                    reflection_names.add(node.name)
                    changed = True
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            call_returns = [
                item.value
                for method in node.body
                if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
                and method.name == "__call__"
                for item in ast.walk(method)
                if isinstance(item, ast.Return) and item.value is not None
            ]
            if any(protected_holder(value) for value in call_returns):
                if node.name not in holder_names:
                    aliases[node.name] = "importlib"
                    holder_names.add(node.name)
                    changed = True

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            call_name = _call_name(node.func) or ""
            if (
                (
                    call_name.rsplit(".", 1)[-1] == "__setattr__"
                    or call_name == "setattr"
                )
                and len(node.args) >= 3
                and protected_holder(node.args[2])
            ):
                return True
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "update"
                and any(protected_holder(argument) for argument in node.args)
            ):
                return True
            if _call_name(node.func) == "getattr" and protected_holder(node):
                return True
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in {"run_module", "run_path"}
                and protected_holder(node.func.value)
            ):
                return True
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and node.args
                and protected_holder(node.args[0])
            ):
                return True
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in {"get", "setdefault", "__getitem__"}
                and protected_holder(node.func.value)
            ):
                return True
            if (
                isinstance(node.func, ast.Name)
                and node.func.id in reflection_names
                and any(protected_holder(argument) for argument in node.args)
            ):
                return True
        elif (
            isinstance(node, ast.Attribute)
            and node.attr in {"get", "setdefault", "__getitem__"}
            and protected_holder(node.value)
        ):
            return True
        elif isinstance(node, ast.Subscript) and protected_holder(node.value):
            return True

    if any(_source_literal_string(node) == "__builtins__" for node in ast.walk(tree)):
        return True
    if any(
        isinstance(node, ast.Call)
        and _call_name(node.func) == "getattr"
        and len(node.args) >= 2
        and _source_literal_string(node.args[1]) == "import_module"
        for node in ast.walk(tree)
    ):
        return True
    if any(
        isinstance(node, ast.Attribute)
        and node.attr in {"__getattribute__", "__getitem__"}
        for node in ast.walk(tree)
    ):
        return True
    if any(
        isinstance(node, ast.Attribute)
        and (
            _call_name(node) == "getattr.__call__"
            or _call_name(node) in {"dict.get", "dict.setdefault"}
            or node.attr in {"get", "setdefault"}
            and isinstance(node.value, ast.Call)
            and _call_name(node.value.func) == "type"
        )
        for node in ast.walk(tree)
    ):
        return True
    if any(
        isinstance(node, ast.Call)
        and _call_name(node.func) in {"partial", "functools.partial"}
        and any(
            _call_name(argument) == "getattr"
            or isinstance(argument, ast.Attribute)
            and argument.attr in {"__getattribute__", "__getitem__"}
            for argument in node.args
        )
        for node in ast.walk(tree)
    ):
        return True
    if any(
        isinstance(node, (ast.Assign, ast.AnnAssign, ast.Return, ast.Lambda))
        and (
            _call_name(node.value if not isinstance(node, ast.Lambda) else node.body)
            == "getattr"
            or isinstance(
                node.value if not isinstance(node, ast.Lambda) else node.body,
                ast.Attribute,
            )
            and (node.value if not isinstance(node, ast.Lambda) else node.body).attr
            in {"__getattribute__", "__getitem__"}
        )
        for node in ast.walk(tree)
        if not isinstance(node, ast.AnnAssign) or node.value is not None
    ):
        return True
    if any(
        isinstance(node, (ast.Tuple, ast.List, ast.Set))
        and any(
            _call_name(item) == "getattr"
            or isinstance(item, ast.Attribute)
            and item.attr in {"__getattribute__", "__getitem__"}
            for item in node.elts
        )
        for node in ast.walk(tree)
    ):
        return True
    if any(
        isinstance(node, ast.Dict)
        and any(
            _call_name(item) == "getattr"
            or isinstance(item, ast.Attribute)
            and item.attr in {"__getattribute__", "__getitem__"}
            for item in node.values
        )
        for node in ast.walk(tree)
    ):
        return True
    if any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(
            carries_reflection_value(default) or carries_mapping_type(default)
            for default in (*node.args.defaults, *node.args.kw_defaults)
            if default is not None
        )
        for node in ast.walk(tree)
    ):
        return True
    if any(
        isinstance(node, ast.Call)
        and _call_name(node.func) in {"__import__", "builtins.__import__"}
        and node.args
        and _source_literal_string(node.args[0]) in {"builtins", "operator"}
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
        if (
            target_value is None
            or _is_module_execution_authority(target_value)
            or target_value in {"builtins", "operator"}
            or any(marker in target_value for marker in _PROTECTED_GATE_IMPORT_MARKERS)
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
