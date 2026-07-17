"""JAX-free AST/source audit for P3.12B literal gate implementation."""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

SCHEMA_VERSION = "radjax.p3_12b_implementation_audit.v2"


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
    """A source-level gate violation, recorded without an expected outcome."""

    code: str
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code:
            raise ValueError("implementation audit blocker code must be nonempty")
        if not isinstance(self.detail, str) or not self.detail:
            raise ValueError("implementation audit blocker detail must be nonempty")

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "detail": self.detail}


@dataclass(frozen=True)
class HFImplementationAuditEntry:
    """A registered literal adversary bound to its own source digest."""

    case_id: str
    function_name: str
    source_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id:
            raise ValueError("implementation audit case_id must be nonempty")
        if not isinstance(self.function_name, str) or not self.function_name:
            raise ValueError("implementation audit function_name must be nonempty")
        _sha(self.source_digest, "implementation audit entry source_digest")

    def to_dict(self) -> dict[str, str]:
        return {
            "case_id": self.case_id,
            "function_name": self.function_name,
            "source_digest": self.source_digest,
        }


@dataclass(frozen=True)
class HFDescriptorGateImplementationAudit:
    """Typed source evidence replacing the former decorative digest."""

    source_evidence_digest: str
    positive_case_ids: tuple[str, ...]
    adversarial_cases: tuple[HFImplementationAuditEntry, ...]
    blockers: tuple[HFImplementationAuditBlocker, ...]

    def __post_init__(self) -> None:
        _sha(self.source_evidence_digest, "implementation audit source_evidence_digest")
        positives = tuple(self.positive_case_ids)
        cases = tuple(self.adversarial_cases)
        blockers = tuple(self.blockers)
        if not all(isinstance(item, str) and item for item in positives):
            raise ValueError("implementation audit positive IDs must be nonempty")
        if not all(isinstance(item, HFImplementationAuditEntry) for item in cases):
            raise TypeError("implementation audit cases must be typed")
        if not all(isinstance(item, HFImplementationAuditBlocker) for item in blockers):
            raise TypeError("implementation audit blockers must be typed")
        object.__setattr__(self, "positive_case_ids", positives)
        object.__setattr__(self, "adversarial_cases", cases)
        object.__setattr__(self, "blockers", blockers)

    @property
    def status(self) -> str:
        return "pass" if not self.blockers else "blocked"

    @property
    def adversarial_case_ids(self) -> tuple[str, ...]:
        return tuple(item.case_id for item in self.adversarial_cases)

    @property
    def implementation_audit_digest(self) -> str:
        return _digest(
            {
                "source_evidence_digest": self.source_evidence_digest,
                "positive_case_ids": list(self.positive_case_ids),
                "adversarial_cases": [
                    item.to_dict() for item in self.adversarial_cases
                ],
                "blockers": [item.to_dict() for item in self.blockers],
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": self.status,
            "source_evidence_digest": self.source_evidence_digest,
            "positive_case_ids": list(self.positive_case_ids),
            "adversarial_cases": [item.to_dict() for item in self.adversarial_cases],
            "blockers": [item.to_dict() for item in self.blockers],
            "implementation_audit_digest": self.implementation_audit_digest,
        }

    @classmethod
    def from_dict(cls, payload: object) -> HFDescriptorGateImplementationAudit:
        if not isinstance(payload, dict) or set(payload) != {
            "schema_version",
            "status",
            "source_evidence_digest",
            "positive_case_ids",
            "adversarial_cases",
            "blockers",
            "implementation_audit_digest",
        }:
            raise ValueError("implementation audit fields are missing or unknown")
        if payload["schema_version"] != SCHEMA_VERSION:
            raise ValueError("implementation audit schema is invalid")
        try:
            audit = cls(
                payload["source_evidence_digest"],
                tuple(payload["positive_case_ids"]),
                tuple(
                    HFImplementationAuditEntry(
                        item["case_id"],
                        item["function_name"],
                        item["source_digest"],
                    )
                    for item in payload["adversarial_cases"]
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
            or payload["implementation_audit_digest"]
            != audit.implementation_audit_digest
        ):
            raise ValueError("implementation audit evidence is invalid")
        return audit


def _assignment(tree: ast.AST, name: str) -> ast.Assign | ast.AnnAssign | None:
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(
            isinstance(target, ast.Name) and target.id == name for target in targets
        ):
            return node
    return None


def _tuple_values(node: ast.AST | None) -> tuple[ast.expr, ...] | None:
    if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(
        node.value, (ast.Tuple, ast.List)
    ):
        return tuple(node.value.elts)
    return None


def _positive_case_ids(tree: ast.Module) -> tuple[str, ...]:
    values = _tuple_values(_assignment(tree, "positives"))
    if values is None:
        return ()
    return tuple(
        call.args[0].value
        for call in values
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "_positive"
            and call.args
            and isinstance(call.args[0], ast.Constant)
            and isinstance(call.args[0].value, str)
        )
    )


def audit_gate_source(
    path: Path,
    *,
    expected_adversarial_count: int = 77,
    expected_positive_case_ids: tuple[str, ...] | None = None,
) -> HFDescriptorGateImplementationAudit:
    """Audit source wiring only; importing this module never imports JAX."""

    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    definitions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    blockers: list[HFImplementationAuditBlocker] = []
    values = _tuple_values(_assignment(tree, "_FUNCTIONS"))
    if values is None or not all(isinstance(item, ast.Name) for item in values):
        names: tuple[str, ...] = ()
        blockers.append(
            HFImplementationAuditBlocker(
                "adversarial_registry_invalid",
                "_FUNCTIONS must be one literal tuple of function names",
            )
        )
    else:
        names = tuple(item.id for item in values if isinstance(item, ast.Name))
    if len(names) != expected_adversarial_count:
        blockers.append(
            HFImplementationAuditBlocker(
                "adversarial_inventory_count_mismatch",
                f"expected {expected_adversarial_count}, observed {len(names)}",
            )
        )
    if len(set(names)) != len(names):
        blockers.append(
            HFImplementationAuditBlocker(
                "adversarial_registry_duplicate",
                "_FUNCTIONS registers a function more than once",
            )
        )
    entries: list[HFImplementationAuditEntry] = []
    for name in names:
        function = definitions.get(name)
        if function is None:
            blockers.append(
                HFImplementationAuditBlocker(
                    "adversarial_function_missing", f"{name} is not defined"
                )
            )
            continue
        arguments = tuple(argument.arg for argument in function.args.args)
        if (
            not name.startswith("adversary_")
            or arguments != ("b",)
            or function.args.vararg is not None
            or function.args.kwarg is not None
            or not any(isinstance(node, ast.Return) for node in ast.walk(function))
        ):
            blockers.append(
                HFImplementationAuditBlocker(
                    "adversarial_function_not_literal",
                    f"{name} must be a one-baseline literal experiment",
                )
            )
        segment = ast.get_source_segment(source, function) or ""
        entries.append(
            HFImplementationAuditEntry(
                name.removeprefix("adversary_"),
                name,
                hashlib.sha256(segment.encode()).hexdigest(),
            )
        )
    for forbidden in (
        "_matches" + "_expected",
        "expected_to" + "_observed",
        "blocker_" + "aliases",
    ):
        if forbidden in source:
            blockers.append(
                HFImplementationAuditBlocker(
                    "forbidden_expected_translation", forbidden
                )
            )
    positives = _positive_case_ids(tree)
    if (
        expected_positive_case_ids is not None
        and positives != expected_positive_case_ids
    ):
        blockers.append(
            HFImplementationAuditBlocker(
                "positive_inventory_mismatch",
                "positive proof IDs must be the canonical ordered inventory",
            )
        )
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
    if audit.blockers:
        raise ValueError(audit.blockers[0].code)


__all__ = [
    "HFDescriptorGateImplementationAudit",
    "HFImplementationAuditBlocker",
    "HFImplementationAuditEntry",
    "SCHEMA_VERSION",
    "audit_gate_source",
    "require_clean_implementation_audit",
]
