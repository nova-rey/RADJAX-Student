"""Versioned leakage-free behavioral partition policy."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from types import MappingProxyType

from radjax_student.behavior.models import BehaviorSplitV1

BEHAVIOR_SPLIT_POLICY_V1 = "behavior_split_policy_v1"
BEHAVIOR_SPLIT_RULE_VERSION_V1 = "1"


class BehaviorSplitError(ValueError):
    """The admitted neutral records cannot support a closed split."""


class BehaviorSplitPolicyV1:
    """Stable-ID partition policy frozen by the approved P5.4 rules."""

    policy_id = BEHAVIOR_SPLIT_POLICY_V1
    rule_version = BEHAVIOR_SPLIT_RULE_VERSION_V1

    def split(
        self,
        *,
        behavioral_source_identity: str,
        example_ids: Iterable[str],
        exemplar_example_ids: Iterable[str],
    ) -> BehaviorSplitV1:
        if not _sha256_identity(behavioral_source_identity):
            raise BehaviorSplitError("behavioral source identity must be sha256")
        examples = _canonical_unique(example_ids, "example ID")
        exemplars = _canonical_ids(exemplar_example_ids, "exemplar example ID")
        if len(exemplars) < 2:
            raise BehaviorSplitError("at least two exemplar-bearing IDs are required")
        if not set(exemplars).issubset(examples):
            raise BehaviorSplitError("exemplar example ID is absent from registry")

        assignments: dict[str, str] = {
            exemplars[0]: "training",
            exemplars[-1]: "held_out",
        }
        counts = {"training": 1, "held_out": 1}
        for example_id in examples:
            if example_id in assignments:
                continue
            partition = (
                "training" if counts["training"] <= counts["held_out"] else "held_out"
            )
            assignments[example_id] = partition
            counts[partition] += 1
        ordered = {example_id: assignments[example_id] for example_id in examples}
        split_identity = _digest(
            {
                "policy_id": self.policy_id,
                "rule_version": self.rule_version,
                "behavioral_source_identity": behavioral_source_identity,
                "assignments": ordered,
            }
        )
        return BehaviorSplitV1(
            policy_id=self.policy_id,
            rule_version=self.rule_version,
            behavioral_source_identity=behavioral_source_identity,
            assignments=MappingProxyType(ordered),
            split_identity=split_identity,
        )


def _canonical_unique(values: Iterable[str], name: str) -> tuple[str, ...]:
    received = tuple(values)
    if any(not isinstance(value, str) or not value for value in received):
        raise BehaviorSplitError(f"{name} must be nonempty strings")
    if len(set(received)) != len(received):
        raise BehaviorSplitError(f"duplicate {name}")
    return tuple(sorted(received, key=lambda value: value.encode("utf-8")))


def _canonical_ids(values: Iterable[str], name: str) -> tuple[str, ...]:
    received = tuple(values)
    if any(not isinstance(value, str) or not value for value in received):
        raise BehaviorSplitError(f"{name} must be nonempty strings")
    return tuple(sorted(set(received), key=lambda value: value.encode("utf-8")))


def _sha256_identity(value: object) -> bool:
    return isinstance(value, str) and _SHA256_IDENTITY.fullmatch(value) is not None


_SHA256_IDENTITY = re.compile(r"sha256:[0-9a-f]{64}\Z")


def _digest(value: Mapping[str, object]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()
