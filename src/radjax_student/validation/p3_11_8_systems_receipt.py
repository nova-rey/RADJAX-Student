"""Evidence-coupled receipt construction for the P3.11.8 systems proof."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

SCHEMA_VERSION = "radjax.p3_11_8_stateful_systems_receipt.v2"
NON_CLAIMS = (
    "no_production_architecture",
    "no_tome_payload_consumption",
    "no_distillation",
    "no_hf_export",
    "no_accelerator_scale_training",
    "no_performance_claim",
    "no_radlads_parity_claim",
)
REQUIRED_FLAGS = (
    "complete_architecture_plugin_used",
    "complete_optimizer_plugin_used",
    "public_runtime_path_used",
    "runtime_owned_rng_used",
    "runtime_owned_placement_used",
    "architecture_scope_routing_used",
    "optimizer_boundary_used",
    "generic_loop_used",
    "hooks_used",
    "metrics_retained",
    "report_produced",
    "stateful_carry_advanced",
    "checkpoint_v3_saved",
    "caller_bound_restore_validated",
    "uninterrupted_resumed_equality_passed",
    "eager_jit_comparison_passed",
    "no_legacy_fallback_used",
)


@dataclass(frozen=True)
class StatefulSystemsProofResult:
    """Assertions and normalized runtime evidence from an executed proof."""

    assertions: Mapping[str, bool]
    mode_evidence: Mapping[str, Mapping[str, Any]]
    cross_mode_evidence: Mapping[str, Any]

    def __post_init__(self) -> None:
        assertions = dict(self.assertions)
        if set(assertions) != set(REQUIRED_FLAGS) or not all(
            isinstance(value, bool) for value in assertions.values()
        ):
            raise ValueError("systems proof must provide every boolean assertion")
        object.__setattr__(self, "assertions", MappingProxyType(assertions))
        object.__setattr__(
            self,
            "mode_evidence",
            MappingProxyType(
                {key: dict(value) for key, value in self.mode_evidence.items()}
            ),
        )
        object.__setattr__(
            self,
            "cross_mode_evidence",
            MappingProxyType(dict(self.cross_mode_evidence)),
        )


def build_stateful_systems_receipt(
    proof: StatefulSystemsProofResult,
) -> dict[str, Any]:
    """Build the immutable receipt from actual assertions and receipt evidence."""

    if not all(proof.assertions.values()):
        raise ValueError("cannot issue a passing systems receipt from failed evidence")
    evidence = {
        "modes": {key: proof.mode_evidence[key] for key in sorted(proof.mode_evidence)},
        "cross_mode": dict(proof.cross_mode_evidence),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        **{key: proof.assertions[key] for key in REQUIRED_FLAGS},
        "proof_evidence": evidence,
        "proof_evidence_digest": _digest(evidence),
        "non_claims": list(NON_CLAIMS),
    }


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()


__all__ = [
    "NON_CLAIMS",
    "REQUIRED_FLAGS",
    "SCHEMA_VERSION",
    "StatefulSystemsProofResult",
    "build_stateful_systems_receipt",
]
