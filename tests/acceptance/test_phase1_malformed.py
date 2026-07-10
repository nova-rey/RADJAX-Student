from __future__ import annotations

import json

import pytest
from radjax_contract.tome.production import validate_production_tome

from radjax_student.artifacts import TomeArtifactError, open_tome_artifact

from .support import (
    add_unknown_optional_content,
    add_unknown_required_capability,
    copy_fixture,
    mutate_artifact,
    run_cli,
)

MALFORMED_CASES = (
    "path_traversal",
    "absolute_path",
    "stale_hash",
    "stale_size",
    "missing_required_role",
    "duplicate_path",
    "duplicate_role",
    "invalid_surface_reference",
    "invalid_pass_reference",
    "bad_mode_linkage",
    "fingerprint_mode_confusion",
    "packed_array_length",
    "invalid_effective_top_k",
    "selection_mask_mismatch",
    "invalid_token_id",
    "bad_exemplar_corridor_linkage",
)


@pytest.mark.parametrize("mutation", MALFORMED_CASES)
def test_contract_failures_survive_student_and_cli(
    tmp_path,
    mutation: str,
) -> None:
    artifact = copy_fixture(tmp_path)
    blocker = mutate_artifact(artifact, mutation)

    contract_result = validate_production_tome(artifact)
    with pytest.raises(TomeArtifactError) as exc_info:
        open_tome_artifact(artifact)
    code, output, error = run_cli("inspect", "--tome", str(artifact))

    assert not contract_result.ok
    assert any(blocker in item for item in contract_result.blockers)
    assert any(blocker in item for item in exc_info.value.blockers)
    assert code == 2
    assert output == ""
    assert blocker in error
    assert "Artifact could not be opened." in error
    assert "Traceback" not in error


def test_unknown_optional_role_and_surface_remain_inspectable(tmp_path) -> None:
    artifact = copy_fixture(tmp_path)
    add_unknown_optional_content(artifact)

    view = open_tome_artifact(artifact)
    code, output, error = run_cli(
        "inspect",
        "--tome",
        str(artifact),
        "--profile",
        "declaration_test_only",
        "--format",
        "json",
    )
    payload = json.loads(output)

    assert view.validation.contract_status == "pass"
    assert view.surface("future_optional") is not None
    assert "unknown_content_role: future_optional_diagnostic" in view.warnings
    assert code == 0
    assert error == ""
    assert payload["compatibility"]["status"] == "pass"
    assert any(
        finding["code"] == "unknown_optional_surface"
        for finding in payload["compatibility"]["warnings"]
    )


def test_unknown_required_capability_is_explicit_not_structural(tmp_path) -> None:
    artifact = copy_fixture(tmp_path)
    capability = add_unknown_required_capability(artifact)

    view = open_tome_artifact(artifact)
    code, output, error = run_cli(
        "inspect",
        "--tome",
        str(artifact),
        "--profile",
        "declaration_test_only",
        "--format",
        "json",
    )
    payload = json.loads(output)

    assert view.validation.contract_status == "pass"
    assert capability in view.validation.unsupported_required_capabilities
    assert code == 1
    assert error == ""
    assert capability in payload["compatibility"]["missing_capabilities"]
    assert any(
        finding["code"] == "missing_required_capability"
        for finding in payload["compatibility"]["blockers"]
    )
