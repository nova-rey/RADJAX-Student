"""Focused JAX-free foundation-closure policy tests."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from radjax_student.validation.foundation_audit_closure import (
    CANONICAL_TRAINING_PATHS,
    PRODUCTION_OWNER_ROOTS,
    SCHEMA_VERSION,
    _bytes,
    _p312b_recorded_evidence_current,
    _test_support_is_hermetic,
    audit_hf_authority_fixture,
    audit_source_fixture,
    build_foundation_audit,
    main,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.jax
def test_foundation_audit_is_clean_and_uses_literal_canonical_paths() -> None:
    report = build_foundation_audit(ROOT)
    assert report.status == "pass"
    assert report.to_dict()["schema_version"] == SCHEMA_VERSION
    assert "recorded_gates_read_only" not in report.to_dict()
    assert "learning/composition.py" in CANONICAL_TRAINING_PATHS
    assert "steps/jax_step.py" in CANONICAL_TRAINING_PATHS


def test_literal_source_fixtures_reject_forbidden_foundation_edges() -> None:
    assert audit_source_fixture(
        "from radjax_student.steps import jax_step\n", relative_path="runtime/x.py"
    ) == ("runtime_steps_import",)
    assert audit_source_fixture(
        "from ..steps import jax_step\n", relative_path="runtime/x.py"
    ) == ("runtime_steps_import",)
    assert audit_source_fixture(
        "from radjax_student import steps\n", relative_path="runtime/x.py"
    ) == ("runtime_steps_import",)
    assert audit_source_fixture(
        "from radjax_student.validation import compatibility\n",
        relative_path="reports/x.py",
    ) == ("production_validation_import",)
    assert audit_source_fixture(
        "from ..validation import compatibility\n", relative_path="runtime/x.py"
    ) == ("production_validation_import",)
    assert audit_source_fixture(
        "from radjax_student import validation\n", relative_path="reports/x.py"
    ) == ("production_validation_import",)
    assert audit_source_fixture(
        "import numpy\n", relative_path="steps/jax_step.py"
    ) == ("canonical_jax_purity",)
    assert audit_source_fixture(
        "from radjax_student.legacy.losses import dense_kl_loss\n",
        relative_path="learning/assembly.py",
    ) == ("canonical_numpy_loss_import",)
    assert audit_source_fixture(
        "from ..legacy.losses import dense_kl_loss\n",
        relative_path="learning/assembly.py",
    ) == ("canonical_numpy_loss_import",)
    assert audit_source_fixture(
        "from radjax_student.legacy import losses\n",
        relative_path="learning/assembly.py",
    ) == ("canonical_numpy_loss_import",)
    assert audit_source_fixture(
        "from ..legacy import losses\n", relative_path="learning/assembly.py"
    ) == ("canonical_numpy_loss_import",)
    assert audit_source_fixture(
        "from ...legacy import losses\n", relative_path="learning/assembly.py"
    ) == ("canonical_numpy_loss_import",)
    assert (
        audit_source_fixture(
            "from ..contracts import ObjectiveConfig\n", relative_path="runtime/x.py"
        )
        == ()
    )
    assert (
        audit_source_fixture(
            "from radjax_student import contracts\n", relative_path="runtime/x.py"
        )
        == ()
    )
    assert (
        audit_source_fixture("from .. import contracts\n", relative_path="runtime/x.py")
        == ()
    )
    assert audit_source_fixture(
        "from radjax_student.validation import gate\n",
        relative_path="cli/inspect.py",
    ) == ("production_validation_import",)
    assert audit_source_fixture(
        "def execute_p3_99_proof(): pass\n",
        relative_path="learning/new_acceptance.py",
    ) == ("new_production_proof_module:learning/new_acceptance.py",)
    assert (
        audit_source_fixture(
            "def execute_p3_5_proof(): pass\n",
            relative_path="learning/p3_5_acceptance.py",
        )
        == ()
    )
    assert audit_source_fixture(
        "from radjax_student.learning import assembly\n",
        relative_path="runtime/x.py",
    ) == ("runtime_learning_import",)
    assert audit_source_fixture(
        "from radjax_student.architecture import registry\n",
        relative_path="runtime/x.py",
    ) == ("runtime_architecture_import",)
    assert audit_source_fixture(
        "from radjax_student import tome\n", relative_path="runtime/x.py"
    ) == ("runtime_tome_import",)
    assert audit_source_fixture(
        "from radjax_student import rwkv\n", relative_path="runtime/x.py"
    ) == ("runtime_rwkv_import",)
    assert audit_source_fixture(
        "from radjax_student import rwkv\n", relative_path="learning/x.py"
    ) == ("production_rwkv_import:learning/x.py",)
    assert audit_source_fixture(
        "import importlib\nimportlib.import_module('radjax_student.' + 'steps')\n",
        relative_path="runtime/x.py",
    ) == ("runtime_steps_import",)
    assert audit_source_fixture(
        "from importlib import import_module as load\nload('radjax_student.steps')\n",
        relative_path="runtime/x.py",
    ) == ("runtime_steps_import",)
    assert audit_source_fixture(
        "from importlib import import_module as load\n"
        "load('radjax_student.' + 'steps')\n",
        relative_path="runtime/x.py",
    ) == ("runtime_steps_import",)
    assert audit_source_fixture(
        "import importlib as loader\n"
        "loader.import_module('.steps', package='radjax_student')\n",
        relative_path="runtime/x.py",
    ) == ("runtime_steps_import",)
    assert audit_source_fixture(
        "import importlib\nimportlib.import_module(name='radjax_student.steps')\n",
        relative_path="runtime/x.py",
    ) == ("runtime_steps_import",)
    assert audit_source_fixture(
        "import builtins\nbuiltins.__import__('radjax_student.steps')\n",
        relative_path="runtime/x.py",
    ) == ("runtime_steps_import",)
    assert audit_source_fixture(
        "from builtins import __import__ as load\nload('radjax_student.steps')\n",
        relative_path="runtime/x.py",
    ) == ("runtime_steps_import",)
    assert audit_source_fixture(
        "import importlib\n"
        "getattr(importlib, 'import_module')('radjax_student.steps')\n",
        relative_path="runtime/x.py",
    ) == ("runtime_steps_import",)
    assert audit_source_fixture(
        "from importlib import import_module as load\n"
        "load('radjax_student.validation')\n",
        relative_path="reports/x.py",
    ) == ("production_validation_import",)
    assert audit_source_fixture(
        "import importlib\nimportlib.import_module('radjax_student.' + 'validation')\n",
        relative_path="reports/x.py",
    ) == ("production_validation_import",)
    assert audit_source_fixture(
        "import importlib\ngetattr(importlib, member_name)(target_name)\n",
        relative_path="reports/x.py",
    ) == ("production_dynamic_import:reports/x.py",)
    assert audit_source_fixture(
        "import jax\njax.device_get(value)\n",
        relative_path="steps/jax_step.py",
    ) == ("canonical_jax_purity",)
    assert audit_source_fixture(
        "import jax\nvalue.item()\n",
        relative_path="steps/jax_step.py",
    ) == ("canonical_jax_purity",)
    assert audit_source_fixture(
        "float(trainable_array)\n", relative_path="steps/jax_step.py"
    ) == ("canonical_jax_purity",)
    assert audit_source_fixture(
        "SCHEMA = 'radjax.p3_99_neutral_gate.v1'\ndef run(): pass\n",
        relative_path="learning/ordinary.py",
    ) == ("new_production_proof_module:learning/ordinary.py",)
    assert audit_source_fixture(
        "SCHEMA = 'radjax.' + 'p3_99_neutral_gate.v1'\ndef run(): pass\n",
        relative_path="learning/ordinary.py",
    ) == ("new_production_proof_module:learning/ordinary.py",)


def test_production_owners_include_cli_and_test_support_beats_competitors() -> None:
    assert "cli" in PRODUCTION_OWNER_ROOTS
    assert _test_support_is_hermetic(ROOT)
    assert audit_source_fixture(
        "from radjax_student import learning\n", relative_path="runtime/x.py"
    ) == ("runtime_learning_import",)
    assert audit_source_fixture(
        "__import__('radjax_student.steps')\n", relative_path="runtime/x.py"
    ) == ("runtime_steps_import",)
    assert audit_source_fixture(
        "import importlib\nimportlib.import_module('radjax_student.validation')\n",
        relative_path="reports/x.py",
    ) == ("production_validation_import",)
    assert audit_source_fixture(
        "if True:\n    def execute_p3_99_proof(): pass\n",
        relative_path="learning/module.py",
    ) == ("new_production_proof_module:learning/module.py",)
    assert audit_source_fixture(
        "def enclosing():\n    class NestedAcceptance: pass\n",
        relative_path="runtime/module.py",
    ) == ("new_production_proof_module:runtime/module.py",)


def test_hf_authority_ast_rejects_independent_path_breakages() -> None:
    source = ROOT / "src" / "radjax_student"
    paths = (
        "architecture/models.py",
        "learning/assembly.py",
        "steps/jax_loop.py",
        "checkpoints/v3.py",
        "validation/p3_11_9_replay/runner_jax.py",
        "learning/run_report.py",
    )
    sources = {path: (source / path).read_text(encoding="utf-8") for path in paths}
    assert audit_hf_authority_fixture(sources) == ()

    assembly = dict(sources)
    assembly["learning/assembly.py"] = assembly["learning/assembly.py"].replace(
        "hf_descriptor=initialized.hf_descriptor", "hf_descriptor=foreign_descriptor"
    )
    assert audit_hf_authority_fixture(assembly) == (
        "hf_assembly_descriptor_substitution",
    )

    checkpoint = dict(sources)
    checkpoint["checkpoints/v3.py"] = checkpoint["checkpoints/v3.py"].replace(
        "if expected_hf_descriptor is None:", "if False:"
    )
    assert audit_hf_authority_fixture(checkpoint) == (
        "hf_checkpoint_descriptor_validation_bypassed",
    )

    unreachable_guard = dict(sources)
    unreachable_guard["checkpoints/v3.py"] = unreachable_guard[
        "checkpoints/v3.py"
    ].replace(
        """if expected_hf_descriptor is None:
        raise CheckpointValidationError(
            "checkpoint_hf_descriptor_missing",
            "caller-bound continuation requires expected HF descriptor",
        )""",
        """if False:
        if expected_hf_descriptor is None:
            raise CheckpointValidationError(
                "checkpoint_hf_descriptor_missing",
                "caller-bound continuation requires expected HF descriptor",
            )""",
    )
    assert audit_hf_authority_fixture(unreachable_guard) == (
        "hf_checkpoint_descriptor_validation_bypassed",
    )

    for falsey_guard in ("0", "1 == 0", "not True"):
        falsey_required_guard = dict(sources)
        falsey_required_guard["checkpoints/v3.py"] = falsey_required_guard[
            "checkpoints/v3.py"
        ].replace(
            """    if expected_hf_descriptor is None:
        raise CheckpointValidationError(
            "checkpoint_hf_descriptor_missing",
            "caller-bound continuation requires expected HF descriptor",
        )""",
            f"""    if {falsey_guard}:
        if expected_hf_descriptor is None:
            raise CheckpointValidationError(
                "checkpoint_hf_descriptor_missing",
                "caller-bound continuation requires expected HF descriptor",
            )""",
        )
        assert audit_hf_authority_fixture(falsey_required_guard) == (
            "hf_checkpoint_descriptor_validation_bypassed",
        )

    comparison = dict(sources)
    comparison["checkpoints/v3.py"] = comparison["checkpoints/v3.py"].replace(
        "if hf_descriptor != expected_hf_descriptor:", "if False:"
    )
    assert audit_hf_authority_fixture(comparison) == (
        "hf_checkpoint_descriptor_validation_bypassed",
    )

    falsey_comparison = dict(sources)
    falsey_comparison["checkpoints/v3.py"] = falsey_comparison[
        "checkpoints/v3.py"
    ].replace(
        """    if hf_descriptor != expected_hf_descriptor:
        raise CheckpointValidationError(
            "checkpoint_hf_descriptor_mismatch",
            "checkpoint HF descriptor does not match caller expectation",
        )""",
        """    if 1 == 0:
        if hf_descriptor != expected_hf_descriptor:
            raise CheckpointValidationError(
                "checkpoint_hf_descriptor_mismatch",
                "checkpoint HF descriptor does not match caller expectation",
            )""",
    )
    assert audit_hf_authority_fixture(falsey_comparison) == (
        "hf_checkpoint_descriptor_validation_bypassed",
    )

    mismatch_noop = dict(sources)
    mismatch_noop["checkpoints/v3.py"] = mismatch_noop["checkpoints/v3.py"].replace(
        """        raise CheckpointValidationError(
            "checkpoint_hf_descriptor_mismatch",
            "checkpoint HF descriptor does not match caller expectation",
        )""",
        "        pass",
    )
    assert audit_hf_authority_fixture(mismatch_noop) == (
        "hf_checkpoint_descriptor_validation_bypassed",
    )

    return_then_raise = dict(sources)
    return_then_raise["checkpoints/v3.py"] = return_then_raise[
        "checkpoints/v3.py"
    ].replace(
        """        raise CheckpointValidationError(
            "checkpoint_hf_descriptor_mismatch",
            "checkpoint HF descriptor does not match caller expectation",
        )""",
        """        return None
        raise CheckpointValidationError(
            "checkpoint_hf_descriptor_mismatch",
            "checkpoint HF descriptor does not match caller expectation",
        )""",
    )
    assert audit_hf_authority_fixture(return_then_raise) == (
        "hf_checkpoint_descriptor_validation_bypassed",
    )

    rebound_expected = dict(sources)
    rebound_expected["checkpoints/v3.py"] = rebound_expected[
        "checkpoints/v3.py"
    ].replace(
        "    if hf_descriptor != expected_hf_descriptor:",
        "    expected_hf_descriptor = hf_descriptor\n"
        "    if hf_descriptor != expected_hf_descriptor:",
    )
    assert audit_hf_authority_fixture(rebound_expected) == (
        "hf_checkpoint_descriptor_validation_bypassed",
    )

    rebound_observed = dict(sources)
    rebound_observed["checkpoints/v3.py"] = rebound_observed[
        "checkpoints/v3.py"
    ].replace(
        """    hf_descriptor = HFCompatibilityDescriptor.from_dict(
        _read_json(directory / "hf_descriptor.json")
    )""",
        "    hf_descriptor = expected_hf_descriptor",
    )
    assert audit_hf_authority_fixture(rebound_observed) == (
        "hf_checkpoint_descriptor_validation_bypassed",
    )

    swallowed_mismatch = dict(sources)
    swallowed_mismatch["checkpoints/v3.py"] = swallowed_mismatch[
        "checkpoints/v3.py"
    ].replace(
        """    if hf_descriptor != expected_hf_descriptor:
        raise CheckpointValidationError(
            "checkpoint_hf_descriptor_mismatch",
            "checkpoint HF descriptor does not match caller expectation",
        )""",
        """    try:
        if hf_descriptor != expected_hf_descriptor:
            raise CheckpointValidationError(
                "checkpoint_hf_descriptor_mismatch",
                "checkpoint HF descriptor does not match caller expectation",
            )
    except CheckpointValidationError:
        pass""",
    )
    assert audit_hf_authority_fixture(swallowed_mismatch) == (
        "hf_checkpoint_descriptor_validation_bypassed",
    )

    swallowed_return_then_reraise = dict(sources)
    swallowed_return_then_reraise["checkpoints/v3.py"] = swallowed_return_then_reraise[
        "checkpoints/v3.py"
    ].replace(
        """    if hf_descriptor != expected_hf_descriptor:
        raise CheckpointValidationError(
            "checkpoint_hf_descriptor_mismatch",
            "checkpoint HF descriptor does not match caller expectation",
        )""",
        """    try:
        if hf_descriptor != expected_hf_descriptor:
            raise CheckpointValidationError(
                "checkpoint_hf_descriptor_mismatch",
                "checkpoint HF descriptor does not match caller expectation",
            )
    except CheckpointValidationError:
        return None
        raise""",
    )
    assert audit_hf_authority_fixture(swallowed_return_then_reraise) == (
        "hf_checkpoint_descriptor_validation_bypassed",
    )

    finally_swallows_reraise = dict(sources)
    finally_swallows_reraise["checkpoints/v3.py"] = finally_swallows_reraise[
        "checkpoints/v3.py"
    ].replace(
        """    if hf_descriptor != expected_hf_descriptor:
        raise CheckpointValidationError(
            "checkpoint_hf_descriptor_mismatch",
            "checkpoint HF descriptor does not match caller expectation",
        )""",
        """    try:
        if hf_descriptor != expected_hf_descriptor:
            raise CheckpointValidationError(
                "checkpoint_hf_descriptor_mismatch",
                "checkpoint HF descriptor does not match caller expectation",
            )
    except CheckpointValidationError:
        try:
            raise
        finally:
            return None""",
    )
    assert audit_hf_authority_fixture(finally_swallows_reraise) == (
        "hf_checkpoint_descriptor_validation_bypassed",
    )

    for handler in (
        "except:",
        "except ValueError:",
        "except Exception:",
        "except BaseException:",
        "except (CheckpointValidationError, RuntimeError):",
        "except (ValueError, RuntimeError):",
    ):
        broadly_swallowed = dict(sources)
        broadly_swallowed["checkpoints/v3.py"] = broadly_swallowed[
            "checkpoints/v3.py"
        ].replace(
            """    if hf_descriptor != expected_hf_descriptor:
        raise CheckpointValidationError(
            "checkpoint_hf_descriptor_mismatch",
            "checkpoint HF descriptor does not match caller expectation",
        )""",
            f"""    try:
        if hf_descriptor != expected_hf_descriptor:
            raise CheckpointValidationError(
                "checkpoint_hf_descriptor_mismatch",
                "checkpoint HF descriptor does not match caller expectation",
            )
    {handler}
        pass""",
        )
        assert audit_hf_authority_fixture(broadly_swallowed) == (
            "hf_checkpoint_descriptor_validation_bypassed",
        )

    irrelevant_handler = dict(sources)
    irrelevant_handler["checkpoints/v3.py"] = irrelevant_handler[
        "checkpoints/v3.py"
    ].replace(
        """    if hf_descriptor != expected_hf_descriptor:
        raise CheckpointValidationError(
            "checkpoint_hf_descriptor_mismatch",
            "checkpoint HF descriptor does not match caller expectation",
        )""",
        """    try:
        if hf_descriptor != expected_hf_descriptor:
            raise CheckpointValidationError(
                "checkpoint_hf_descriptor_mismatch",
                "checkpoint HF descriptor does not match caller expectation",
            )
    except RuntimeError:
        pass""",
    )
    assert audit_hf_authority_fixture(irrelevant_handler) == ()

    lifecycle = dict(sources)
    lifecycle["steps/jax_loop.py"] = lifecycle["steps/jax_loop.py"].replace(
        "expected_hf_descriptor=self.hf_descriptor",
        "expected_hf_descriptor=None",
    )
    assert audit_hf_authority_fixture(lifecycle) == ("hf_checkpoint_lifecycle_bypass",)

    replay = dict(sources)
    replay["validation/p3_11_9_replay/runner_jax.py"] = replay[
        "validation/p3_11_9_replay/runner_jax.py"
    ].replace("lifecycle.hf_descriptor", "foreign_descriptor")
    assert audit_hf_authority_fixture(replay) == ("hf_replay_non_lifecycle_descriptor",)

    report = dict(sources)
    report["learning/run_report.py"] = report["learning/run_report.py"].replace(
        "validate_hf_descriptor_match", "compare_unchecked_descriptor"
    )
    assert audit_hf_authority_fixture(report) == (
        "hf_report_descriptor_validation_bypassed",
    )

    report_operands = dict(sources)
    report_operands["learning/run_report.py"] = report_operands[
        "learning/run_report.py"
    ].replace(
        "validate_hf_descriptor_match(executed_descriptor, summary.descriptor)",
        "validate_hf_descriptor_match(foreign_descriptor, foreign_summary)",
    )
    assert audit_hf_authority_fixture(report_operands) == (
        "hf_report_descriptor_validation_bypassed",
    )

    dead_report_validation = dict(sources)
    dead_report_validation["learning/run_report.py"] = dead_report_validation[
        "learning/run_report.py"
    ].replace(
        "        validate_hf_descriptor_match(executed_descriptor, summary.descriptor)",
        "        if False:\n"
        "            validate_hf_descriptor_match("
        "executed_descriptor, summary.descriptor)",
    )
    assert audit_hf_authority_fixture(dead_report_validation) == (
        "hf_report_descriptor_validation_bypassed",
    )

    early_return_report = dict(sources)
    early_return_report["learning/run_report.py"] = early_return_report[
        "learning/run_report.py"
    ].replace(
        "        validate_hf_descriptor_match(executed_descriptor, summary.descriptor)",
        "        return\n"
        "        validate_hf_descriptor_match(executed_descriptor, summary.descriptor)",
    )
    assert audit_hf_authority_fixture(early_return_report) == (
        "hf_report_descriptor_validation_bypassed",
    )

    for condition in ("True", "may_return"):
        conditional_return_report = dict(sources)
        conditional_return_report["learning/run_report.py"] = conditional_return_report[
            "learning/run_report.py"
        ].replace(
            "        validate_hf_descriptor_match("
            "executed_descriptor, summary.descriptor)",
            f"        if {condition}:\n"
            "            return\n"
            "        validate_hf_descriptor_match("
            "executed_descriptor, summary.descriptor)",
        )
        assert audit_hf_authority_fixture(conditional_return_report) == (
            "hf_report_descriptor_validation_bypassed",
        )

    match_return_report = dict(sources)
    match_return_report["learning/run_report.py"] = match_return_report[
        "learning/run_report.py"
    ].replace(
        "        validate_hf_descriptor_match(executed_descriptor, summary.descriptor)",
        "        match summary:\n"
        "            case _:\n"
        "                return\n"
        "        validate_hf_descriptor_match("
        "executed_descriptor, summary.descriptor)",
    )
    assert audit_hf_authority_fixture(match_return_report) == (
        "hf_report_descriptor_validation_bypassed",
    )


@pytest.mark.jax
def test_foundation_report_bytes_are_deterministic_and_public_checker_detects_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(ROOT)
    report = build_foundation_audit(ROOT)
    generated = _bytes(report)
    assert generated == _bytes(build_foundation_audit(ROOT))
    recorded = json.loads(
        (ROOT / "docs/FOUNDATION_AUDIT_CLOSURE_REPORT.json").read_text()
    )
    assert recorded["schema_version"] == SCHEMA_VERSION
    corrupted = tmp_path / "FOUNDATION_AUDIT_CLOSURE_REPORT.json"
    corrupted.write_bytes(generated[:-1] + b" ")
    assert main(["--check-recorded", "--recorded", str(corrupted)]) == 1


def test_foundation_rejects_invalid_p312b_receipt_without_jax(tmp_path: Path) -> None:
    receipt = json.loads(
        (ROOT / "docs/P3_12B_HF_DESCRIPTOR_AUTHORITY_RECEIPT.json").read_text()
    )
    receipt["status"] = "fail"
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "P3_12B_HF_DESCRIPTOR_AUTHORITY_RECEIPT.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )
    assert not _p312b_recorded_evidence_current(tmp_path)


def test_foundation_rejects_stale_p312b_descriptor_attestation_without_jax(
    tmp_path: Path,
) -> None:
    receipt = json.loads(
        (ROOT / "docs/P3_12B_HF_DESCRIPTOR_AUTHORITY_RECEIPT.json").read_text()
    )
    shutil.copytree(ROOT / "src", tmp_path / "src")
    shutil.copytree(ROOT / "tests", tmp_path / "tests")
    docs = tmp_path / "docs"
    docs.mkdir()
    shutil.copyfile(
        ROOT / "docs/RADJAX_DEVELOPMENT_ROADMAP.md",
        docs / "RADJAX_DEVELOPMENT_ROADMAP.md",
    )
    script = "\n".join(
        (
            "import importlib.abc",
            "import sys",
            "from pathlib import Path",
            "class BlockJax(importlib.abc.MetaPathFinder):",
            "    def find_spec(self, fullname, path=None, target=None):",
            "        if fullname in {'jax', 'jaxlib'} or fullname.startswith(",
            "('jax.', 'jaxlib.')):",
            "            raise ModuleNotFoundError(",
            "'simulated missing JAX: ' + fullname)",
            "        return None",
            "sys.meta_path.insert(0, BlockJax())",
            "from radjax_student.validation import foundation_audit_closure",
            "report = foundation_audit_closure.build_foundation_audit(Path.cwd())",
            "assert report.status == 'fail', report.blockers",
            "assert report.blockers == ('p312b_recorded_evidence_stale',)",
        )
    )
    mutations = (
        lambda payload: payload.__setitem__("descriptor_digest", "0" * 64),
        lambda payload: payload.__setitem__(
            "checkpoint_hf_descriptor_digest", "0" * 64
        ),
        lambda payload: payload.__setitem__("replay_hf_evidence_digest", "0" * 64),
        lambda payload: payload.__setitem__("report_hf_evidence_digest", "0" * 64),
        lambda payload: payload.__setitem__("dependency_audit_digest", "0" * 64),
        lambda payload: payload.__setitem__("evidence_digest", "0" * 64),
        lambda payload: payload["positive_proof_results"][0].__setitem__(
            "evidence_digest", "0" * 64
        ),
        lambda payload: payload["adversarial_case_results"][0].__setitem__(
            "observed_boundary", "foreign_boundary"
        ),
    )
    for mutate in mutations:
        stale = json.loads(json.dumps(receipt))
        mutate(stale)
        (docs / "P3_12B_HF_DESCRIPTOR_AUTHORITY_RECEIPT.json").write_text(
            json.dumps(stale), encoding="utf-8"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=tmp_path,
            env={**os.environ, "PYTHONPATH": str(tmp_path / "src")},
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr


def test_foundation_audit_remains_jax_free_when_jax_imports_are_blocked() -> None:
    script = "\n".join(
        (
            "import importlib.abc",
            "import sys",
            "from pathlib import Path",
            "class BlockJax(importlib.abc.MetaPathFinder):",
            "    def find_spec(self, fullname, path=None, target=None):",
            "        if fullname in {'jax', 'jaxlib'} or fullname.startswith("
            "('jax.', 'jaxlib.')):",
            "            raise ModuleNotFoundError("
            "'simulated missing optional JAX: ' + fullname)",
            "        return None",
            "sys.meta_path.insert(0, BlockJax())",
            "from radjax_student.validation.foundation_audit_closure "
            "import build_foundation_audit",
            "report = build_foundation_audit(Path.cwd())",
            "assert report.status == 'pass', report.blockers",
        )
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_runtime_import_and_local_test_support_are_hermetic() -> None:
    script = "\n".join(
        (
            "import importlib.util",
            "import pathlib",
            "import sys",
            "import radjax_student.runtime.callables",
            "assert not any("
            "name.startswith('radjax_student.steps') for name in sys.modules"
            ")",
            "spec = importlib.util.find_spec('tests.support.linear_objective')",
            "assert spec is not None and spec.origin is not None",
            "assert pathlib.Path(spec.origin).resolve().is_relative_to("
            "pathlib.Path.cwd() / 'tests'"
            ")",
        )
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert importlib.util.find_spec("radjax_student.losses") is None
