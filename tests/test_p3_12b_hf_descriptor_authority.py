from __future__ import annotations

import json
import runpy
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from radjax_student.architecture import (
    ArchitectureInitResult,
    ParameterCatalog,
    ParameterDescriptor,
)
from radjax_student.contracts import (
    HFArchitectureProjection,
    HFCompatibilityDescriptor,
    HFContractError,
    HFParameterProjection,
    HFPreservationReference,
    HFSpecialTokenIdentity,
    HFTokenizerIdentity,
    HFVocabularyIdentity,
    ParameterTreeLayout,
    ParameterTreeLayoutEntry,
    hf_digest,
)
from radjax_student.learning import RunHFSummary
from radjax_student.validation.p3_12b_hf_descriptor_authority import (
    implementation_audit,
)
from radjax_student.validation.p3_12b_hf_descriptor_authority.models import (
    ADVERSARIAL_CASE_COUNT,
    validate_receipt,
)

EXPECTED_PRODUCTION_OWNERS = (
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
)


def _descriptor() -> HFCompatibilityDescriptor:
    return HFCompatibilityDescriptor(
        "hf_compatibility_descriptor.v2",
        "test.hf_descriptor",
        1,
        "radjax_validation",
        hf_digest("config"),
        hf_digest("catalog"),
        hf_digest("layout"),
        HFTokenizerIdentity(
            "synthetic-tokenizer",
            "r1",
            hf_digest("tokenizer"),
            hf_digest("tokenizer-config"),
            "synthetic",
            hf_digest("normalization"),
            "synthetic",
        ),
        HFVocabularyIdentity(
            8, hf_digest("vocabulary"), hf_digest("tokens"), hf_digest("added"), None
        ),
        HFSpecialTokenIdentity(0, 1, 2, 3, None),
        (
            HFParameterProjection(
                "weight",
                ("weight",),
                (1,),
                "float32",
                "exportable",
                "model.weight",
                "identity",
            ),
        ),
        HFArchitectureProjection("synthetic", "linear", 1, 1, 8, 1, {}),
        ("hf_export_not_implemented",),
        "descriptive prose only",
    )


def test_reference_is_exact_descriptor_projection():
    descriptor = _descriptor()
    reference = descriptor.preservation_reference()
    assert reference == descriptor.preservation_reference()
    assert HFPreservationReference.from_dict(reference.to_dict()) == reference


def test_descriptor_parse_rejects_unknown_or_missing_fields():
    payload = _descriptor().to_dict()
    with pytest.raises(HFContractError, match="fields are invalid"):
        HFCompatibilityDescriptor.from_dict({**payload, "unknown": True})
    payload.pop("model_type")
    with pytest.raises(HFContractError, match="fields are invalid"):
        HFCompatibilityDescriptor.from_dict(payload)


def test_descriptor_digest_excludes_only_descriptive_prose():
    descriptor = _descriptor()
    assert (
        replace(descriptor, notes="different explanation").digest == descriptor.digest
    )
    assert replace(descriptor, model_type="foreign").digest != descriptor.digest


def test_exportability_requires_a_canonical_key_or_reason():
    with pytest.raises(HFContractError, match="requires exactly one HF key"):
        HFParameterProjection(
            "weight",
            ("weight",),
            (),
            "float32",
            "exportable",
            None,
            "identity",
        )
    with pytest.raises(HFContractError, match="requires a stable reason"):
        HFParameterProjection(
            "buffer",
            ("buffer",),
            (),
            "int32",
            "non_exportable",
            None,
            "identity",
        )


def test_report_carries_summary_not_full_descriptor():
    descriptor = _descriptor()
    summary = RunHFSummary(descriptor).to_dict()
    assert summary["descriptor_digest"] == descriptor.digest
    assert "parameter_projections" not in summary
    assert "tokenizer" not in summary


def test_initialization_rejects_reference_without_descriptor():
    descriptor = _descriptor()
    catalog = ParameterCatalog(
        "test.hf_descriptor", (ParameterDescriptor("weight", (1,), "float32"),)
    )
    layout = ParameterTreeLayout(
        "test.hf_descriptor",
        (
            ParameterTreeLayoutEntry(
                "weight",
                ("weight",),
                (1,),
                "float32",
                "other",
                ("whole_student",),
                exportable=True,
                hf_distribution_key="model.weight",
            ),
        ),
    )
    with pytest.raises(ValueError, match="hf_descriptor_missing"):
        ArchitectureInitResult(
            parameter_catalog=catalog,
            parameter_layout=layout,
            hf_reference=descriptor.preservation_reference(),
        )


@pytest.mark.jax
def test_literal_gate_inventory_has_exactly_77_distinct_experiments():
    from radjax_student.validation.p3_12b_hf_descriptor_authority.runner_jax import (
        SPECS,
    )

    assert len(SPECS) == ADVERSARIAL_CASE_COUNT
    assert len({spec.case_id for spec in SPECS}) == ADVERSARIAL_CASE_COUNT
    assert len({spec.experiment for spec in SPECS}) == ADVERSARIAL_CASE_COUNT


def test_v2_receipt_rejects_incomplete_adversarial_inventory():
    payload = json.loads(
        Path("docs/P3_12B_HF_DESCRIPTOR_AUTHORITY_RECEIPT.json").read_text()
    )
    assert validate_receipt(payload)["adversarial_case_count"] == 77
    payload["adversarial_case_count"] = 76
    with pytest.raises(ValueError, match="schema or status"):
        validate_receipt(payload)


@pytest.mark.jax
def test_normal_checker_rejects_schema_valid_stale_descriptor_receipt(tmp_path: Path):
    from radjax_student.validation.p3_12b_hf_descriptor_authority.__main__ import (
        main,
    )

    payload = json.loads(
        Path("docs/P3_12B_HF_DESCRIPTOR_AUTHORITY_RECEIPT.json").read_text()
    )
    payload["descriptor_digest"] = "0" * 64
    payload["checkpoint_hf_descriptor_digest"] = "0" * 64
    recorded = tmp_path / "P3_12B_HF_DESCRIPTOR_AUTHORITY_RECEIPT.json"
    recorded.write_text(json.dumps(payload), encoding="utf-8")
    assert main(["--check-recorded", "--recorded", str(recorded)]) == 1


def test_jax_free_implementation_audit_binds_literal_source_and_fixtures():
    fixtures = Path(__file__).parent / "fixtures" / "p3_12b_implementation_audit"
    expected_positives = ("first", "second")
    valid = implementation_audit.audit_gate_source(
        fixtures / "valid.py",
        expected_adversarial_case_ids=("first", "second"),
        expected_positive_case_ids=expected_positives,
    )
    assert valid.status == "pass"
    assert valid.adversarial_case_ids == ("first", "second")
    assert valid.positive_case_ids == expected_positives
    assert type(valid).from_dict(valid.to_dict()) == valid

    missing = implementation_audit.audit_gate_source(
        fixtures / "missing_experiment.py",
        expected_adversarial_case_ids=("first", "second"),
        expected_positive_case_ids=expected_positives,
    )
    translated = implementation_audit.audit_gate_source(
        fixtures / "expected_translation.py",
        expected_adversarial_case_ids=("first", "second"),
        expected_positive_case_ids=expected_positives,
    )
    wrong_positive_order = implementation_audit.audit_gate_source(
        fixtures / "wrong_positive_order.py",
        expected_adversarial_case_ids=("first", "second"),
        expected_positive_case_ids=expected_positives,
    )
    assert [item.code for item in missing.blockers] == ["wrong_adversarial_count"]
    assert [item.code for item in translated.blockers] == [
        "forbidden_expected_translation"
    ]
    assert [item.code for item in wrong_positive_order.blockers] == [
        "positive_inventory_mismatch"
    ]


def test_production_import_audit_rejects_gate_imports_from_every_protected_owner(
    tmp_path: Path,
) -> None:
    assert implementation_audit._PRODUCTION_OWNERS == set(EXPECTED_PRODUCTION_OWNERS)
    for index, owner in enumerate(EXPECTED_PRODUCTION_OWNERS):
        directory = tmp_path / "src" / "radjax_student" / owner
        directory.mkdir(parents=True)
        source = (
            "from radjax_student.validation import implementation_audit\n"
            if index % 2 == 0
            else (
                "from radjax_student.validation.p3_12b_hf_descriptor_authority "
                "import models\n"
            )
        )
        (directory / "gate_import.py").write_text(source, encoding="utf-8")
    blockers: list[implementation_audit.HFImplementationAuditBlocker] = []
    implementation_audit._audit_production_imports(tmp_path, blockers)
    assert [(item.code, item.detail) for item in blockers] == [
        ("production_imports_gate_code", f"{owner}/gate_import.py")
        for owner in EXPECTED_PRODUCTION_OWNERS
    ]


def test_production_import_audit_rejects_cli_gate_import_exactly(
    tmp_path: Path,
) -> None:
    cli = tmp_path / "src" / "radjax_student" / "cli"
    cli.mkdir(parents=True)
    (cli / "gate_import.py").write_text(
        "from radjax_student.validation import implementation_audit\n",
        encoding="utf-8",
    )
    blockers: list[implementation_audit.HFImplementationAuditBlocker] = []
    implementation_audit._audit_production_imports(tmp_path, blockers)
    assert [(item.code, item.detail) for item in blockers] == [
        ("production_imports_gate_code", "cli/gate_import.py"),
    ]


def test_production_import_audit_rejects_constructed_dynamic_gate_import(
    tmp_path: Path,
) -> None:
    cli = tmp_path / "src" / "radjax_student" / "cli"
    cli.mkdir(parents=True)
    (cli / "gate_import.py").write_text(
        "import importlib\n"
        "importlib.import_module(\n"
        "    'radjax_student.validation.p3_12b_'\n"
        "    + 'hf_descriptor_authority.implementation_audit'\n"
        ")\n",
        encoding="utf-8",
    )
    blockers: list[implementation_audit.HFImplementationAuditBlocker] = []
    implementation_audit._audit_production_imports(tmp_path, blockers)
    assert [(item.code, item.detail) for item in blockers] == [
        ("production_imports_gate_code", "cli/gate_import.py"),
    ]


def test_production_import_audit_rejects_fully_split_dynamic_gate_import(
    tmp_path: Path,
) -> None:
    """The all-source AST audit must not rely on raw protected-marker text."""
    cli = tmp_path / "src" / "radjax_student" / "cli"
    cli.mkdir(parents=True)
    (cli / "gate_import.py").write_text(
        "import importlib\n"
        "load = importlib.import_module\n"
        "load(\n"
        "    'radjax_student.validation.p3_12b_'\n"
        "    + 'hf_descriptor_'\n"
        "    + 'authority.implementation_'\n"
        "    + 'audit'\n"
        ")\n",
        encoding="utf-8",
    )
    blockers: list[implementation_audit.HFImplementationAuditBlocker] = []
    implementation_audit._audit_production_imports(tmp_path, blockers)
    assert [(item.code, item.detail) for item in blockers] == [
        ("production_imports_gate_code", "cli/gate_import.py"),
    ]


@pytest.mark.parametrize(
    "reflection_source",
    (
        "reflect = object.__getattribute__\n"
        "load = reflect(importlib, 'import_module')\n",
        "fetch = dict.__getitem__\nload = fetch(importlib.__dict__, 'import_module')\n",
        "table = importlib.__dict__\n"
        "fetch = table.get\n"
        "load = fetch('import_module')\n",
        "reflect = getattr\nload = reflect(importlib, 'import_module')\n",
        "getter_type = object\n"
        "load = getter_type.__getattribute__(importlib, 'import_module')\n",
        "reflect, = (object.__getattribute__,)\n"
        "load = reflect(importlib, 'import_module')\n",
        "base = __builtins__\n"
        "members = base if isinstance(base, dict) else vars(base)\n"
        "fetch = dict.__getitem__\n"
        "load = fetch(members, '__' + 'import__')\n",
        "op = __import__('operator')\n"
        "load = op.getitem(importlib.__dict__, 'import_' + 'module')\n",
        "reflectors = {'load': getattr}\n"
        "reflect = reflectors['load']\n"
        "load = reflect(importlib, 'import_module')\n",
        "fetch = dict.get\nload = fetch(importlib.__dict__, 'import_module')\n",
        "reflect = getattr.__call__\nload = reflect(importlib, 'import_module')\n",
        "import functools\n"
        "reflect = functools.partial(getattr)\n"
        "load = reflect(importlib, 'import_module')\n",
        "fetch = vars(importlib).get\nload = fetch('import_module')\n",
        "def factory():\n    return getattr\n"
        "reflect = factory()\n"
        "load = reflect(importlib, 'import_module')\n",
        "il = __import__('importlib')\n"
        "get_module = getattr(il, 'import_module')\n"
        "op = get_module('operator')\n"
        "load = op.getitem(il.__dict__, 'import_module')\n",
        "def identity(value):\n    return value\n"
        "reflect = identity(getattr)\n"
        "load = reflect(importlib, 'import_module')\n",
        "reflect = getattr if True else print\n"
        "load = reflect(importlib, 'import_module')\n",
        "mapping_type = dict\n"
        "fetch = mapping_type.get\n"
        "load = fetch(importlib.__dict__, 'import_module')\n",
        "fetch = type({}).get\nload = fetch(importlib.__dict__, 'import_module')\n",
        "reflect = [getattr if True else print][0]\n"
        "load = reflect(importlib, 'import_module')\n",
        "def factory(reflect=getattr if True else print):\n"
        "    return reflect(importlib, 'import_module')\n"
        "load = factory()\n",
        "def factory(mapping_type=dict if True else dict):\n"
        "    return mapping_type.get(importlib.__dict__, 'import_module')\n"
        "load = factory()\n",
        "holder = {'module': importlib}['module']\n"
        "load = getattr(holder, 'import_' + 'module')\n",
        "holder = {'module': importlib}['module']\n"
        "member = 'import_module'\n"
        "load = getattr(holder, member)\n",
        "mapping_type = type({})\n"
        "fetch = mapping_type.get\n"
        "load = fetch(importlib.__dict__, 'import_module')\n",
        "def map_factory():\n    return dict\n"
        "mapping_type = map_factory()\n"
        "fetch = mapping_type.get\n"
        "load = fetch(importlib.__dict__, 'import_module')\n",
        "mapping_type = (lambda: dict)()\n"
        "fetch = mapping_type.get\n"
        "load = fetch(importlib.__dict__, 'import_module')\n",
        "def identity(value):\n    return value\n"
        "mapping_type = identity(dict)\n"
        "fetch = mapping_type.get\n"
        "load = fetch(importlib.__dict__, 'import_module')\n",
        "holder = [importlib].pop()\n"
        "member = 'import_' + 'module'\n"
        "load = getattr(holder, member)\n",
        "holder = next({'module': importlib}.values())\n"
        "load = getattr(holder, 'import_' + 'module')\n",
        "holder = next(importlib for _ in (0,))\n"
        "load = getattr(holder, 'import_' + 'module')\n",
        "def holder_factory():\n    return importlib\n"
        "holder = holder_factory()\n"
        "load = getattr(holder, 'import_' + 'module')\n",
        "class Holder: pass\nholder_object = Holder()\n"
        "holder_object.module = importlib\n"
        "load = getattr(holder_object.module, 'import_' + 'module')\n",
        "reflect = [getattr if True else print].pop()\n"
        "load = reflect(importlib, 'import_' + 'module')\n",
        "reflect = {'reflect': getattr}.pop('reflect')\n"
        "load = reflect(importlib, 'import_' + 'module')\n",
        "reflect = next({'reflect': getattr}.values())\n"
        "load = reflect(importlib, 'import_' + 'module')\n",
        "mapping_type = {}.__class__\nfetch = mapping_type.get\n"
        "load = fetch(importlib.__dict__, 'import_' + 'module')\n",
        "mapping_type = type(dict())\nfetch = mapping_type.get\n"
        "load = fetch(importlib.__dict__, 'import_' + 'module')\n",
        "mapping_type = [dict].pop()\nfetch = mapping_type.get\n"
        "load = fetch(importlib.__dict__, 'import_' + 'module')\n",
        "mapping_type = next({'mapping': dict}.values())\n"
        "fetch = mapping_type.get\n"
        "load = fetch(importlib.__dict__, 'import_' + 'module')\n",
        "holder = [importlib].copy().pop()\n"
        "load = getattr(holder, 'import_' + 'module')\n",
        "holder = iter([importlib]).__next__()\n"
        "load = getattr(holder, 'import_' + 'module')\n",
        "class Holder: pass\nholder = Holder()\n"
        "setattr(holder, 'module', importlib)\n"
        "load = getattr(holder, 'import_' + 'module')\n",
        "class Holder: pass\nholder = Holder()\n"
        "holder.__dict__['module'] = importlib\n"
        "load = getattr(holder, 'module')\n",
        "class Holder:\n    def __call__(self):\n        return importlib\n"
        "holder = Holder()()\n"
        "load = getattr(holder, 'import_' + 'module')\n",
        "reflect = [getattr].copy().pop()\n"
        "load = reflect(importlib, 'import_' + 'module')\n",
        "mapping_type = dict.fromkeys(('mapping',), dict)['mapping']\n"
        "fetch = mapping_type.get\n"
        "load = fetch(importlib.__dict__, 'import_' + 'module')\n",
        "holder = [importlib].__reversed__().__next__()\n"
        "load = getattr(holder, 'import_' + 'module')\n",
        "mapping_type = type({}).mro()[0]\n"
        "fetch = mapping_type.get\n"
        "load = fetch(importlib.__dict__, 'import_' + 'module')\n",
        "class Holder: pass\nholder = Holder()\n"
        "field = 'module'\nsetattr(holder, field, importlib)\n"
        "load = getattr(holder, 'module')\n",
        "class Holder: pass\nholder = Holder()\n"
        "holder.__dict__.update({'module': importlib})\n"
        "load = getattr(holder, 'module')\n",
        "holder = ((importlib,) * 1)[0]\nload = getattr(holder, 'import_module')\n",
        "holder = ({'module': importlib} | {})['module']\n"
        "load = getattr(holder, 'import_module')\n",
        "holder = (alias := importlib)\nload = getattr(holder, 'import_module')\n",
        "from collections import deque\nholder = deque([importlib]).popleft()\n"
        "load = getattr(holder, 'import_module')\n",
        "reflect = (((getattr if True else print),) * 1)[0]\n"
        "load = reflect(importlib, 'import_module')\n",
        "reflect = ({'r': getattr if True else print} | {})['r']\n"
        "load = reflect(importlib, 'import_module')\n",
        "reflect = (alias := (getattr if True else print))\n"
        "load = reflect(importlib, 'import_module')\n",
        "mapping_type = type({}).__mro__[0]\nfetch = mapping_type.get\n"
        "load = fetch(importlib.__dict__, 'import_module')\n",
        "mapping_type = ((dict,) * 1)[0]\nfetch = mapping_type.get\n"
        "load = fetch(importlib.__dict__, 'import_module')\n",
        "mapping_type = ({'d': dict} | {})['d']\nfetch = mapping_type.get\n"
        "load = fetch(importlib.__dict__, 'import_module')\n",
        "from collections import deque\nmapping_type = deque([dict]).popleft()\n"
        "fetch = mapping_type.get\nload = fetch(importlib.__dict__, 'import_module')\n",
    ),
)
def test_production_import_audit_rejects_reflection_alias_gate_import(
    tmp_path: Path, reflection_source: str
) -> None:
    cli = tmp_path / "src" / "radjax_student" / "cli"
    cli.mkdir(parents=True)
    (cli / "gate_import.py").write_text(
        "import importlib\n" + reflection_source + "load(\n"
        "    'radjax_student.validation.p3_12b_'\n"
        "    + 'hf_descriptor_'\n"
        "    + 'authority.implementation_'\n"
        "    + 'audit'\n"
        ")\n",
        encoding="utf-8",
    )
    blockers: list[implementation_audit.HFImplementationAuditBlocker] = []
    implementation_audit._audit_production_imports(tmp_path, blockers)
    assert [(item.code, item.detail) for item in blockers] == [
        ("production_imports_gate_code", "cli/gate_import.py"),
    ]


def test_production_import_audit_rejects_indirect_dynamic_gate_import(
    tmp_path: Path,
) -> None:
    cli = tmp_path / "src" / "radjax_student" / "cli"
    cli.mkdir(parents=True)
    (cli / "gate_import.py").write_text(
        "getattr(__import__('importlib'), 'import_module')(\n"
        "    'radjax_student.validation.p3_12b_'\n"
        "    + 'hf_descriptor_authority.implementation_audit'\n"
        ")\n",
        encoding="utf-8",
    )
    blockers: list[implementation_audit.HFImplementationAuditBlocker] = []
    implementation_audit._audit_production_imports(tmp_path, blockers)
    assert [(item.code, item.detail) for item in blockers] == [
        ("production_imports_gate_code", "cli/gate_import.py"),
    ]


def test_production_import_audit_rejects_runpy_gate_execution(tmp_path: Path) -> None:
    cli = tmp_path / "src" / "radjax_student" / "cli"
    cli.mkdir(parents=True)
    (cli / "gate_import.py").write_text(
        "import runpy\n"
        "runpy.run_module(\n"
        "    'radjax_student.validation.p3_12b_hf_descriptor_authority'\n"
        ")\n",
        encoding="utf-8",
    )
    blockers: list[implementation_audit.HFImplementationAuditBlocker] = []
    implementation_audit._audit_production_imports(tmp_path, blockers)
    assert [(item.code, item.detail) for item in blockers] == [
        ("production_imports_gate_code", "cli/gate_import.py"),
    ]


@pytest.mark.parametrize(
    "source",
    (
        "import importlib\n"
        "runner = importlib.import_module('runpy')\n"
        "runner.run_module('radjax_student.validation.p3_12b_hf_descriptor_authority')\n",
        "runner = __import__('runpy')\n"
        "runner.run_module('radjax_student.validation.p3_12b_hf_descriptor_authority')\n",
        "import importlib as loader\n"
        "runner = loader.import_module('runpy')\n"
        "runner.run_module('radjax_student.validation.p3_12b_hf_descriptor_authority')\n",
        "from importlib import import_module as acquire\n"
        "runner = acquire('runpy')\n"
        "runner.run_module('radjax_student.validation.p3_12b_hf_descriptor_authority')\n",
        "import importlib as loader\n"
        "runner = loader.import_module(name='runpy')\n"
        "runner.run_module('radjax_student.validation.p3_12b_hf_descriptor_authority')\n",
        "from importlib import import_module as acquire\n"
        "runner = acquire(name='runpy')\n"
        "runner.run_module('radjax_student.validation.p3_12b_hf_descriptor_authority')\n",
    ),
)
def test_production_import_audit_rejects_indirect_runpy_execution(
    tmp_path: Path, source: str
) -> None:
    cli = tmp_path / "src" / "radjax_student" / "cli"
    cli.mkdir(parents=True)
    (cli / "gate_import.py").write_text(source, encoding="utf-8")
    blockers: list[implementation_audit.HFImplementationAuditBlocker] = []
    implementation_audit._audit_production_imports(tmp_path, blockers)
    assert [(item.code, item.detail) for item in blockers] == [
        ("production_imports_gate_code", "cli/gate_import.py"),
    ]


@pytest.mark.parametrize(
    "source",
    (
        "import pkgutil\npkgutil.resolve_name('radjax_student.validation.fixture')\n",
        "import pydoc\npydoc.locate('radjax_student.validation.fixture')\n",
        "import importlib.util\n"
        "spec = importlib.util.find_spec('radjax_student.validation.fixture')\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n",
    ),
)
def test_production_import_audit_rejects_module_execution_authorities(
    tmp_path: Path, source: str
) -> None:
    cli = tmp_path / "src" / "radjax_student" / "cli"
    cli.mkdir(parents=True)
    (cli / "gate_import.py").write_text(source, encoding="utf-8")
    blockers: list[implementation_audit.HFImplementationAuditBlocker] = []
    implementation_audit._audit_production_imports(tmp_path, blockers)
    assert [(item.code, item.detail) for item in blockers] == [
        ("production_imports_gate_code", "cli/gate_import.py"),
    ]


@pytest.mark.parametrize(
    "source",
    (
        "import importlib\nfrom functools import partial\n"
        "load = partial(\n"
        "    importlib.import_module,\n"
        "    'radjax_student.validation.p3_12b_hf_descriptor_authority.inventory',\n"
        ")\nload()\n",
        "import importlib\n"
        "load = (lambda value: value)(importlib.import_module)\n"
        "load('radjax_student.validation.p3_12b_hf_descriptor_authority.inventory')\n",
        "import importlib\nclass Holder:\n    load = importlib.import_module\n"
        "Holder.load('radjax_student.validation.p3_12b_hf_descriptor_authority.inventory')\n",
        "import importlib\nholder = {}\n"
        "holder.update(load=importlib.import_module)\n"
        "holder['load']('radjax_student.validation.p3_12b_hf_descriptor_authority.inventory')\n",
        "import importlib\ndef run(*, load=importlib.import_module):\n"
        "    return load(\n"
        "        'radjax_student.validation.p3_12b_hf_descriptor_authority.inventory'\n"
        "    )\n"
        "run()\n",
        "import importlib.util\n"
        "find = importlib.util.find_spec\n"
        "create = importlib.util.module_from_spec\n"
        "spec = find(\n"
        "    'radjax_student.validation.p3_12b_hf_descriptor_authority.inventory'\n"
        ")\n"
        "module = create(spec)\n"
        "execute = spec.loader.exec_module\nexecute(module)\n",
    ),
)
def test_production_import_audit_rejects_import_primitive_transport(
    tmp_path: Path, source: str
) -> None:
    cli = tmp_path / "src" / "radjax_student" / "cli"
    cli.mkdir(parents=True)
    (cli / "gate_import.py").write_text(source, encoding="utf-8")
    blockers: list[implementation_audit.HFImplementationAuditBlocker] = []
    implementation_audit._audit_production_imports(tmp_path, blockers)
    assert [(item.code, item.detail) for item in blockers] == [
        ("production_imports_gate_code", "cli/gate_import.py"),
    ]


def test_production_import_audit_rejects_multistage_dynamic_gate_import(
    tmp_path: Path,
) -> None:
    cli = tmp_path / "src" / "radjax_student" / "cli"
    cli.mkdir(parents=True)
    (cli / "gate_import.py").write_text(
        "loader = __import__('importlib')\n"
        "load = getattr(loader, 'import_module')\n"
        "load(\n"
        "    'radjax_student.validation.p3_12b_'\n"
        "    + 'hf_descriptor_authority.implementation_audit'\n"
        ")\n",
        encoding="utf-8",
    )
    blockers: list[implementation_audit.HFImplementationAuditBlocker] = []
    implementation_audit._audit_production_imports(tmp_path, blockers)
    assert [(item.code, item.detail) for item in blockers] == [
        ("production_imports_gate_code", "cli/gate_import.py"),
    ]


def test_production_import_audit_rejects_mapping_dynamic_gate_import(
    tmp_path: Path,
) -> None:
    cli = tmp_path / "src" / "radjax_student" / "cli"
    cli.mkdir(parents=True)
    (cli / "gate_import.py").write_text(
        "import importlib\n"
        "load = vars(importlib)['import_module']\n"
        "load(\n"
        "    'radjax_student.validation.p3_12b_'\n"
        "    + 'hf_descriptor_authority.implementation_audit'\n"
        ")\n",
        encoding="utf-8",
    )
    blockers: list[implementation_audit.HFImplementationAuditBlocker] = []
    implementation_audit._audit_production_imports(tmp_path, blockers)
    assert [(item.code, item.detail) for item in blockers] == [
        ("production_imports_gate_code", "cli/gate_import.py"),
    ]


def test_production_import_audit_rejects_bound_mapping_dynamic_gate_import(
    tmp_path: Path,
) -> None:
    cli = tmp_path / "src" / "radjax_student" / "cli"
    cli.mkdir(parents=True)
    (cli / "gate_import.py").write_text(
        "loader = __import__('builtins')\n"
        "members = loader.__dict__\n"
        "load = members['__import__']\n"
        "load(\n"
        "    'radjax_student.validation.p3_12b_'\n"
        "    + 'hf_descriptor_authority.implementation_audit'\n"
        ")\n",
        encoding="utf-8",
    )
    blockers: list[implementation_audit.HFImplementationAuditBlocker] = []
    implementation_audit._audit_production_imports(tmp_path, blockers)
    assert [(item.code, item.detail) for item in blockers] == [
        ("production_imports_gate_code", "cli/gate_import.py"),
    ]


def test_production_import_audit_rejects_global_dynamic_gate_import(
    tmp_path: Path,
) -> None:
    cli = tmp_path / "src" / "radjax_student" / "cli"
    cli.mkdir(parents=True)
    (cli / "gate_import.py").write_text(
        "load = globals()['__import__']\n"
        "load(\n"
        "    'radjax_student.validation.p3_12b_'\n"
        "    + 'hf_descriptor_authority.implementation_audit'\n"
        ")\n",
        encoding="utf-8",
    )
    blockers: list[implementation_audit.HFImplementationAuditBlocker] = []
    implementation_audit._audit_production_imports(tmp_path, blockers)
    assert [(item.code, item.detail) for item in blockers] == [
        ("production_imports_gate_code", "cli/gate_import.py"),
    ]


def test_production_import_audit_rejects_builtins_chain_gate_import(
    tmp_path: Path,
) -> None:
    cli = tmp_path / "src" / "radjax_student" / "cli"
    cli.mkdir(parents=True)
    (cli / "gate_import.py").write_text(
        "base = globals()['__builtins__']\n"
        "load = getattr(base, '__import__')\n"
        "load(\n"
        "    'radjax_student.validation.p3_12b_'\n"
        "    + 'hf_descriptor_authority.implementation_audit'\n"
        ")\n",
        encoding="utf-8",
    )
    blockers: list[implementation_audit.HFImplementationAuditBlocker] = []
    implementation_audit._audit_production_imports(tmp_path, blockers)
    assert [(item.code, item.detail) for item in blockers] == [
        ("production_imports_gate_code", "cli/gate_import.py"),
    ]


def test_production_import_audit_rejects_operator_dynamic_gate_import(
    tmp_path: Path,
) -> None:
    cli = tmp_path / "src" / "radjax_student" / "cli"
    cli.mkdir(parents=True)
    (cli / "gate_import.py").write_text(
        "import importlib, operator\n"
        "load = operator.getitem(importlib.__dict__, 'import_module')\n"
        "load(\n"
        "    'radjax_student.validation.p3_12b_'\n"
        "    + 'hf_descriptor_authority.implementation_audit'\n"
        ")\n",
        encoding="utf-8",
    )
    blockers: list[implementation_audit.HFImplementationAuditBlocker] = []
    implementation_audit._audit_production_imports(tmp_path, blockers)
    assert [(item.code, item.detail) for item in blockers] == [
        ("production_imports_gate_code", "cli/gate_import.py"),
    ]


def test_production_import_audit_rejects_dictionary_dynamic_gate_import(
    tmp_path: Path,
) -> None:
    cli = tmp_path / "src" / "radjax_student" / "cli"
    cli.mkdir(parents=True)
    (cli / "gate_import.py").write_text(
        "import importlib\n"
        "load = dict.__getitem__(importlib.__dict__, 'import_module')\n"
        "load(\n"
        "    'radjax_student.validation.p3_12b_'\n"
        "    + 'hf_descriptor_authority.implementation_audit'\n"
        ")\n",
        encoding="utf-8",
    )
    blockers: list[implementation_audit.HFImplementationAuditBlocker] = []
    implementation_audit._audit_production_imports(tmp_path, blockers)
    assert [(item.code, item.detail) for item in blockers] == [
        ("production_imports_gate_code", "cli/gate_import.py"),
    ]


def test_p312b3_anti_cheat_source_fixtures_execute_with_stable_blockers(tmp_path):
    fixtures = Path(__file__).parent / "fixtures" / "p3_12b_implementation_audit"
    source = runpy.run_path(fixtures / "anti_cheat_sources.py")
    expected = {
        "missing_adversarial_function": "missing_adversarial_function",
        "duplicate_adversarial_function": "duplicate_adversarial_function",
        "wrong_adversarial_count": "wrong_adversarial_count",
        "reordered_adversarial_ids": "reordered_adversarial_ids",
        "lambda_canonical_experiment": "lambda_canonical_experiment",
        "partial_canonical_experiment": "partial_canonical_experiment",
        "loop_generated_experiment": "loop_generated_experiment",
        "filesystem_discovered_inventory": "filesystem_discovered_inventory",
        "experiment_parameter_case_id": "adversarial_signature_metadata",
        "experiment_parameter_expected_code": "adversarial_signature_metadata",
        "experiment_parameter_spec": "adversarial_signature_metadata",
        "matches_expected": "forbidden_expected_translation",
        "hf_prefix": "forbidden_prefix_family_match",
        "checkpoint_prefix": "forbidden_prefix_family_match",
        "replay_prefix": "forbidden_prefix_family_match",
        "report_prefix": "forbidden_prefix_family_match",
        "wrong_positive_order": "positive_inventory_mismatch",
        "missing_positive": "positive_inventory_mismatch",
        "unexpected_positive": "positive_inventory_mismatch",
    }
    for name, code in expected.items():
        path = tmp_path / f"{name}.py"
        path.write_text(source["source"](name), encoding="utf-8")
        audit = implementation_audit.audit_gate_source(
            path,
            expected_adversarial_case_ids=("first", "second"),
            expected_positive_case_ids=("first", "second"),
        )
        assert code in {item.code for item in audit.blockers}, name


def test_p312b3_real_audit_round_trips_and_stays_jax_free():
    path = Path(
        "src/radjax_student/validation/p3_12b_hf_descriptor_authority/runner_jax.py"
    )
    audit = implementation_audit.audit_gate_source(path)
    assert audit.status == "pass"
    assert audit.adversarial_inventory_count == 77
    assert audit.positive_inventory_count == 22
    recorded = json.loads(
        Path("docs/P3_12B_HF_DESCRIPTOR_AUTHORITY_RECEIPT.json").read_text()
    )
    assert (
        audit.source_evidence_digest
        == recorded["implementation_audit"]["source_evidence_digest"]
    )
    assert (
        audit.implementation_audit_digest
        == recorded["implementation_audit"]["implementation_audit_digest"]
    )
    assert type(audit).from_dict(audit.to_dict()) == audit
    malformed = audit.to_dict()
    malformed["unknown"] = True
    with pytest.raises(ValueError, match="missing or unknown"):
        type(audit).from_dict(malformed)
    code = (
        "import sys; "
        "from radjax_student.validation.p3_12b_hf_descriptor_authority "
        "import implementation_audit; "
        "assert 'jax' not in sys.modules and 'jaxlib' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.jax
def test_lifecycle_rejects_independently_fabricated_reference():
    from radjax_student.validation.p3_11_9_replay.runner_jax import _new_lifecycle

    lifecycle = _new_lifecycle("eager", [])
    fabricated = HFPreservationReference.from_dict(
        {**lifecycle.hf_reference.to_dict(), "descriptor_digest": "0" * 64}
    )
    with pytest.raises(ValueError, match="not derived"):
        replace(lifecycle, hf_reference=fabricated)
