"""One-principal-defect JAX-free source fixtures for P3.12D audit coverage."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BadSourceFixture:
    name: str
    source: str
    expected_blocker: str


_FIXTURES: tuple[tuple[str, str, str], ...] = (
    (
        "second_binder",
        "def bind_runtime_callable_two(): pass",
        "callable_audit_competing_authority",
    ),
    (
        "binder_under_validation",
        "validation.assembly",
        "callable_audit_validation_binder",
    ),
    (
        "production_validation_import",
        "from radjax_student.validation import x",
        "callable_audit_production_validation_import",
    ),
    (
        "request_raw_callable",
        "raw_callable: Callable",
        "callable_audit_request_raw_callable",
    ),
    ("request_function", "function: object", "callable_audit_request_function"),
    (
        "request_caller_digest",
        "caller_identity_digest: str",
        "callable_audit_caller_digest",
    ),
    (
        "binding_caller_digest",
        "caller_source_digest: str",
        "callable_audit_caller_source_digest",
    ),
    ("id_identity", "value = id(function)", "callable_audit_unsafe_identity"),
    ("repr_identity", "value = repr(function)", "callable_audit_unsafe_identity"),
    ("hash_identity", "value = hash(function)", "callable_audit_unsafe_identity"),
    (
        "module_only",
        "module_only_identity = module",
        "callable_audit_module_only_identity",
    ),
    (
        "qualname_only",
        "qualname_only_identity = qualname",
        "callable_audit_qualname_only_identity",
    ),
    (
        "module_qualname_only",
        "module_qualname_only_identity = module + qualname",
        "callable_audit_module_qualname_only_identity",
    ),
    (
        "filename_only",
        "filename_only_identity = filename",
        "callable_audit_filename_only_identity",
    ),
    ("mtime", "mtime_identity = mtime", "callable_audit_mtime_identity"),
    (
        "validation_observed",
        "validation_observed_identity = value",
        "callable_audit_validation_identity",
    ),
    (
        "observer_expected",
        "def observe(expected_code): pass",
        "callable_audit_observer_expected_metadata",
    ),
    (
        "permissive_helper",
        "def _matches_expected(): pass",
        "callable_audit_permissive_matcher",
    ),
    (
        "prefix_matcher",
        "observed.startswith('execution_callable_')",
        "callable_audit_permissive_matcher",
    ),
    (
        "positive_missing",
        "missing_positive_inventory = True",
        "callable_audit_positive_inventory",
    ),
    (
        "adversarial_missing",
        "missing_adversarial_inventory = True",
        "callable_audit_adversarial_inventory",
    ),
    (
        "dynamic_adversary",
        "dynamic_adversary_generator = lambda: None",
        "callable_audit_dynamic_adversary",
    ),
    (
        "reused_adversary",
        "reused_adversarial_callable = handler",
        "callable_audit_reused_adversary",
    ),
    (
        "assembler_identity",
        "assembler_source_identity = source",
        "callable_audit_assembler_identity",
    ),
    (
        "backend_identity",
        "backend_source_identity = source",
        "callable_audit_backend_identity",
    ),
    (
        "static_omitted",
        "static_argument_omitted = True",
        "callable_audit_static_digest_missing",
    ),
    (
        "options_omitted",
        "compilation_options_omitted = True",
        "callable_audit_compilation_digest_missing",
    ),
    (
        "input_omitted",
        "input_signature_omitted = True",
        "callable_audit_input_signature_missing",
    ),
)

REQUIRED_BAD_SOURCE_FIXTURES: tuple[BadSourceFixture, ...] = tuple(
    BadSourceFixture(*item) for item in _FIXTURES
)
