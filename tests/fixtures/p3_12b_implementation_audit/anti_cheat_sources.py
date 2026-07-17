"""Synthetic source fixtures executed by the JAX-free P3.12B.3 audit tests."""

_BASE = """
def _positive(*args): return args
def adversary_first(b): return b
def adversary_second(b): return b
_FUNCTIONS = (adversary_first, adversary_second)
positives = (_positive("first", "x", {}), _positive("second", "x", {}))
"""


def source(name: str) -> str:
    fixtures = {
        "missing_adversarial_function": _BASE.replace(
            "adversary_second)", "adversary_missing)"
        ),
        "duplicate_adversarial_function": _BASE.replace(
            "adversary_second)", "adversary_first)"
        ),
        "wrong_adversarial_count": _BASE.replace(", adversary_second", ""),
        "reordered_adversarial_ids": _BASE.replace(
            "adversary_first, adversary_second", "adversary_second, adversary_first"
        ),
        "lambda_canonical_experiment": _BASE.replace(
            "adversary_first, adversary_second", "lambda b: b, adversary_second"
        ),
        "partial_canonical_experiment": "from functools import partial\n"
        + _BASE.replace(
            "adversary_first, adversary_second",
            "partial(adversary_first), adversary_second",
        ),
        "loop_generated_experiment": _BASE
        + "\nfor name in ('x',):\n    def adversary_loop(b): return b\n",
        "filesystem_discovered_inventory": (
            "from pathlib import Path\n_FUNCTIONS = tuple(Path('.').iterdir())\n"
            "positives = ()\n"
        ),
        "experiment_parameter_case_id": _BASE.replace(
            "adversary_first(b)", "adversary_first(case_id)"
        ),
        "experiment_parameter_expected_code": _BASE.replace(
            "adversary_first(b)", "adversary_first(expected_code)"
        ),
        "experiment_parameter_spec": _BASE.replace(
            "adversary_first(b)", "adversary_first(spec)"
        ),
        "experiment_case_branch": _BASE
        + "\ncase_id = 'x'\ndef adversary_branch(b):\n    if case_id: return b\n",
        "mutation_dictionary_case_key": _BASE
        + (
            "\ncase_name = 'x'\nmutations = {case_name: 1}\n"
            "def adversary_keyed(b): return mutations[case_name]\n"
        ),
        "generic_run_case": _BASE
        + (
            "\ndef run_case(spec): return spec\n"
            "def adversary_generic(b): return run_case(b)\n"
        ),
        "reused_semantic_handler": _BASE.replace(
            "def adversary_second(b): return b", "def adversary_second(b): return b"
        ),
        "observer_parameter_expected_code": _BASE
        + "\ndef _observe(invocation, expected_code): return None\n",
        "observer_expected_global": _BASE
        + "\nexpected_code = 'x'\ndef _observe(invocation): return expected_code\n",
        "matches_expected": _BASE + "\ndef _matches_expected(value): return value\n",
        "hf_prefix": _BASE + "\ndef bad(observed): return observed.startswith('hf_')\n",
        "checkpoint_prefix": _BASE
        + "\ndef bad(observed): return observed.startswith('checkpoint_hf_')\n",
        "replay_prefix": _BASE
        + "\ndef bad(observed): return observed.startswith('replay_hf_')\n",
        "report_prefix": _BASE
        + "\ndef bad(observed): return observed.startswith('report_hf_')\n",
        "alias_dictionary": _BASE + "\nblocker_aliases = {'x': ('y',)}\n",
        "free_boundary": _BASE + "\ndef _run():\n    observed_boundary = 'free'\n",
        "copied_boundary": _BASE
        + "\ndef _run(spec):\n    observed_boundary = spec.intended_boundary\n",
        "receipt_passed": _BASE + "\ndef build_receipt(passed=True): return passed\n",
        "receipt_deterministic": _BASE
        + "\ndef build_receipt(deterministic=True): return deterministic\n",
        "missing_77_enforcement": _BASE,
        "missing_22_enforcement": _BASE,
        "decorative_digest": _BASE + "\ndef digest(): return 'functions-only'\n",
        "wrong_positive_order": _BASE.replace(
            '_positive("first", "x", {}), _positive("second", "x", {})',
            '_positive("second", "x", {}), _positive("first", "x", {})',
        ),
        "missing_positive": _BASE.replace(', _positive("second", "x", {})', ""),
        "unexpected_positive": _BASE.replace(
            '_positive("second", "x", {})', '_positive("third", "x", {})'
        ),
    }
    return fixtures[name]


ALL_FIXTURES = tuple(
    sorted(
        {
            "missing_adversarial_function",
            "duplicate_adversarial_function",
            "wrong_adversarial_count",
            "reordered_adversarial_ids",
            "lambda_canonical_experiment",
            "partial_canonical_experiment",
            "loop_generated_experiment",
            "filesystem_discovered_inventory",
            "experiment_parameter_case_id",
            "experiment_parameter_expected_code",
            "experiment_parameter_spec",
            "experiment_case_branch",
            "mutation_dictionary_case_key",
            "generic_run_case",
            "reused_semantic_handler",
            "observer_parameter_expected_code",
            "observer_expected_global",
            "matches_expected",
            "hf_prefix",
            "checkpoint_prefix",
            "replay_prefix",
            "report_prefix",
            "alias_dictionary",
            "free_boundary",
            "copied_boundary",
            "receipt_passed",
            "receipt_deterministic",
            "missing_77_enforcement",
            "missing_22_enforcement",
            "decorative_digest",
            "wrong_positive_order",
            "missing_positive",
            "unexpected_positive",
        }
    )
)
