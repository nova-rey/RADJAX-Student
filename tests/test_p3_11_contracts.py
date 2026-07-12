from __future__ import annotations

import pytest

from radjax_student.contracts import (
    HFPreservationReference,
    JaxOptimizerStateDescriptor,
    ParameterTreeLayout,
    ParameterTreeLayoutEntry,
)


def _entry(path: str, keypath: tuple[str, ...], *, exportable: bool = False):
    return ParameterTreeLayoutEntry(
        path,
        keypath,
        (),
        "float32",
        "other",
        exportable=exportable,
        hf_distribution_key=f"hf.{path}" if exportable else None,
    )


def test_layout_bijects_logical_and_jax_keypaths():
    layout = ParameterTreeLayout("test", (_entry("b", ("b",)), _entry("a", ("a",))))
    assert layout.logical_paths == ("a", "b")
    assert layout.entry_for_jax_keypath(("a",)).logical_path == "a"


def test_layout_rejects_duplicate_jax_keypaths():
    with pytest.raises(ValueError, match="bijective"):
        ParameterTreeLayout("test", (_entry("a", ("same",)), _entry("b", ("same",))))


def test_exportable_layout_entry_requires_hf_key():
    with pytest.raises(ValueError, match="HF distribution"):
        ParameterTreeLayoutEntry("a", ("a",), (), "float32", "other", exportable=True)


def test_nonexportable_layout_entry_has_no_hf_placeholder():
    entry = _entry("internal.cache", ("internal", "cache"))
    assert not entry.exportable and entry.hf_distribution_key is None


def test_layout_digest_is_order_independent():
    first = ParameterTreeLayout("test", (_entry("b", ("b",)), _entry("a", ("a",))))
    second = ParameterTreeLayout("test", (_entry("a", ("a",)), _entry("b", ("b",))))
    assert first.digest() == second.digest()


def test_nested_layout_metadata_is_immutable_and_canonically_serializable():
    entry = ParameterTreeLayoutEntry(
        "quantized.weight",
        ("quantized", "weight"),
        (2,),
        "float32",
        "other",
        metadata={"quantization": {"scheme": "none"}, "axes": ["output"]},
    )
    layout = ParameterTreeLayout("test", (entry,))
    with pytest.raises(TypeError):
        entry.metadata["quantization"]["scheme"] = "int8"
    assert layout.to_dict()["entries"][0]["metadata"] == {
        "axes": ["output"],
        "quantization": {"scheme": "none"},
    }
    assert layout.digest() == ParameterTreeLayout("test", (entry,)).digest()


def test_hf_reference_requires_lifecycle_identity():
    reference = HFPreservationReference(
        "v1", "d", "model", "arch", "token", 4, "special", "layout", "config"
    )
    assert reference.to_dict()["architecture_id"] == "arch"


def test_jax_optimizer_descriptor_rejects_duplicate_state_paths():
    with pytest.raises(ValueError, match="unique"):
        JaxOptimizerStateDescriptor(
            "sgd", "optimizer.jax_execution_v1", "v1", "layout", (("step",), ("step",))
        )
