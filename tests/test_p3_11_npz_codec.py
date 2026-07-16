"""Base-environment proofs for the deterministic NPZ codec."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest

from radjax_student.checkpoints.npz_codec import (
    read_deterministic_npz,
    write_deterministic_npz,
)


def test_c_and_fortran_order_arrays_have_identical_canonical_bytes(tmp_path: Path):
    c_order = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32, order="C")
    f_order = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32, order="F")
    first = write_deterministic_npz(tmp_path / "c.npz", {"weights": c_order})
    second = write_deterministic_npz(tmp_path / "f.npz", {"weights": f_order})
    assert first == second
    assert (tmp_path / "c.npz").read_bytes() == (tmp_path / "f.npz").read_bytes()
    assert np.array_equal(
        read_deterministic_npz(tmp_path / "f.npz", second)["weights"], c_order
    )


def test_structured_dtype_is_rejected(tmp_path: Path):
    value = np.asarray([(1, 2.0)], dtype=[("step", "i4"), ("value", "f4")])
    with pytest.raises(TypeError, match="structured dtypes"):
        write_deterministic_npz(tmp_path / "structured.npz", {"value": value})


def test_same_payload_written_twice_is_byte_identical(tmp_path: Path):
    tree = {"b": np.asarray((2, 3), dtype=np.int32), "a": np.asarray(1.0)}
    first = write_deterministic_npz(tmp_path / "one.npz", tree)
    second = write_deterministic_npz(tmp_path / "two.npz", tree)
    assert first == second
    assert (tmp_path / "one.npz").read_bytes() == (tmp_path / "two.npz").read_bytes()


def test_reader_rejects_a_real_fortran_order_member(tmp_path: Path):
    path = tmp_path / "canonical.npz"
    descriptor = write_deterministic_npz(
        path,
        {"weights": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)},
    )
    member = descriptor["leaves"][0]["member"]
    with zipfile.ZipFile(path, "r") as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    raw = io.BytesIO()
    np.lib.format.write_array(
        raw,
        np.asfortranarray(np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)),
        version=(1, 0),
        allow_pickle=False,
    )
    members[member] = raw.getvalue()
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(members):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, members[name])
    with pytest.raises(ValueError, match="not canonical"):
        read_deterministic_npz(path, descriptor)
