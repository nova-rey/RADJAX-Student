"""Deterministic project-owned ZIP_STORED codec for mapping array pytrees."""

from __future__ import annotations

import hashlib
import io
import json
import struct
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

NPZ_CODEC_VERSION = "radjax_deterministic_npz.v1"


def encode_keypath(keypath: tuple[str, ...]) -> str:
    """Encode mapping keypaths without separator or collision ambiguity."""

    if not keypath or any(not isinstance(part, str) or not part for part in keypath):
        raise ValueError("keypath must contain nonempty mapping keys")
    return (
        "k_"
        + "_".join(
            f"{len(part.encode('utf-8')):08x}{part.encode('utf-8').hex()}"
            for part in keypath
        )
        + ".npy"
    )


def decode_member_name(member_name: str) -> tuple[str, ...]:
    if not isinstance(member_name, str) or not member_name.startswith("k_"):
        raise ValueError("invalid deterministic NPZ member name")
    raw = member_name[2:]
    if not raw.endswith(".npy"):
        raise ValueError("invalid deterministic NPZ member name")
    raw = raw[:-4]
    parts: list[str] = []
    while raw:
        if len(raw) < 8:
            raise ValueError("invalid deterministic NPZ member name")
        size = int(raw[:8], 16)
        raw = raw[8:]
        encoded_size = size * 2
        if size == 0 or len(raw) < encoded_size:
            raise ValueError("invalid deterministic NPZ member name")
        value = bytes.fromhex(raw[:encoded_size]).decode("utf-8")
        parts.append(value)
        raw = raw[encoded_size:]
        if raw:
            if not raw.startswith("_"):
                raise ValueError("invalid deterministic NPZ member name")
            raw = raw[1:]
    return tuple(parts)


def descriptor_digest(descriptor: Mapping[str, Any]) -> str:
    return hashlib.sha256(_json_bytes(descriptor)).hexdigest()


def mapping_pytree_digest(tree: Mapping[str, Any]) -> str:
    """Return an in-memory identity for a canonical mapping-only pytree.

    The checkpoint writer and replay evidence use this exact encoder.  It
    includes descriptor identity *and* canonical array bytes, so it detects
    value changes without creating a temporary sidecar.
    """

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency boundary
        raise RuntimeError("numpy is required for checkpoint tensor payloads") from exc

    digest = hashlib.sha256()
    for keypath, value in sorted(_flatten_mapping(tree).items()):
        array = _canonical_array(np.asarray(value), np)
        member = encode_keypath(keypath)
        descriptor = _leaf_descriptor(keypath, member, array)
        descriptor_bytes = _json_bytes(descriptor)
        # Replay identity intentionally hashes only canonical logical bytes,
        # not an implementation-selected .npy header version.
        array_bytes = array.tobytes(order="C")
        digest.update(len(descriptor_bytes).to_bytes(8, "big"))
        digest.update(descriptor_bytes)
        digest.update(len(array_bytes).to_bytes(8, "big"))
        digest.update(array_bytes)
    return digest.hexdigest()


def write_deterministic_npz(path: Path, tree: Mapping[str, Any]) -> dict[str, Any]:
    """Write canonical little-endian .npy members in a fixed ZIP container."""

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency boundary
        raise RuntimeError("numpy is required for checkpoint tensor payloads") from exc

    leaves = _flatten_mapping(tree)
    members: dict[str, bytes] = {}
    descriptors: list[dict[str, Any]] = []
    for keypath, value in sorted(leaves.items()):
        member = encode_keypath(keypath)
        if member in members:
            raise ValueError("deterministic NPZ keypath encoding collision")
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError("object dtype and pickle are forbidden in NPZ payloads")
        if array.dtype.fields is not None:
            raise TypeError("structured dtypes are not supported in NPZ payloads")
        if array.dtype.byteorder == ">" or (
            array.dtype.byteorder == "=" and not np.little_endian
        ):
            array = array.astype(array.dtype.newbyteorder("<"), copy=False)
        if array.ndim:
            array = np.ascontiguousarray(array)
        members[member] = _canonical_npy_bytes(array)
        descriptors.append(_leaf_descriptor(keypath, member, array))
    descriptor = _descriptor(descriptors)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for member in sorted(members):
            info = zipfile.ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, members[member])
    return descriptor


def describe_mapping_pytree(tree: Mapping[str, Any]) -> dict[str, Any]:
    """Describe the canonical mapping-pytree identity without writing a sidecar.

    Architectures can declare a carry identity during initialization before a
    continuation checkpoint exists. The descriptor is identical to the one
    produced by :func:`write_deterministic_npz`; only the byte container is
    deferred to checkpoint ownership.
    """

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency boundary
        raise RuntimeError("numpy is required for checkpoint tensor payloads") from exc

    descriptors: list[dict[str, Any]] = []
    for keypath, value in sorted(_flatten_mapping(tree).items()):
        array = _canonical_array(np.asarray(value), np)
        descriptors.append(_leaf_descriptor(keypath, encode_keypath(keypath), array))
    return _descriptor(descriptors)


def read_deterministic_npz(path: Path, descriptor: Mapping[str, Any]) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency boundary
        raise RuntimeError("numpy is required for checkpoint tensor payloads") from exc
    if descriptor.get("codec") != NPZ_CODEC_VERSION:
        raise ValueError("unsupported deterministic NPZ codec")
    expected: dict[tuple[str, ...], tuple[str, tuple[int, ...], str]] = {}
    for leaf in descriptor.get("leaves", ()):
        keypath = tuple(leaf["keypath"])
        member = str(leaf["member"])
        if keypath in expected or member != encode_keypath(keypath):
            raise ValueError("invalid or colliding NPZ descriptor")
        expected[keypath] = (member, tuple(leaf["shape"]), str(leaf["dtype"]))
    result: dict[str, Any] = {}
    with zipfile.ZipFile(path, "r") as archive:
        if set(archive.namelist()) != {item[0] for item in expected.values()}:
            raise ValueError("NPZ members do not match the canonical descriptor")
        for keypath, (member, shape, dtype) in expected.items():
            with archive.open(member, "r") as stream:
                member_bytes = stream.read()
            array = np.lib.format.read_array(
                io.BytesIO(member_bytes), allow_pickle=False
            )
            if array.dtype.hasobject or array.dtype.fields is not None:
                raise ValueError("object and structured dtypes are not supported")
            if tuple(array.shape) != shape or str(array.dtype) != dtype:
                raise ValueError("NPZ leaf shape or dtype does not match descriptor")
            if member_bytes != _canonical_npy_bytes(_canonical_array(array, np)):
                raise ValueError("NPZ member bytes are not canonical")
            _set_mapping_leaf(result, keypath, array)
    return result


def _flatten_mapping(
    value: Mapping[str, Any], prefix: tuple[str, ...] = ()
) -> dict[tuple[str, ...], Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("tensor payload trees must be nonempty mappings")
    result: dict[tuple[str, ...], Any] = {}
    for key in sorted(value):
        if not isinstance(key, str) or not key:
            raise ValueError("tensor payload mapping keys must be nonempty strings")
        child = value[key]
        if isinstance(child, Mapping):
            result.update(_flatten_mapping(child, (*prefix, key)))
        else:
            result[(*prefix, key)] = child
    return result


def _canonical_array(array: Any, np: Any) -> Any:
    if array.dtype.hasobject:
        raise TypeError("object dtype and pickle are forbidden in NPZ payloads")
    if array.dtype.fields is not None:
        raise TypeError("structured dtypes are not supported in NPZ payloads")
    if array.dtype.byteorder == ">" or (
        array.dtype.byteorder == "=" and not np.little_endian
    ):
        array = array.astype(array.dtype.newbyteorder("<"), copy=False)
    return np.ascontiguousarray(array) if array.ndim else array


def _canonical_npy_bytes(array: Any) -> bytes:
    """Encode a v1 ``.npy`` member without NumPy-version-dependent headers."""

    shape = _npy_shape(array.shape)
    header = (
        "{'descr': "
        + repr(array.dtype.str)
        + ", 'fortran_order': False, 'shape': "
        + shape
        + ", }"
    )
    # NPY v1.0 aligns the complete preamble and ASCII header to 16 bytes.
    preamble_size = 10
    padding = (-((preamble_size + len(header) + 1) % 16)) % 16
    header_bytes = (header + (" " * padding) + "\n").encode("latin1")
    if len(header_bytes) > 0xFFFF:
        raise ValueError("canonical NPY header exceeds v1.0 size limit")
    prefix = b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header_bytes))
    return prefix + header_bytes + array.tobytes(order="C")


def _npy_shape(shape: tuple[int, ...]) -> str:
    if not shape:
        return "()"
    if len(shape) == 1:
        return f"({shape[0]},)"
    return "(" + ", ".join(str(value) for value in shape) + ")"


def _leaf_descriptor(
    keypath: tuple[str, ...], member: str, array: Any
) -> dict[str, Any]:
    return {
        "keypath": list(keypath),
        "member": member,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }


def _descriptor(leaves: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "jax_pytree_payload.v1",
        "codec": NPZ_CODEC_VERSION,
        "tree_kind": "mapping_only",
        "leaves": leaves,
    }


def _set_mapping_leaf(
    result: dict[str, Any], keypath: tuple[str, ...], value: Any
) -> None:
    branch = result
    for key in keypath[:-1]:
        branch = branch.setdefault(key, {})
    if keypath[-1] in branch:
        raise ValueError("duplicate NPZ descriptor keypath")
    branch[keypath[-1]] = value


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


__all__ = [
    "NPZ_CODEC_VERSION",
    "decode_member_name",
    "describe_mapping_pytree",
    "descriptor_digest",
    "encode_keypath",
    "mapping_pytree_digest",
    "read_deterministic_npz",
    "write_deterministic_npz",
]
