from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from radjax_student.runtime.errors import RuntimeIssue
from radjax_student.runtime.models import DeviceDescriptor


def normalize_devices(
    raw_devices: tuple[Any, ...],
    process_count: int | None,
    warnings: list[RuntimeIssue],
    blockers: list[RuntimeIssue],
) -> tuple[DeviceDescriptor, ...]:
    devices: list[DeviceDescriptor] = []
    memory_unknown: list[str] = []
    precision_unknown: list[str] = []
    for index, device in enumerate(raw_devices):
        try:
            platform = _optional_device_string(device, "platform")
            process_index = _optional_device_int(device, "process_index")
            local_hardware_id = _optional_device_scalar(
                device,
                "local_hardware_id",
            )
            process_label = (
                str(process_index) if process_index is not None else "unknown"
            )
            device_id = f"{platform or 'unknown'}:{process_label}:{index}"
            descriptor = DeviceDescriptor(
                device_id=device_id,
                platform=platform,
                device_kind=_optional_device_string(device, "device_kind"),
                process_index=process_index,
                local_hardware_id=local_hardware_id,
                memory_bytes=None,
                supported_precisions=(),
                metadata=_device_metadata(device),
            )
        except (TypeError, ValueError) as exc:
            blockers.append(
                RuntimeIssue.create(
                    "device_normalization_failed",
                    "visible JAX device could not be normalized",
                    device_index=index,
                    **exception_details(exc),
                )
            )
            continue
        if (
            process_count is not None
            and descriptor.process_index is not None
            and descriptor.process_index >= process_count
        ):
            blockers.append(
                RuntimeIssue.create(
                    "device_normalization_failed",
                    "device process index is outside observed process count",
                    device_id=descriptor.device_id,
                    process_index=descriptor.process_index,
                    process_count=process_count,
                )
            )
        devices.append(descriptor)
        memory_unknown.append(descriptor.device_id)
        precision_unknown.append(descriptor.device_id)
    if memory_unknown:
        warnings.append(
            RuntimeIssue.create(
                "device_memory_unknown",
                "device memory capacity is not exposed through the stable "
                "inspection boundary",
                device_ids=tuple(memory_unknown),
            )
        )
    if precision_unknown:
        warnings.append(
            RuntimeIssue.create(
                "device_precision_unknown",
                "device precision support was not proven by inspection",
                device_ids=tuple(precision_unknown),
            )
        )
    return tuple(devices)


def check_consistency(
    *,
    devices: tuple[DeviceDescriptor, ...],
    process_count: int | None,
    process_index: int | None,
    local_device_count: int | None,
    global_device_count: int | None,
    blockers: list[RuntimeIssue],
) -> None:
    if (
        process_count is not None
        and process_index is not None
        and process_index >= process_count
    ):
        blockers.append(
            RuntimeIssue.create(
                "device_normalization_failed",
                "process index is outside observed process count",
                process_count=process_count,
                process_index=process_index,
            )
        )
    if global_device_count is not None and global_device_count != len(devices):
        blockers.append(
            RuntimeIssue.create(
                "device_normalization_failed",
                "global device count does not match normalized inventory",
                global_device_count=global_device_count,
                normalized_device_count=len(devices),
            )
        )
    if (
        local_device_count is not None
        and global_device_count is not None
        and local_device_count > global_device_count
    ):
        blockers.append(
            RuntimeIssue.create(
                "device_normalization_failed",
                "local device count exceeds global device count",
                local_device_count=local_device_count,
                global_device_count=global_device_count,
            )
        )
    identifiers = [device.device_id for device in devices]
    if len(identifiers) != len(set(identifiers)):
        blockers.append(
            RuntimeIssue.create(
                "device_normalization_failed",
                "normalized device IDs are not unique",
            )
        )


def exception_details(exc: Exception) -> dict[str, str]:
    return {
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
    }


def _optional_device_string(device: Any, name: str) -> str | None:
    value = getattr(device, name, None)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise TypeError(f"device {name} must be a nonempty string")
    return value


def _optional_device_int(device: Any, name: str) -> int | None:
    value = getattr(device, name, None)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"device {name} must be a nonnegative integer")
    return value


def _optional_device_scalar(device: Any, name: str) -> str | int | None:
    value = getattr(device, name, None)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise TypeError(f"device {name} must be a string or integer")
    return value


def _device_metadata(device: Any) -> Mapping[str, Any]:
    reported_id = getattr(device, "id", None)
    if isinstance(reported_id, bool) or not isinstance(reported_id, (str, int)):
        return {}
    return {"jax_reported_device_id": reported_id}
