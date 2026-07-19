"""Frozen static configuration for the tiny RWKV-7 reference domain."""

from __future__ import annotations

from dataclasses import dataclass

from radjax_student.architecture.errors import ArchitectureContractError
from radjax_student.architecture.models import ArchitectureConfig

RWKV7_REFERENCE_ARCHITECTURE_ID = "radjax.architecture.rwkv7_reference"
RWKV7_REFERENCE_ARCHITECTURE_VERSION = 1
RWKV7_REFERENCE_VOCABULARY_SIZE = 16
RWKV7_REFERENCE_HIDDEN_SIZE = 8
RWKV7_REFERENCE_LAYER_COUNT = 2
RWKV7_REFERENCE_HEAD_SIZE = 4
RWKV7_REFERENCE_HEAD_COUNT = 2
RWKV7_REFERENCE_FFN_WIDTH = 16
RWKV7_REFERENCE_CONTEXT_LENGTH = 4
RWKV7_REFERENCE_DTYPE = "float32"
RWKV7_REFERENCE_TIME_DECAY_RANK = 32
RWKV7_REFERENCE_TIME_AAA_RANK = 32
RWKV7_REFERENCE_TIME_VALUE_RANK = 32
RWKV7_REFERENCE_TIME_GATE_RANK = 32


@dataclass(frozen=True)
class RWKV7ReferenceConfig:
    """The sole P4.2 configuration; numerical initialization is deferred."""

    vocabulary_size: int = RWKV7_REFERENCE_VOCABULARY_SIZE
    hidden_size: int = RWKV7_REFERENCE_HIDDEN_SIZE
    layer_count: int = RWKV7_REFERENCE_LAYER_COUNT
    head_size: int = RWKV7_REFERENCE_HEAD_SIZE
    head_count: int = RWKV7_REFERENCE_HEAD_COUNT
    ffn_width: int = RWKV7_REFERENCE_FFN_WIDTH
    context_length: int = RWKV7_REFERENCE_CONTEXT_LENGTH
    dtype: str = RWKV7_REFERENCE_DTYPE
    time_decay_rank: int = RWKV7_REFERENCE_TIME_DECAY_RANK
    time_aaa_rank: int = RWKV7_REFERENCE_TIME_AAA_RANK
    time_value_rank: int = RWKV7_REFERENCE_TIME_VALUE_RANK
    time_gate_rank: int = RWKV7_REFERENCE_TIME_GATE_RANK

    def __post_init__(self) -> None:
        if (
            self.vocabulary_size,
            self.hidden_size,
            self.layer_count,
            self.head_size,
            self.head_count,
            self.ffn_width,
            self.context_length,
            self.dtype,
            self.time_decay_rank,
            self.time_aaa_rank,
            self.time_value_rank,
            self.time_gate_rank,
        ) != (
            RWKV7_REFERENCE_VOCABULARY_SIZE,
            RWKV7_REFERENCE_HIDDEN_SIZE,
            RWKV7_REFERENCE_LAYER_COUNT,
            RWKV7_REFERENCE_HEAD_SIZE,
            RWKV7_REFERENCE_HEAD_COUNT,
            RWKV7_REFERENCE_FFN_WIDTH,
            RWKV7_REFERENCE_CONTEXT_LENGTH,
            RWKV7_REFERENCE_DTYPE,
            RWKV7_REFERENCE_TIME_DECAY_RANK,
            RWKV7_REFERENCE_TIME_AAA_RANK,
            RWKV7_REFERENCE_TIME_VALUE_RANK,
            RWKV7_REFERENCE_TIME_GATE_RANK,
        ):
            raise ValueError("RWKV-7 reference configuration is frozen in P4.2")

    def to_architecture_config(self) -> ArchitectureConfig:
        return ArchitectureConfig(
            architecture_id=RWKV7_REFERENCE_ARCHITECTURE_ID,
            vocab_size=self.vocabulary_size,
            sequence_length=self.context_length,
            dtype_intent=self.dtype,
            model_config={
                "ffn_width": self.ffn_width,
                "head_count": self.head_count,
                "head_size": self.head_size,
                "hidden_size": self.hidden_size,
                "layer_count": self.layer_count,
                "time_aaa_rank": self.time_aaa_rank,
                "time_decay_rank": self.time_decay_rank,
                "time_gate_rank": self.time_gate_rank,
                "time_value_rank": self.time_value_rank,
                "vocabulary_size": self.vocabulary_size,
            },
        )


def reference_config() -> RWKV7ReferenceConfig:
    """Return the domain declared by the Phase 4 contract."""

    return RWKV7ReferenceConfig()


def reference_architecture_config() -> ArchitectureConfig:
    """Return the generic typed projection of the frozen configuration."""

    return RWKV7ReferenceConfig().to_architecture_config()


def validate_reference_config(config: ArchitectureConfig) -> None:
    """Reject every configuration outside the declared tiny reference domain."""

    expected = reference_architecture_config()
    if not isinstance(config, ArchitectureConfig) or config != expected:
        received = getattr(config, "architecture_id", None)
        raise ArchitectureContractError(
            "architecture_config_invalid",
            "configuration does not match the frozen RWKV-7 reference domain",
            details={
                "expected_architecture_id": RWKV7_REFERENCE_ARCHITECTURE_ID,
                "received_architecture_id": received,
            },
        )


__all__ = [
    "RWKV7_REFERENCE_ARCHITECTURE_ID",
    "RWKV7_REFERENCE_ARCHITECTURE_VERSION",
    "RWKV7_REFERENCE_CONTEXT_LENGTH",
    "RWKV7_REFERENCE_DTYPE",
    "RWKV7_REFERENCE_FFN_WIDTH",
    "RWKV7_REFERENCE_HEAD_COUNT",
    "RWKV7_REFERENCE_HEAD_SIZE",
    "RWKV7_REFERENCE_HIDDEN_SIZE",
    "RWKV7_REFERENCE_LAYER_COUNT",
    "RWKV7_REFERENCE_TIME_AAA_RANK",
    "RWKV7_REFERENCE_TIME_DECAY_RANK",
    "RWKV7_REFERENCE_TIME_GATE_RANK",
    "RWKV7_REFERENCE_TIME_VALUE_RANK",
    "RWKV7_REFERENCE_VOCABULARY_SIZE",
    "RWKV7ReferenceConfig",
    "reference_architecture_config",
    "reference_config",
    "validate_reference_config",
]
