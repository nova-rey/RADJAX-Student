"""Frozen static configuration for the tiny RWKV-7 reference domain."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from radjax_student.architecture.errors import ArchitectureContractError
from radjax_student.architecture.models import ArchitectureConfig
from radjax_student.contracts import (
    HFSpecialTokenIdentity,
    HFTokenizerIdentity,
    HFVocabularyIdentity,
)

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

_STRUCTURAL_KEYS = (
    "ffn_width",
    "head_count",
    "head_size",
    "hidden_size",
    "layer_count",
    "time_aaa_rank",
    "time_decay_rank",
    "time_gate_rank",
    "time_value_rank",
    "vocabulary_size",
)


@dataclass(frozen=True)
class RWKV7EquationShape:
    """Dimensions authorized by the pinned RWKV-7 equation contractions."""

    vocabulary_size: int
    context_length: int
    hidden_size: int
    layer_count: int
    head_size: int
    head_count: int
    ffn_width: int
    time_decay_rank: int
    time_aaa_rank: int
    time_value_rank: int
    time_gate_rank: int
    dtype: str = RWKV7_REFERENCE_DTYPE

    def __post_init__(self) -> None:
        values = (
            self.vocabulary_size,
            self.context_length,
            self.hidden_size,
            self.layer_count,
            self.head_size,
            self.head_count,
            self.ffn_width,
            self.time_decay_rank,
            self.time_aaa_rank,
            self.time_value_rank,
            self.time_gate_rank,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in values
        ):
            raise ValueError("RWKV-7 equation dimensions must be positive integers")
        if self.hidden_size != self.head_count * self.head_size:
            raise ValueError("hidden_size must equal head_count * head_size")
        if self.dtype != RWKV7_REFERENCE_DTYPE:
            raise ValueError("RWKV-7 equation shape requires float32")

    def model_config(self) -> dict[str, int]:
        return {
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
        }


def equation_shape(config: ArchitectureConfig) -> RWKV7EquationShape:
    """Validate and project every dimension used by the pinned equations."""

    if not isinstance(config, ArchitectureConfig):
        raise ArchitectureContractError(
            "architecture_config_invalid", "configuration must be ArchitectureConfig"
        )
    try:
        model = dict(config.model_config)
        if set(model) != set(_STRUCTURAL_KEYS):
            raise ValueError("configuration must declare every equation dimension")
        if config.vocab_size != model["vocabulary_size"]:
            raise ValueError("vocabulary_size must match ArchitectureConfig.vocab_size")
        shape = RWKV7EquationShape(
            vocabulary_size=config.vocab_size,
            context_length=config.sequence_length,
            hidden_size=model["hidden_size"],
            layer_count=model["layer_count"],
            head_size=model["head_size"],
            head_count=model["head_count"],
            ffn_width=model["ffn_width"],
            time_decay_rank=model["time_decay_rank"],
            time_aaa_rank=model["time_aaa_rank"],
            time_value_rank=model["time_value_rank"],
            time_gate_rank=model["time_gate_rank"],
            dtype=config.dtype_intent,
        )
    except (TypeError, ValueError, KeyError) as exc:
        raise ArchitectureContractError(
            "architecture_config_invalid",
            "configuration does not satisfy the RWKV-7 equation dimension map",
        ) from exc
    return shape


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


def configurable_architecture_config(
    vocabulary_size: int,
    context_length: int,
    *,
    tokenizer: HFTokenizerIdentity,
    vocabulary: HFVocabularyIdentity,
    special_tokens: HFSpecialTokenIdentity,
    hidden_size: int = RWKV7_REFERENCE_HIDDEN_SIZE,
    layer_count: int = RWKV7_REFERENCE_LAYER_COUNT,
    head_size: int = RWKV7_REFERENCE_HEAD_SIZE,
    head_count: int = RWKV7_REFERENCE_HEAD_COUNT,
    ffn_width: int = RWKV7_REFERENCE_FFN_WIDTH,
    time_decay_rank: int = RWKV7_REFERENCE_TIME_DECAY_RANK,
    time_aaa_rank: int = RWKV7_REFERENCE_TIME_AAA_RANK,
    time_value_rank: int = RWKV7_REFERENCE_TIME_VALUE_RANK,
    time_gate_rank: int = RWKV7_REFERENCE_TIME_GATE_RANK,
) -> ArchitectureConfig:
    """Build a stateless configurable RWKV-7 architecture projection.

    Only vocabulary and maximum host context are variable.  The complete
    neutral language identity is serialized into the generic configuration so
    the plugin never owns language-specific state or an opaque tokenizer hash.
    """

    if (
        not isinstance(vocabulary_size, int)
        or isinstance(vocabulary_size, bool)
        or vocabulary_size <= 0
    ):
        raise ValueError("vocabulary_size must be positive")
    if (
        not isinstance(context_length, int)
        or isinstance(context_length, bool)
        or context_length <= 0
    ):
        raise ValueError("context_length must be positive")
    if not isinstance(tokenizer, HFTokenizerIdentity):
        raise TypeError("tokenizer must be HFTokenizerIdentity")
    if not isinstance(vocabulary, HFVocabularyIdentity):
        raise TypeError("vocabulary must be HFVocabularyIdentity")
    if not isinstance(special_tokens, HFSpecialTokenIdentity):
        raise TypeError("special_tokens must be HFSpecialTokenIdentity")
    if vocabulary.vocabulary_size != vocabulary_size:
        raise ValueError("vocabulary identity size must match vocabulary_size")
    special_tokens.validate_for_vocabulary(vocabulary)
    shape = RWKV7EquationShape(
        vocabulary_size=vocabulary_size,
        context_length=context_length,
        hidden_size=hidden_size,
        layer_count=layer_count,
        head_size=head_size,
        head_count=head_count,
        ffn_width=ffn_width,
        time_decay_rank=time_decay_rank,
        time_aaa_rank=time_aaa_rank,
        time_value_rank=time_value_rank,
        time_gate_rank=time_gate_rank,
    )
    return ArchitectureConfig(
        architecture_id=RWKV7_REFERENCE_ARCHITECTURE_ID,
        vocab_size=vocabulary_size,
        sequence_length=context_length,
        dtype_intent=RWKV7_REFERENCE_DTYPE,
        model_config=shape.model_config(),
        metadata={
            "hf_language_identities": {
                "tokenizer": tokenizer.to_dict(),
                "vocabulary": vocabulary.to_dict(),
                "special_tokens": special_tokens.to_dict(),
            }
        },
    )


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


def language_identities(
    config: ArchitectureConfig,
) -> tuple[HFTokenizerIdentity, HFVocabularyIdentity, HFSpecialTokenIdentity] | None:
    """Return validated configurable language identities, if present."""

    if not isinstance(config, ArchitectureConfig):
        raise ArchitectureContractError(
            "architecture_config_invalid", "configuration must be ArchitectureConfig"
        )
    if config == reference_architecture_config():
        return None
    shape = equation_shape(config)
    expected_model = shape.model_config()
    identities = config.metadata.get("hf_language_identities")
    try:
        if (
            config.architecture_id != RWKV7_REFERENCE_ARCHITECTURE_ID
            or not isinstance(config.vocab_size, int)
            or config.vocab_size <= 0
            or not isinstance(config.sequence_length, int)
            or config.sequence_length <= 0
            or config.dtype_intent != RWKV7_REFERENCE_DTYPE
            or dict(config.model_config) != expected_model
            or set(config.metadata) != {"hf_language_identities"}
            or not isinstance(identities, Mapping)
            or set(identities) != {"tokenizer", "vocabulary", "special_tokens"}
        ):
            raise ValueError
        tokenizer = HFTokenizerIdentity.from_dict(identities["tokenizer"])
        vocabulary = HFVocabularyIdentity.from_dict(identities["vocabulary"])
        special_tokens = HFSpecialTokenIdentity.from_dict(identities["special_tokens"])
        if vocabulary.vocabulary_size != config.vocab_size:
            raise ValueError
        special_tokens.validate_for_vocabulary(vocabulary)
    except (TypeError, ValueError, KeyError) as exc:
        raise ArchitectureContractError(
            "architecture_config_invalid",
            "configuration lacks a complete valid neutral HF language identity",
        ) from exc
    return tokenizer, vocabulary, special_tokens


def validate_rwkv7_config(config: ArchitectureConfig) -> None:
    """Accept the frozen projection or a complete equation-authorized shape."""

    equation_shape(config)
    if config == reference_architecture_config():
        return
    language_identities(config)


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
    "RWKV7EquationShape",
    "equation_shape",
    "RWKV7ReferenceConfig",
    "configurable_architecture_config",
    "language_identities",
    "reference_architecture_config",
    "reference_config",
    "validate_reference_config",
    "validate_rwkv7_config",
]
