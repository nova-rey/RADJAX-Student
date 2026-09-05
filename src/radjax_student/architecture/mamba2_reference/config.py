"""Static configuration for the pinned Mamba-2 reference plugin.

The structural values in this module are the M2 reference profile.  Vocabulary
size and host chunk capacity are deliberately configuration inputs, rather
than part of the mathematical identity of the profile.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from radjax_student.architecture.errors import ArchitectureContractError
from radjax_student.architecture.models import ArchitectureConfig
from radjax_student.contracts import (
    HFSpecialTokenIdentity,
    HFTokenizerIdentity,
    HFVocabularyIdentity,
)

MAMBA2_REFERENCE_ARCHITECTURE_ID = "radjax.architecture.mamba2_reference"
MAMBA2_REFERENCE_ARCHITECTURE_VERSION = 1
MAMBA2_REFERENCE_VOCABULARY_SIZE = 16
MAMBA2_REFERENCE_CONTEXT_LENGTH = 4
MAMBA2_REFERENCE_D_MODEL = 8
MAMBA2_REFERENCE_LAYER_COUNT = 2
MAMBA2_REFERENCE_EXPAND = 2
MAMBA2_REFERENCE_D_INNER = 16
MAMBA2_REFERENCE_D_SSM = 16
MAMBA2_REFERENCE_NHEADS = 4
MAMBA2_REFERENCE_HEAD_DIM = 4
MAMBA2_REFERENCE_D_STATE = 4
MAMBA2_REFERENCE_D_CONV = 4
MAMBA2_REFERENCE_NGROUPS = 1
MAMBA2_REFERENCE_CHUNK_SIZE = 4
MAMBA2_REFERENCE_DTYPE = "float32"
MAMBA2_REFERENCE_BATCH_SIZE = 1
MAMBA2_REFERENCE_D_IN_PROJ = 44
MAMBA2_REFERENCE_CONV_DIM = 24
MAMBA2_REFERENCE_DT_LIMIT = {"min": 0.0, "max": "UNBOUNDED"}

_STRUCTURAL_KEYS = (
    "batch_size",
    "chunk_size",
    "d_conv",
    "d_in_proj",
    "d_inner",
    "d_model",
    "d_ssm",
    "d_state",
    "dt_limit",
    "expand",
    "headdim",
    "n_layer",
    "ngroups",
    "nheads",
    "vocab_size",
)


@dataclass(frozen=True)
class Mamba2EquationShape:
    """Dimensions used by the pinned upstream Mamba-2 equations."""

    vocabulary_size: int
    context_length: int
    d_model: int = MAMBA2_REFERENCE_D_MODEL
    n_layer: int = MAMBA2_REFERENCE_LAYER_COUNT
    expand: int = MAMBA2_REFERENCE_EXPAND
    d_inner: int = MAMBA2_REFERENCE_D_INNER
    d_ssm: int = MAMBA2_REFERENCE_D_SSM
    nheads: int = MAMBA2_REFERENCE_NHEADS
    headdim: int = MAMBA2_REFERENCE_HEAD_DIM
    d_state: int = MAMBA2_REFERENCE_D_STATE
    d_conv: int = MAMBA2_REFERENCE_D_CONV
    ngroups: int = MAMBA2_REFERENCE_NGROUPS
    chunk_size: int = MAMBA2_REFERENCE_CHUNK_SIZE
    dtype: str = MAMBA2_REFERENCE_DTYPE
    batch_size: int = MAMBA2_REFERENCE_BATCH_SIZE

    def __post_init__(self) -> None:
        values = (
            self.vocabulary_size,
            self.context_length,
            self.d_model,
            self.n_layer,
            self.expand,
            self.d_inner,
            self.d_ssm,
            self.nheads,
            self.headdim,
            self.d_state,
            self.d_conv,
            self.ngroups,
            self.chunk_size,
            self.batch_size,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in values
        ):
            raise ValueError("Mamba-2 dimensions must be positive integers")
        if self.dtype != MAMBA2_REFERENCE_DTYPE:
            raise ValueError("Mamba-2 reference profile requires float32")
        if self.d_inner != self.expand * self.d_model:
            raise ValueError("d_inner must equal expand * d_model")
        if self.d_ssm != self.d_inner:
            raise ValueError("the reference profile requires d_ssm == d_inner")
        if self.d_ssm != self.nheads * self.headdim:
            raise ValueError("d_ssm must equal nheads * headdim")
        if self.ngroups > self.d_ssm:
            raise ValueError("ngroups cannot exceed d_ssm")

    @property
    def conv_dim(self) -> int:
        return self.d_ssm + 2 * self.ngroups * self.d_state

    @property
    def d_in_proj(self) -> int:
        return 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads

    def model_config(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "chunk_size": self.chunk_size,
            "d_conv": self.d_conv,
            "d_in_proj": self.d_in_proj,
            "d_inner": self.d_inner,
            "d_model": self.d_model,
            "d_ssm": self.d_ssm,
            "d_state": self.d_state,
            "dt_limit": dict(MAMBA2_REFERENCE_DT_LIMIT),
            "expand": self.expand,
            "headdim": self.headdim,
            "n_layer": self.n_layer,
            "ngroups": self.ngroups,
            "nheads": self.nheads,
            "vocab_size": self.vocabulary_size,
        }


def equation_shape(config: ArchitectureConfig) -> Mamba2EquationShape:
    if not isinstance(config, ArchitectureConfig):
        raise ArchitectureContractError(
            "architecture_config_invalid", "configuration must be ArchitectureConfig"
        )
    try:
        model = dict(config.model_config)
        if set(model) != set(_STRUCTURAL_KEYS):
            raise ValueError("configuration must declare every Mamba-2 dimension")
        if config.vocab_size != model["vocab_size"]:
            raise ValueError("vocab_size must match ArchitectureConfig.vocab_size")
        shape = Mamba2EquationShape(
            vocabulary_size=config.vocab_size,
            context_length=config.sequence_length,
            d_model=model["d_model"],
            n_layer=model["n_layer"],
            expand=model["expand"],
            d_inner=model["d_inner"],
            d_ssm=model["d_ssm"],
            nheads=model["nheads"],
            headdim=model["headdim"],
            d_state=model["d_state"],
            d_conv=model["d_conv"],
            ngroups=model["ngroups"],
            chunk_size=model["chunk_size"],
            batch_size=model["batch_size"],
            dtype=config.dtype_intent,
        )
        if model["d_in_proj"] != shape.d_in_proj:
            raise ValueError("d_in_proj does not match the source contraction")
        if model["dt_limit"] != MAMBA2_REFERENCE_DT_LIMIT:
            raise ValueError("dt_limit must use the finite JSON unbounded encoding")
    except (TypeError, ValueError, KeyError) as exc:
        raise ArchitectureContractError(
            "architecture_config_invalid",
            "configuration does not satisfy the Mamba-2 reference dimension map",
        ) from exc
    return shape


@dataclass(frozen=True)
class Mamba2ReferenceConfig:
    """Reference structural profile with configurable language dimensions."""

    vocabulary_size: int = MAMBA2_REFERENCE_VOCABULARY_SIZE
    context_length: int = MAMBA2_REFERENCE_CONTEXT_LENGTH
    d_model: int = MAMBA2_REFERENCE_D_MODEL
    n_layer: int = MAMBA2_REFERENCE_LAYER_COUNT
    expand: int = MAMBA2_REFERENCE_EXPAND
    d_inner: int = MAMBA2_REFERENCE_D_INNER
    d_ssm: int = MAMBA2_REFERENCE_D_SSM
    nheads: int = MAMBA2_REFERENCE_NHEADS
    headdim: int = MAMBA2_REFERENCE_HEAD_DIM
    d_state: int = MAMBA2_REFERENCE_D_STATE
    d_conv: int = MAMBA2_REFERENCE_D_CONV
    ngroups: int = MAMBA2_REFERENCE_NGROUPS
    chunk_size: int = MAMBA2_REFERENCE_CHUNK_SIZE
    dtype: str = MAMBA2_REFERENCE_DTYPE
    batch_size: int = MAMBA2_REFERENCE_BATCH_SIZE

    def __post_init__(self) -> None:
        expected = (
            MAMBA2_REFERENCE_D_MODEL,
            MAMBA2_REFERENCE_LAYER_COUNT,
            MAMBA2_REFERENCE_EXPAND,
            MAMBA2_REFERENCE_D_INNER,
            MAMBA2_REFERENCE_D_SSM,
            MAMBA2_REFERENCE_NHEADS,
            MAMBA2_REFERENCE_HEAD_DIM,
            MAMBA2_REFERENCE_D_STATE,
            MAMBA2_REFERENCE_D_CONV,
            MAMBA2_REFERENCE_NGROUPS,
            MAMBA2_REFERENCE_CHUNK_SIZE,
            MAMBA2_REFERENCE_DTYPE,
            MAMBA2_REFERENCE_BATCH_SIZE,
        )
        actual = (
            self.d_model,
            self.n_layer,
            self.expand,
            self.d_inner,
            self.d_ssm,
            self.nheads,
            self.headdim,
            self.d_state,
            self.d_conv,
            self.ngroups,
            self.chunk_size,
            self.dtype,
            self.batch_size,
        )
        if actual != expected:
            raise ValueError("Mamba-2 structural profile is frozen")
        Mamba2EquationShape(self.vocabulary_size, self.context_length)

    def to_architecture_config(self) -> ArchitectureConfig:
        return ArchitectureConfig(
            architecture_id=MAMBA2_REFERENCE_ARCHITECTURE_ID,
            vocab_size=self.vocabulary_size,
            sequence_length=self.context_length,
            dtype_intent=self.dtype,
            model_config=Mamba2EquationShape(
                self.vocabulary_size,
                self.context_length,
                d_model=self.d_model,
                n_layer=self.n_layer,
                expand=self.expand,
                d_inner=self.d_inner,
                d_ssm=self.d_ssm,
                nheads=self.nheads,
                headdim=self.headdim,
                d_state=self.d_state,
                d_conv=self.d_conv,
                ngroups=self.ngroups,
                chunk_size=self.chunk_size,
                batch_size=self.batch_size,
            ).model_config(),
        )


def reference_config() -> Mamba2ReferenceConfig:
    return Mamba2ReferenceConfig()


def reference_architecture_config() -> ArchitectureConfig:
    return reference_config().to_architecture_config()


def configurable_architecture_config(
    vocabulary_size: int,
    context_length: int,
    *,
    tokenizer: HFTokenizerIdentity,
    vocabulary: HFVocabularyIdentity,
    special_tokens: HFSpecialTokenIdentity,
) -> ArchitectureConfig:
    """Build a profile with neutral language identities and variable V/T."""

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
    config = Mamba2ReferenceConfig(
        vocabulary_size, context_length
    ).to_architecture_config()
    return ArchitectureConfig(
        architecture_id=config.architecture_id,
        schema_version=config.schema_version,
        model_config=config.model_config,
        vocab_size=config.vocab_size,
        sequence_length=config.sequence_length,
        dtype_intent=config.dtype_intent,
        metadata={
            "hf_language_identities": {
                "tokenizer": tokenizer.to_dict(),
                "vocabulary": vocabulary.to_dict(),
                "special_tokens": special_tokens.to_dict(),
            }
        },
    )


def language_identities(
    config: ArchitectureConfig,
) -> tuple[HFTokenizerIdentity, HFVocabularyIdentity, HFSpecialTokenIdentity] | None:
    """Return validated neutral language identities, if configured."""

    equation_shape(config)
    if config == reference_architecture_config():
        return None
    identities = config.metadata.get("hf_language_identities")
    try:
        if set(config.metadata) != {"hf_language_identities"} or not isinstance(
            identities, Mapping
        ) or set(identities) != {"tokenizer", "vocabulary", "special_tokens"}:
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


def validate_mamba2_config(config: ArchitectureConfig) -> None:
    equation_shape(config)
    if config == reference_architecture_config():
        return
    language_identities(config)


validate_reference_config = validate_mamba2_config

__all__ = [name for name in globals() if name.startswith("MAMBA2_")] + [
    "Mamba2EquationShape",
    "Mamba2ReferenceConfig",
    "configurable_architecture_config",
    "equation_shape",
    "language_identities",
    "reference_architecture_config",
    "reference_config",
    "validate_mamba2_config",
    "validate_reference_config",
]
