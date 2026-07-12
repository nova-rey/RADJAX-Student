"""HF identity that survives training without becoming an export format."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HFPreservationReference:
    descriptor_schema_version: str
    descriptor_digest: str
    model_type: str
    architecture_id: str
    tokenizer_id: str
    vocabulary_size: int
    special_token_digest: str
    parameter_layout_digest: str
    architecture_config_digest: str

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value
            for value in (
                self.descriptor_schema_version,
                self.descriptor_digest,
                self.model_type,
                self.architecture_id,
                self.tokenizer_id,
                self.special_token_digest,
                self.parameter_layout_digest,
                self.architecture_config_digest,
            )
        ):
            raise ValueError("HF preservation identity fields must be nonempty")
        if not isinstance(self.vocabulary_size, int) or self.vocabulary_size <= 0:
            raise ValueError("vocabulary_size must be positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "descriptor_schema_version": self.descriptor_schema_version,
            "descriptor_digest": self.descriptor_digest,
            "model_type": self.model_type,
            "architecture_id": self.architecture_id,
            "tokenizer_id": self.tokenizer_id,
            "vocabulary_size": self.vocabulary_size,
            "special_token_digest": self.special_token_digest,
            "parameter_layout_digest": self.parameter_layout_digest,
            "architecture_config_digest": self.architecture_config_digest,
        }
