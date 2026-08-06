"""Canonical neutral HF language projection from a Contract binding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from radjax_student.contracts.hf import (
    HFSpecialTokenIdentity,
    HFTokenizerIdentity,
    HFVocabularyIdentity,
    hf_digest,
)

HF_LANGUAGE_PROJECTION_V1 = "hf_language_projection_v1"


@dataclass(frozen=True)
class HFLanguageProjectionV1:
    """HF identities derived solely from the public language binding descriptor."""

    schema_version: str
    canonical_binding_digest: str
    tokenizer: HFTokenizerIdentity
    vocabulary: HFVocabularyIdentity
    special_tokens: HFSpecialTokenIdentity

    def __post_init__(self) -> None:
        if self.schema_version != HF_LANGUAGE_PROJECTION_V1:
            raise ValueError("unsupported HF language projection schema")
        if not self.canonical_binding_digest.startswith("sha256:"):
            raise ValueError("language binding digest must be sha256")
        self.special_tokens.validate_for_vocabulary(self.vocabulary)


def project_hf_language_binding(descriptor: Any) -> HFLanguageProjectionV1:
    """Project Contract's generic v5 language descriptor without resource access."""

    tokenizer_data, vocabulary_data = descriptor.tokenizer, descriptor.vocabulary
    revision = tokenizer_data.get("revision")
    if not isinstance(revision, dict) or not isinstance(revision.get("value"), str):
        raise ValueError("language tokenizer revision is invalid")
    declarations = vocabulary_data.get("special_tokens")
    if not isinstance(declarations, list):
        raise ValueError("language special-token declarations are invalid")
    token_ids = {item.get("name"): item.get("token_id") for item in declarations}
    if set(token_ids) != {"eos", "pad"} or any(
        type(value) is not int for value in token_ids.values()
    ):
        raise ValueError("language EOS/PAD declarations are invalid")
    reserved = vocabulary_data.get("reserved_token_ids")
    if not isinstance(reserved, list) or any(
        type(value) is not int for value in reserved
    ):
        raise ValueError("language reserved token declarations are invalid")
    vocabulary_identity = _bare_digest(vocabulary_data.get("vocabulary_identity"))
    return HFLanguageProjectionV1(
        schema_version=HF_LANGUAGE_PROJECTION_V1,
        canonical_binding_digest=descriptor.canonical_binding_digest,
        tokenizer=HFTokenizerIdentity(
            tokenizer_id=_string(tokenizer_data, "implementation_id"),
            tokenizer_revision=revision["value"],
            tokenizer_content_digest=_bare_digest(revision["value"]),
            tokenizer_config_digest=_bare_digest(
                tokenizer_data.get("configuration_identity")
            ),
            tokenizer_family=_string(tokenizer_data, "family"),
            normalization_digest=_bare_digest(
                tokenizer_data.get("normalization_identity")
            ),
            identity_availability="embedded",
        ),
        vocabulary=HFVocabularyIdentity(
            vocabulary_size=_integer(vocabulary_data, "vocabulary_size"),
            vocabulary_content_digest=vocabulary_identity,
            token_to_id_digest=_bare_digest(
                vocabulary_data.get("vocabulary_map_digest")
            ),
            added_token_digest=hf_digest(
                {"added_tokens": vocabulary_data.get("added_tokens")}
            ),
            reserved_token_range=",".join(str(value) for value in reserved) or None,
        ),
        special_tokens=HFSpecialTokenIdentity(
            None, token_ids["eos"], token_ids["pad"], None, None
        ),
    )


def _bare_digest(value: object) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise ValueError("language identity must be sha256")
    return value.removeprefix("sha256:")


def _string(value: dict[str, Any], field: str) -> str:
    result = value.get(field)
    if not isinstance(result, str) or not result:
        raise ValueError(f"language {field} is invalid")
    return result


def _integer(value: dict[str, Any], field: str) -> int:
    result = value.get(field)
    if type(result) is not int or result <= 0:
        raise ValueError(f"language {field} is invalid")
    return result
