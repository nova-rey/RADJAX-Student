from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from radjax_contract.vocab import VocabContract


class StudentBackend(Protocol):
    architecture_id: str

    def default_config(self, *, vocab_size: int) -> dict[str, Any]: ...

    def init(
        self,
        config: dict[str, Any],
        vocab_contract: VocabContract,
        seed: int,
    ) -> dict[str, np.ndarray]: ...

    def forward(
        self,
        params: dict[str, np.ndarray],
        input_ids: np.ndarray,
        *,
        train: bool = False,
    ) -> np.ndarray: ...

    def parameter_fingerprint(self, params: dict[str, np.ndarray]) -> str: ...
