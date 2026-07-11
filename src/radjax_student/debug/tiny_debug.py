from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from radjax_contract.provenance import stable_hash
from radjax_contract.vocab import VocabContract


@dataclass(frozen=True)
class TinyDebugStudentBackend:
    """NumPy smoke backend; not a production architecture plugin."""

    architecture_id: str = "tiny_debug"

    def default_config(self, *, vocab_size: int) -> dict[str, Any]:
        return {"vocab_size": int(vocab_size), "hidden_size": 4}

    def init(
        self,
        config: dict[str, Any],
        vocab_contract: VocabContract,
        seed: int,
    ) -> dict[str, np.ndarray]:
        if int(config["vocab_size"]) != vocab_contract.vocab_size:
            raise ValueError("config vocab_size must match vocab contract")
        rng = np.random.default_rng(seed)
        return {
            "embedding": rng.normal(
                0.0,
                0.01,
                size=(vocab_contract.vocab_size, int(config["hidden_size"])),
            ).astype(np.float32),
            "head": rng.normal(
                0.0,
                0.01,
                size=(int(config["hidden_size"]), vocab_contract.vocab_size),
            ).astype(np.float32),
        }

    def forward(
        self,
        params: dict[str, np.ndarray],
        input_ids: np.ndarray,
        *,
        train: bool = False,
    ) -> np.ndarray:
        del train
        hidden = params["embedding"][np.asarray(input_ids, dtype=np.int32)]
        return hidden @ params["head"]

    def parameter_fingerprint(self, params: dict[str, np.ndarray]) -> str:
        return stable_hash(
            {
                name: {
                    "shape": list(value.shape),
                    "sum": float(np.sum(value, dtype=np.float64)),
                }
                for name, value in sorted(params.items())
            }
        )
