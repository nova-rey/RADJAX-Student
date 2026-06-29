import numpy as np
from radjax_contract.vocab import VocabContract

from radjax_student.students import StudentBackendRegistry


def test_tiny_debug_backend_registers_and_forwards() -> None:
    registry = StudentBackendRegistry.with_defaults()
    backend = registry.get("tiny_debug")
    vocab = VocabContract(tokenizer_id="toy", vocab_size=5)
    params = backend.init(backend.default_config(vocab_size=5), vocab, seed=1)
    logits = backend.forward(params, np.asarray([[0, 1]], dtype=np.int32))

    assert registry.names() == ("tiny_debug",)
    assert logits.shape == (1, 2, 5)
    assert len(backend.parameter_fingerprint(params)) == 64
