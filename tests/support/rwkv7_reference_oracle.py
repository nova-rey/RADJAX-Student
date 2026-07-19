"""Independent NumPy oracle for the pinned tiny RWKV-7 inference equations."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

VOCABULARY_SIZE = 16
HIDDEN_SIZE = 8
LAYER_COUNT = 2
HEAD_COUNT = 2
HEAD_SIZE = 4
FFN_WIDTH = 16
TIME_RANK = 32


def _parameter_values():
    cursor = 0

    def values(shape: tuple[int, ...], *, norm_weight: bool = False) -> np.ndarray:
        nonlocal cursor
        count = int(np.prod(shape))
        index = np.arange(cursor, cursor + count, dtype=np.float32)
        cursor += count
        value = np.sin(index * np.float32(0.173)) + np.cos(index * np.float32(0.071))
        value = value.astype(np.float32) * np.float32(0.025)
        if norm_weight:
            value = value + np.float32(1.0)
        return value.reshape(shape)

    parameters: dict[str, object] = {
        "emb": {"weight": values((VOCABULARY_SIZE, HIDDEN_SIZE))},
        "blocks": {},
    }
    blocks = parameters["blocks"]
    assert isinstance(blocks, dict)
    blocks["0"] = {
        "ln0": {
            "weight": values((HIDDEN_SIZE,), norm_weight=True),
            "bias": values((HIDDEN_SIZE,)),
        }
    }
    for block_index in range(LAYER_COUNT):
        block = blocks.setdefault(str(block_index), {})
        assert isinstance(block, dict)
        block["ln1"] = {
            "weight": values((HIDDEN_SIZE,), norm_weight=True),
            "bias": values((HIDDEN_SIZE,)),
        }
        attention = {
            name: values((HIDDEN_SIZE,))
            for name in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g")
        }
        attention.update(
            {
                "w0": values((HIDDEN_SIZE,)),
                "r_k": values((HEAD_COUNT, HEAD_SIZE)),
                "w1": values((HIDDEN_SIZE, TIME_RANK)),
                "w2": values((TIME_RANK, HIDDEN_SIZE)),
                "a1": values((HIDDEN_SIZE, TIME_RANK)),
                "a2": values((TIME_RANK, HIDDEN_SIZE)),
                "a0": values((HIDDEN_SIZE,)),
            }
        )
        if block_index > 0:
            attention.update(
                {
                    "v2": values((TIME_RANK, HIDDEN_SIZE)),
                    "v1": values((HIDDEN_SIZE, TIME_RANK)),
                    "v0": values((HIDDEN_SIZE,)),
                }
            )
        attention.update(
            {
                "g1": values((HIDDEN_SIZE, TIME_RANK)),
                "g2": values((TIME_RANK, HIDDEN_SIZE)),
                "k_k": values((HIDDEN_SIZE,)),
                "k_a": values((HIDDEN_SIZE,)),
                "receptance": {"weight": values((HIDDEN_SIZE, HIDDEN_SIZE))},
                "key": {"weight": values((HIDDEN_SIZE, HIDDEN_SIZE))},
                "value": {"weight": values((HIDDEN_SIZE, HIDDEN_SIZE))},
                "output": {"weight": values((HIDDEN_SIZE, HIDDEN_SIZE))},
                "ln_x": {
                    "weight": values((HIDDEN_SIZE,), norm_weight=True),
                    "bias": values((HIDDEN_SIZE,)),
                },
            }
        )
        block["att"] = attention
        block["ln2"] = {
            "weight": values((HIDDEN_SIZE,), norm_weight=True),
            "bias": values((HIDDEN_SIZE,)),
        }
        block["ffn"] = {
            "x_k": values((HIDDEN_SIZE,)),
            "key": {"weight": values((FFN_WIDTH, HIDDEN_SIZE))},
            "value": {"weight": values((HIDDEN_SIZE, FFN_WIDTH))},
        }
    parameters["ln_out"] = {
        "weight": values((HIDDEN_SIZE,), norm_weight=True),
        "bias": values((HIDDEN_SIZE,)),
    }
    parameters["head"] = {"weight": values((VOCABULARY_SIZE, HIDDEN_SIZE))}
    return parameters


def fixture_parameters() -> dict[str, object]:
    """Build a deterministic parameter tree without using production helpers."""

    return _parameter_values()


def fixture_carry() -> dict[str, np.ndarray]:
    """Return the source-shaped zero state for the frozen tiny domain."""

    return {
        "last_x_time": np.zeros((LAYER_COUNT, HIDDEN_SIZE), dtype=np.float32),
        "last_x_channel": np.zeros((LAYER_COUNT, HIDDEN_SIZE), dtype=np.float32),
        "time_state_matrix": np.zeros(
            (LAYER_COUNT, HEAD_COUNT, HEAD_SIZE, HEAD_SIZE), dtype=np.float32
        ),
    }


def _layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.var() + 1e-5) ** 0.5 * weight + bias


def _group_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return (
        (x - x.mean(axis=1, keepdims=True))
        / (x.var(axis=1, keepdims=True) + 64e-5) ** 0.5
    ).flatten() * weight + bias


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def _attention(
    x: np.ndarray,
    v0: np.ndarray | None,
    last_x: np.ndarray,
    state: np.ndarray,
    parameters: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vectors = tuple(
        np.asarray(parameters[name])
        for name in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g")
    )
    xr, xw, xk, xv, xa, xg = (x + mix * (last_x - x) for mix in vectors)
    w0 = np.asarray(parameters["w0"])
    r_k = np.asarray(parameters["r_k"])
    w1, w2 = np.asarray(parameters["w1"]), np.asarray(parameters["w2"])
    a1, a2 = np.asarray(parameters["a1"]), np.asarray(parameters["a2"])
    a0 = np.asarray(parameters["a0"])
    g1, g2 = np.asarray(parameters["g1"]), np.asarray(parameters["g2"])
    k_k, k_a = np.asarray(parameters["k_k"]), np.asarray(parameters["k_a"])
    receptance = parameters["receptance"]
    key = parameters["key"]
    value = parameters["value"]
    output = parameters["output"]
    ln_x = parameters["ln_x"]
    assert all(
        isinstance(item, Mapping) for item in (receptance, key, value, output, ln_x)
    )

    r = np.asarray(receptance["weight"]) @ xr
    w = np.exp(-_sigmoid(np.tanh(xw @ w1) @ w2 + w0) / np.e**0.5)
    k = np.asarray(key["weight"]) @ xk
    v = np.asarray(value["weight"]) @ xv
    if v0 is None:
        v0 = v
    else:
        v2, v1, v_bias = (np.asarray(parameters[name]) for name in ("v2", "v1", "v0"))
        v += (v0 - v) * _sigmoid(xv @ v1 @ v2 + v_bias)
    a = _sigmoid(xa @ a1 @ a2 + a0)
    g = _sigmoid(xg @ g1) @ g2
    kk = k * k_k
    k += k * (a - 1) * k_a

    r, w, k, v, kk, a, r_k = (
        item.reshape(HEAD_COUNT, HEAD_SIZE, 1) for item in (r, w, k, v, kk, a, r_k)
    )
    kk /= np.maximum(np.linalg.norm(kk, axis=1, keepdims=True), 1e-12)
    next_state = state * np.swapaxes(w, -2, -1)
    next_state -= (state @ kk) * np.swapaxes(kk * a, -2, -1)
    next_state += v * np.swapaxes(k, -2, -1)
    y = next_state @ r
    y = _group_norm(y, np.asarray(ln_x["weight"]), np.asarray(ln_x["bias"]))
    y += ((r * k * r_k).sum(axis=1, keepdims=True) * v).flatten()
    return np.asarray(output["weight"]) @ (y * g), v0, x, next_state


def _channel(
    x: np.ndarray, last_x: np.ndarray, parameters: Mapping[str, object]
) -> tuple[np.ndarray, np.ndarray]:
    mix = np.asarray(parameters["x_k"])
    key = parameters["key"]
    value = parameters["value"]
    assert isinstance(key, Mapping) and isinstance(value, Mapping)
    k = np.asarray(key["weight"]) @ (x + mix * (last_x - x))
    return np.asarray(value["weight"]) @ np.maximum(k, 0) ** 2, x


def rwkv7_step(
    parameters: Mapping[str, object], token: int, carry: Mapping[str, np.ndarray]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Evaluate the pinned source equations for one token without plugin code."""

    blocks = parameters["blocks"]
    ln0 = blocks["0"]["ln0"]
    assert isinstance(blocks, Mapping) and isinstance(ln0, Mapping)
    x = np.asarray(parameters["emb"]["weight"])[token]
    x = _layer_norm(x, np.asarray(ln0["weight"]), np.asarray(ln0["bias"]))
    time_values = np.asarray(carry["last_x_time"]).copy()
    channel_values = np.asarray(carry["last_x_channel"]).copy()
    state_matrices = np.asarray(carry["time_state_matrix"]).copy()
    v0: np.ndarray | None = None
    for block_index in range(LAYER_COUNT):
        block = blocks[str(block_index)]
        ln1, attention = block["ln1"], block["att"]
        assert isinstance(ln1, Mapping) and isinstance(attention, Mapping)
        x_norm = _layer_norm(x, np.asarray(ln1["weight"]), np.asarray(ln1["bias"]))
        delta, v0, time_values[block_index], state_matrices[block_index] = _attention(
            x_norm, v0, time_values[block_index], state_matrices[block_index], attention
        )
        x = x + delta
        ln2, channel = block["ln2"], block["ffn"]
        assert isinstance(ln2, Mapping) and isinstance(channel, Mapping)
        x_norm = _layer_norm(x, np.asarray(ln2["weight"]), np.asarray(ln2["bias"]))
        delta, channel_values[block_index] = _channel(
            x_norm, channel_values[block_index], channel
        )
        x = x + delta
    ln_out = parameters["ln_out"]
    assert isinstance(ln_out, Mapping)
    x = _layer_norm(x, np.asarray(ln_out["weight"]), np.asarray(ln_out["bias"]))
    logits = np.asarray(parameters["head"]["weight"]) @ x
    return logits, {
        "last_x_time": time_values,
        "last_x_channel": channel_values,
        "time_state_matrix": state_matrices,
    }


def rwkv7_sequence(
    parameters: Mapping[str, object],
    tokens: np.ndarray,
    carry: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Evaluate a token sequence by repeatedly applying the independent step."""

    logits: list[np.ndarray] = []
    next_carry = {name: np.asarray(value).copy() for name, value in carry.items()}
    for token in np.asarray(tokens).tolist():
        output, next_carry = rwkv7_step(parameters, int(token), next_carry)
        logits.append(output)
    return np.stack(logits), next_carry
