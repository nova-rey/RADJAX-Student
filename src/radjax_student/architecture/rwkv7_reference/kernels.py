"""Pure JAX kernels for the frozen tiny RWKV-7 reference domain.

The equations in this module are a direct structural translation of the pinned
``rwkv_v7_numpy.py`` source.  The parameter mapping is the mapping-pytree
declared by :mod:`radjax_student.architecture.rwkv7_reference.schema`; carry
uses its three persistent leaves.  ``v0`` deliberately remains token-local,
as it does in the authority source.

JAX is imported only inside executable entry points so importing the static
architecture package remains independent of the optional JAX installation.
"""

from __future__ import annotations

from typing import Any


def _layer_norm(x: Any, weight: Any, bias: Any, jnp: Any) -> Any:
    """Apply the pinned scalar layer-normalization equation."""

    return (x - jnp.mean(x)) / jnp.sqrt(jnp.var(x) + 1e-5) * weight + bias


def _group_norm(x: Any, weight: Any, bias: Any, jnp: Any) -> Any:
    """Apply the pinned per-head group normalization equation."""

    normalized = (x - jnp.mean(x, axis=1, keepdims=True)) / jnp.sqrt(
        jnp.var(x, axis=1, keepdims=True) + 64e-5
    )
    return jnp.reshape(normalized, (-1,)) * weight + bias


def _time_mixing(
    x: Any,
    v0: Any | None,
    last_x: Any,
    time_state_matrix: Any,
    parameters: Any,
    jnp: Any,
) -> tuple[Any, Any, Any, Any]:
    """Execute one pinned RWKV-7 time-mixing block."""

    attention = parameters["att"]
    xr = x + attention["x_r"] * (last_x - x)
    xw = x + attention["x_w"] * (last_x - x)
    xk = x + attention["x_k"] * (last_x - x)
    xv = x + attention["x_v"] * (last_x - x)
    xa = x + attention["x_a"] * (last_x - x)
    xg = x + attention["x_g"] * (last_x - x)

    receptance = attention["receptance"]["weight"] @ xr
    decay = jnp.exp(
        -_sigmoid(
            jnp.tanh(xw @ attention["w1"]) @ attention["w2"] + attention["w0"],
            jnp,
        )
        / jnp.sqrt(jnp.asarray(jnp.e, dtype=jnp.float32))
    )
    key = attention["key"]["weight"] @ xk
    value = attention["value"]["weight"] @ xv
    if v0 is None:
        v0 = value
    else:
        value = value + (v0 - value) * _sigmoid(
            (xv @ attention["v1"]) @ attention["v2"] + attention["v0"], jnp
        )
    aaa = _sigmoid((xa @ attention["a1"]) @ attention["a2"] + attention["a0"], jnp)
    gate = _sigmoid(xg @ attention["g1"], jnp) @ attention["g2"]
    normalized_key = key * attention["k_k"]
    key = key + key * (aaa - 1) * attention["k_a"]

    head_count, head_size = attention["r_k"].shape
    head_shape = (head_count, head_size, 1)
    receptance = jnp.reshape(receptance, head_shape)
    decay = jnp.reshape(decay, head_shape)
    key = jnp.reshape(key, head_shape)
    value = jnp.reshape(value, head_shape)
    normalized_key = jnp.reshape(normalized_key, head_shape)
    aaa = jnp.reshape(aaa, head_shape)
    r_k = jnp.reshape(attention["r_k"], head_shape)
    normalized_key = normalized_key / jnp.maximum(
        jnp.linalg.norm(normalized_key, axis=1, keepdims=True), 1e-12
    )

    time_state_matrix = (
        time_state_matrix * jnp.swapaxes(decay, -1, -2)
        - jnp.matmul(time_state_matrix, normalized_key)
        * jnp.swapaxes(normalized_key * aaa, -1, -2)
        + value * jnp.swapaxes(key, -1, -2)
    )
    y = jnp.matmul(time_state_matrix, receptance)
    y = _group_norm(y, attention["ln_x"]["weight"], attention["ln_x"]["bias"], jnp)
    y = y + jnp.reshape(
        jnp.sum(receptance * key * r_k, axis=1, keepdims=True) * value, (-1,)
    )
    output = attention["output"]["weight"] @ (y * gate)
    return output, v0, x, time_state_matrix


def _channel_mixing(x: Any, last_x: Any, parameters: Any, jnp: Any) -> tuple[Any, Any]:
    """Execute one pinned RWKV-7 channel-mixing block."""

    ffn = parameters["ffn"]
    key = ffn["key"]["weight"] @ (x + ffn["x_k"] * (last_x - x))
    value = ffn["value"]["weight"] @ jnp.maximum(key, 0) ** 2
    return value, x


def _rwkv7_step(parameters: Any, token: Any, carry: Any, jnp: Any) -> tuple[Any, Any]:
    """JAX-independent implementation body shared by step and scan entry points."""

    x = parameters["emb"]["weight"][token]
    ln0 = parameters["blocks"]["0"]["ln0"]
    x = _layer_norm(x, ln0["weight"], ln0["bias"], jnp)

    next_last_x_time = []
    next_last_x_channel = []
    next_time_state_matrix = []
    v0 = None
    for block_index in range(len(parameters["blocks"])):
        block = parameters["blocks"][str(block_index)]
        x_time = _layer_norm(x, block["ln1"]["weight"], block["ln1"]["bias"], jnp)
        time_delta, v0, last_time, time_state = _time_mixing(
            x_time,
            v0,
            carry["last_x_time"][block_index],
            carry["time_state_matrix"][block_index],
            block,
            jnp,
        )
        x = x + time_delta

        x_channel = _layer_norm(x, block["ln2"]["weight"], block["ln2"]["bias"], jnp)
        channel_delta, last_channel = _channel_mixing(
            x_channel,
            carry["last_x_channel"][block_index],
            block,
            jnp,
        )
        x = x + channel_delta
        next_last_x_time.append(last_time)
        next_last_x_channel.append(last_channel)
        next_time_state_matrix.append(time_state)

    ln_out = parameters["ln_out"]
    x = _layer_norm(x, ln_out["weight"], ln_out["bias"], jnp)
    logits = parameters["head"]["weight"] @ x
    return logits, {
        "last_x_time": jnp.stack(next_last_x_time),
        "last_x_channel": jnp.stack(next_last_x_channel),
        "time_state_matrix": jnp.stack(next_time_state_matrix),
    }


def _sigmoid(x: Any, jnp: Any) -> Any:
    """Use the authority source's logistic sigmoid definition."""

    return 1 / (1 + jnp.exp(-x))


def rwkv7_step(parameters: Any, token: Any, carry: Any) -> tuple[Any, Any]:
    """Run one RWKV-7 token step and return ``(logits, next_carry)``.

    ``parameters`` is the declared nested parameter mapping.  ``carry`` must
    contain ``last_x_time``, ``last_x_channel``, and ``time_state_matrix``;
    token-local ``v0`` is intentionally recreated for this token.
    """

    import jax.numpy as jnp

    return _rwkv7_step(parameters, token, carry, jnp)


def rwkv7_sequence(parameters: Any, tokens: Any, carry: Any) -> tuple[Any, Any]:
    """Run a rank-one token sequence with JAX ``lax.scan``.

    Returns logits with shape ``[T, V]`` plus the carry after the final token.
    """

    import jax
    import jax.numpy as jnp

    tokens = jnp.asarray(tokens)
    if tokens.ndim != 1:
        raise ValueError("RWKV-7 reference tokens must have rank one")

    def run_token(current_carry: Any, current_token: Any) -> tuple[Any, Any]:
        logits, next_carry = _rwkv7_step(parameters, current_token, current_carry, jnp)
        return next_carry, logits

    final_carry, logits = jax.lax.scan(run_token, carry, tokens)
    return logits, final_carry


__all__ = ["rwkv7_sequence", "rwkv7_step"]
