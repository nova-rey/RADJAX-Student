"""Portable pure-JAX Mamba-2 recurrent reference equations.

JAX is imported only by the public execution entry points.  The implementation
uses functional state transitions and ``lax.scan``; it deliberately does not
call Triton, Pallas, CUDA extensions, or custom XLA lowering.
"""

from __future__ import annotations

from typing import Any


def _rms_norm(x: Any, weight: Any, jnp: Any, *, eps: float = 1e-5) -> Any:
    x_float = x.astype(jnp.float32)
    return (
        x_float
        * jax_lax_rsqrt(jnp.mean(x_float * x_float, axis=-1, keepdims=True) + eps, jnp)
    ) * weight


def jax_lax_rsqrt(value: Any, jnp: Any) -> Any:
    return jnp.reciprocal(jnp.sqrt(value))


def _silu(x: Any, jnp: Any) -> Any:
    return x * jax_sigmoid(x, jnp)


def jax_sigmoid(x: Any, jnp: Any) -> Any:
    return jnp.reciprocal(1.0 + jnp.exp(-x))


def _gated_rms_norm(x: Any, z: Any, weight: Any, jnp: Any) -> Any:
    # Upstream norm_before_gate=False for the frozen profile: gate before RMS.
    gated = x.astype(jnp.float32) * _silu(z.astype(jnp.float32), jnp)
    return _rms_norm(gated, weight, jnp)


def _mamba2_layer_step(
    hidden: Any,
    layer_parameters: Any,
    conv_state: Any,
    ssm_state: Any,
    jnp: Any,
) -> tuple[Any, Any, Any]:
    mixer = layer_parameters["mixer"]
    projected = hidden @ jnp.swapaxes(mixer["in_proj"]["weight"], -1, -2)
    d_ssm = mixer["norm"]["weight"].shape[0]
    # d_ssm is explicitly the configured inner width for this profile.  The
    # projection order is [z, x, B, C, dt], with no MLP remainder.
    nheads = mixer["A_log"].shape[0]
    headdim = mixer["norm"]["weight"].shape[0] // nheads
    d_state = ssm_state.shape[-1]
    conv_dim = conv_state.shape[1]
    d_mlp = (projected.shape[-1] - 2 * d_ssm - 2 * d_state - nheads) // 2
    if d_mlp != 0:
        raise ValueError("Mamba-2 reference profile does not support an MLP remainder")
    z, input_xbc, dt_raw = jnp.split(
        projected, (d_ssm, d_ssm + d_ssm + 2 * d_state), axis=-1
    )
    del conv_dim

    next_conv_state = jnp.concatenate(
        (conv_state[:, :, 1:], input_xbc[:, :, None]), axis=-1
    )
    conv_weight = mixer["conv1d"]["weight"][:, 0, :]
    xbc = jnp.sum(next_conv_state * conv_weight[None, :, :], axis=-1)
    xbc = xbc + mixer["conv1d"]["bias"]
    xbc = _silu(xbc, jnp)
    x, b_value, c_value = jnp.split(xbc, (d_ssm, d_ssm + d_state), axis=-1)

    dt = jax_softplus(dt_raw + mixer["dt_bias"], jnp)
    decay = jnp.exp(dt * (-jnp.exp(mixer["A_log"].astype(jnp.float32))))
    x_heads = jnp.reshape(x, (x.shape[0], nheads, headdim))
    next_ssm_state = (
        ssm_state * decay[:, :, None, None]
        + dt[:, :, None, None] * b_value[:, None, None, :] * x_heads[:, :, :, None]
    )
    y = jnp.einsum("bhpn,bn->bhp", next_ssm_state, c_value)
    y = y + mixer["D"][None, :, None] * x_heads
    y = jnp.reshape(y, (y.shape[0], d_ssm))
    y = _gated_rms_norm(y, z, mixer["norm"]["weight"], jnp)
    output = y @ jnp.swapaxes(mixer["out_proj"]["weight"], -1, -2)
    return output, next_conv_state, next_ssm_state


def _mamba2_token(parameters: Any, token: Any, carry: Any, jnp: Any) -> tuple[Any, Any]:
    hidden = parameters["backbone"]["embedding"]["weight"][token][None, :]
    residual = None
    next_carry: dict[str, Any] = {}
    n_layer = len(parameters["backbone"]["layers"])
    for layer_index in range(n_layer):
        layer = parameters["backbone"]["layers"][str(layer_index)]
        residual = hidden if residual is None else hidden + residual
        hidden = _rms_norm(
            hidden if residual is None else residual, layer["norm"]["weight"], jnp
        )
        output, conv_state, ssm_state = _mamba2_layer_step(
            hidden,
            layer,
            carry[f"layers.{layer_index}.conv_state"],
            carry[f"layers.{layer_index}.ssm_state"],
            jnp,
        )
        hidden = output
        next_carry[f"layers.{layer_index}.conv_state"] = conv_state
        next_carry[f"layers.{layer_index}.ssm_state"] = ssm_state
    if residual is not None:
        hidden = hidden + residual
    hidden = _rms_norm(hidden, parameters["backbone"]["norm_f"]["weight"], jnp)
    logits = hidden @ jnp.swapaxes(parameters["lm_head"]["weight"], -1, -2)
    return logits, next_carry


def jax_softplus(x: Any, jnp: Any) -> Any:
    # Stable equivalent of torch.nn.functional.softplus for float32 fixtures.
    return jnp.maximum(x, 0) + jnp.log1p(jnp.exp(-jnp.abs(x)))


def mamba2_step(parameters: Any, token: Any, carry: Any) -> tuple[Any, Any]:
    import jax.numpy as jnp

    return _mamba2_token(parameters, token, carry, jnp)


def mamba2_sequence(parameters: Any, tokens: Any, carry: Any) -> tuple[Any, Any]:
    import jax
    import jax.numpy as jnp

    tokens = jnp.asarray(tokens)
    if tokens.ndim != 1:
        raise ValueError("Mamba-2 reference tokens must have rank one")

    def run_token(current_carry: Any, token: Any) -> tuple[Any, Any]:
        logits, next_carry = _mamba2_token(parameters, token, current_carry, jnp)
        return next_carry, logits

    final_carry, logits = jax.lax.scan(run_token, carry, tokens)
    return logits[:, 0, :], final_carry


__all__ = ["mamba2_sequence", "mamba2_step"]
