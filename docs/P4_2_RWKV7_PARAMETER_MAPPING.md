# P4.2 RWKV-7 Reference Parameter Mapping

P4.2 records the static, float32 parameter schema for
`radjax.architecture.rwkv7_reference`, version `1`. The mathematical authority
remains BlinkDL/RWKV-LM commit `442120a5b40f7d764328bebde94324bc8790806f`,
`RWKV-v7/rwkv_v7_numpy.py`, SHA-256
`dd683466cf97880c82879afbc8abb27a9596b12344a825d8325a1a1753597ee6`.

This is a source-prefix consumption specification for that NumPy file, not a
claim that a PyTorch state dictionary, checkpoint file, or Hugging Face model
can be loaded. The source's `weights = {k: v.squeeze().float().numpy() ...}`
normalization is represented only where noted below. Every present leaf is
float32 and has initialization claim **not initialized until P4.3**.

## Frozen tiny-domain shapes

The declared domain is vocabulary `V=16`, hidden `H=8`, layers `L=2`, heads
`N=2`, head size `Hd=4`, FFN width `F=16`, context `T=4`, and time-mix rank
`R=32`. The four time-mix ranks (`w`, `a`, `v`, and `g`) are frozen at `32` so
the pinned matrix products are conformable on the tiny domain; this is a shape
schema choice, not an initialization or training-recipe claim.

`schema.pinned_numpy_parameter_order()` is the checked-in order fixture
specification. Its values are the exact ordered lists that P4.4 must pass for
each `params(prefix)` access. The `ParameterCatalog` remains path-sorted for
generic inspection, so it intentionally does not encode source list order.

## Literal mapping and consumption order

| Pinned NumPy prefix / consumed slot | RADJAX pytree leaf | Shape / dtype | Initialization | Representation and equation basis |
| --- | --- | --- | --- | --- |
| `emb`: `params("emb")[0]` | `emb.weight` | `[V,H] = [16,8]`, float32 | P4.3 | direct matrix used for token lookup |
| `blocks.0.ln0`: `w` | `blocks.0.ln0.weight` | `[H] = [8]`, float32 | P4.3 | direct |
| `blocks.0.ln0`: `b` | `blocks.0.ln0.bias` | `[H] = [8]`, float32 | P4.3 | direct |
| `blocks.0.ln1`: `w` | `blocks.0.ln1.weight` | `[H] = [8]`, float32 | P4.3 | direct |
| `blocks.0.ln1`: `b` | `blocks.0.ln1.bias` | `[H] = [8]`, float32 | P4.3 | direct |
| `blocks.0.att`: `mr` | `blocks.0.att.x_r` | `[H] = [8]`, float32 | P4.3 | transformed only by pinned `.squeeze()` to vector |
| `blocks.0.att`: `mw` | `blocks.0.att.x_w` | `[H] = [8]`, float32 | P4.3 | transformed only by pinned `.squeeze()` to vector |
| `blocks.0.att`: `mk` | `blocks.0.att.x_k` | `[H] = [8]`, float32 | P4.3 | transformed only by pinned `.squeeze()` to vector |
| `blocks.0.att`: `mv` | `blocks.0.att.x_v` | `[H] = [8]`, float32 | P4.3 | transformed only by pinned `.squeeze()` to vector |
| `blocks.0.att`: `ma` | `blocks.0.att.x_a` | `[H] = [8]`, float32 | P4.3 | transformed only by pinned `.squeeze()` to vector |
| `blocks.0.att`: `mg` | `blocks.0.att.x_g` | `[H] = [8]`, float32 | P4.3 | transformed only by pinned `.squeeze()` to vector |
| `blocks.0.att`: `w_bias` | `blocks.0.att.w0` | `[H] = [8]`, float32 | P4.3 | transformed only by pinned `.squeeze()` to vector |
| `blocks.0.att`: `r_k` | `blocks.0.att.r_k` | `[N,Hd] = [2,4]`, float32 | P4.3 | transformed from flat `H`; kernel reshapes to `[N,Hd,1]` exactly where NumPy does |
| `blocks.0.att`: `Ww1` | `blocks.0.att.w1` | `[H,R] = [8,32]`, float32 | P4.3 | direct matrix |
| `blocks.0.att`: `Ww2` | `blocks.0.att.w2` | `[R,H] = [32,8]`, float32 | P4.3 | direct matrix |
| `blocks.0.att`: `Wa1` | `blocks.0.att.a1` | `[H,R] = [8,32]`, float32 | P4.3 | direct matrix |
| `blocks.0.att`: `Wa2` | `blocks.0.att.a2` | `[R,H] = [32,8]`, float32 | P4.3 | direct matrix |
| `blocks.0.att`: `a_bias` | `blocks.0.att.a0` | `[H] = [8]`, float32 | P4.3 | transformed only by pinned `.squeeze()` to vector |
| `blocks.0.att`: `Wg1` | `blocks.0.att.g1` | `[H,R] = [8,32]`, float32 | P4.3 | direct matrix |
| `blocks.0.att`: `Wg2` | `blocks.0.att.g2` | `[R,H] = [32,8]`, float32 | P4.3 | direct matrix |
| `blocks.0.att`: `Wv2` | omitted | no leaf | not initialized | intentionally omitted: `v0 is None` on the first block, so NumPy never reads `params[15:18]` |
| `blocks.0.att`: `Wv1` | omitted | no leaf | not initialized | intentionally omitted: same first-block branch; no equation-bearing value is lost |
| `blocks.0.att`: `v_bias` | omitted | no leaf | not initialized | intentionally omitted: same first-block branch; `v0` becomes the block-0 value |
| `blocks.0.att`: `k_k` | `blocks.0.att.k_k` | `[H] = [8]`, float32 | P4.3 | transformed only by pinned `.squeeze()` to vector |
| `blocks.0.att`: `k_a` | `blocks.0.att.k_a` | `[H] = [8]`, float32 | P4.3 | transformed only by pinned `.squeeze()` to vector |
| `blocks.0.att`: `Wr` | `blocks.0.att.receptance.weight` | `[H,H] = [8,8]`, float32 | P4.3 | direct matrix |
| `blocks.0.att`: `Wk` | `blocks.0.att.key.weight` | `[H,H] = [8,8]`, float32 | P4.3 | direct matrix |
| `blocks.0.att`: `Wv` | `blocks.0.att.value.weight` | `[H,H] = [8,8]`, float32 | P4.3 | direct matrix |
| `blocks.0.att`: `Wo` | `blocks.0.att.output.weight` | `[H,H] = [8,8]`, float32 | P4.3 | direct matrix |
| `blocks.0.att`: `ln_w` | `blocks.0.att.ln_x.weight` | `[H] = [8]`, float32 | P4.3 | direct |
| `blocks.0.att`: `ln_b` | `blocks.0.att.ln_x.bias` | `[H] = [8]`, float32 | P4.3 | direct |
| `blocks.0.ln2`: `w` | `blocks.0.ln2.weight` | `[H] = [8]`, float32 | P4.3 | direct |
| `blocks.0.ln2`: `b` | `blocks.0.ln2.bias` | `[H] = [8]`, float32 | P4.3 | direct |
| `blocks.0.ffn`: `mix` | `blocks.0.ffn.x_k` | `[H] = [8]`, float32 | P4.3 | transformed only by pinned `.squeeze()` to vector |
| `blocks.0.ffn`: `Wk` | `blocks.0.ffn.key.weight` | `[F,H] = [16,8]`, float32 | P4.3 | direct matrix |
| `blocks.0.ffn`: `Wv` | `blocks.0.ffn.value.weight` | `[H,F] = [8,16]`, float32 | P4.3 | direct matrix |
| `blocks.1.ln1`: `w` | `blocks.1.ln1.weight` | `[H] = [8]`, float32 | P4.3 | direct |
| `blocks.1.ln1`: `b` | `blocks.1.ln1.bias` | `[H] = [8]`, float32 | P4.3 | direct |
| `blocks.1.att`: `mr` | `blocks.1.att.x_r` | `[H] = [8]`, float32 | P4.3 | transformed only by pinned `.squeeze()` to vector |
| `blocks.1.att`: `mw` | `blocks.1.att.x_w` | `[H] = [8]`, float32 | P4.3 | transformed only by pinned `.squeeze()` to vector |
| `blocks.1.att`: `mk` | `blocks.1.att.x_k` | `[H] = [8]`, float32 | P4.3 | transformed only by pinned `.squeeze()` to vector |
| `blocks.1.att`: `mv` | `blocks.1.att.x_v` | `[H] = [8]`, float32 | P4.3 | transformed only by pinned `.squeeze()` to vector |
| `blocks.1.att`: `ma` | `blocks.1.att.x_a` | `[H] = [8]`, float32 | P4.3 | transformed only by pinned `.squeeze()` to vector |
| `blocks.1.att`: `mg` | `blocks.1.att.x_g` | `[H] = [8]`, float32 | P4.3 | transformed only by pinned `.squeeze()` to vector |
| `blocks.1.att`: `w_bias` | `blocks.1.att.w0` | `[H] = [8]`, float32 | P4.3 | transformed only by pinned `.squeeze()` to vector |
| `blocks.1.att`: `r_k` | `blocks.1.att.r_k` | `[N,Hd] = [2,4]`, float32 | P4.3 | transformed from flat `H`; kernel reshapes to `[N,Hd,1]` exactly where NumPy does |
| `blocks.1.att`: `Ww1` | `blocks.1.att.w1` | `[H,R] = [8,32]`, float32 | P4.3 | direct matrix |
| `blocks.1.att`: `Ww2` | `blocks.1.att.w2` | `[R,H] = [32,8]`, float32 | P4.3 | direct matrix |
| `blocks.1.att`: `Wa1` | `blocks.1.att.a1` | `[H,R] = [8,32]`, float32 | P4.3 | direct matrix |
| `blocks.1.att`: `Wa2` | `blocks.1.att.a2` | `[R,H] = [32,8]`, float32 | P4.3 | direct matrix |
| `blocks.1.att`: `a_bias` | `blocks.1.att.a0` | `[H] = [8]`, float32 | P4.3 | transformed only by pinned `.squeeze()` to vector |
| `blocks.1.att`: `Wg1` | `blocks.1.att.g1` | `[H,R] = [8,32]`, float32 | P4.3 | direct matrix |
| `blocks.1.att`: `Wg2` | `blocks.1.att.g2` | `[R,H] = [32,8]`, float32 | P4.3 | direct matrix |
| `blocks.1.att`: `Wv2` | `blocks.1.att.v2` | `[R,H] = [32,8]`, float32 | P4.3 | direct matrix; NumPy consumes `Wv2` before `Wv1` |
| `blocks.1.att`: `Wv1` | `blocks.1.att.v1` | `[H,R] = [8,32]`, float32 | P4.3 | direct matrix |
| `blocks.1.att`: `v_bias` | `blocks.1.att.v0` | `[H] = [8]`, float32 | P4.3 | transformed only by pinned `.squeeze()` to vector |
| `blocks.1.att`: `k_k` | `blocks.1.att.k_k` | `[H] = [8]`, float32 | P4.3 | transformed only by pinned `.squeeze()` to vector |
| `blocks.1.att`: `k_a` | `blocks.1.att.k_a` | `[H] = [8]`, float32 | P4.3 | transformed only by pinned `.squeeze()` to vector |
| `blocks.1.att`: `Wr` | `blocks.1.att.receptance.weight` | `[H,H] = [8,8]`, float32 | P4.3 | direct matrix |
| `blocks.1.att`: `Wk` | `blocks.1.att.key.weight` | `[H,H] = [8,8]`, float32 | P4.3 | direct matrix |
| `blocks.1.att`: `Wv` | `blocks.1.att.value.weight` | `[H,H] = [8,8]`, float32 | P4.3 | direct matrix |
| `blocks.1.att`: `Wo` | `blocks.1.att.output.weight` | `[H,H] = [8,8]`, float32 | P4.3 | direct matrix |
| `blocks.1.att`: `ln_w` | `blocks.1.att.ln_x.weight` | `[H] = [8]`, float32 | P4.3 | direct |
| `blocks.1.att`: `ln_b` | `blocks.1.att.ln_x.bias` | `[H] = [8]`, float32 | P4.3 | direct |
| `blocks.1.ln2`: `w` | `blocks.1.ln2.weight` | `[H] = [8]`, float32 | P4.3 | direct |
| `blocks.1.ln2`: `b` | `blocks.1.ln2.bias` | `[H] = [8]`, float32 | P4.3 | direct |
| `blocks.1.ffn`: `mix` | `blocks.1.ffn.x_k` | `[H] = [8]`, float32 | P4.3 | transformed only by pinned `.squeeze()` to vector |
| `blocks.1.ffn`: `Wk` | `blocks.1.ffn.key.weight` | `[F,H] = [16,8]`, float32 | P4.3 | direct matrix |
| `blocks.1.ffn`: `Wv` | `blocks.1.ffn.value.weight` | `[H,F] = [8,16]`, float32 | P4.3 | direct matrix |
| `ln_out`: `w` | `ln_out.weight` | `[H] = [8]`, float32 | P4.3 | direct |
| `ln_out`: `b` | `ln_out.bias` | `[H] = [8]`, float32 | P4.3 | direct |
| `head`: `params("head")[0]` | `head.weight` | `[V,H] = [16,8]`, float32 | P4.3 | direct matrix; explicitly untied from `emb.weight` |

The first-block omissions are not catalog leaves and are not included in
`pinned_numpy_parameter_order()["blocks.0.att"]`. They are documented above so
P4.4 cannot accidentally introduce them. `v0` itself is token-local: block 0
produces it, block 1 consumes it, and it is not part of persistent carry.

## Static descriptor and non-claims

At the P4.2 checkpoint the plugin had static schema capability only and
executable assembly rejected it before initialization or forward. P4.4 later
adds the narrowly scoped JAX execution capability and separate parity proof;
this mapping remains the static source-to-pytree authority. The parameter tree
layout and HF descriptor cover every catalog leaf, but all HF projections are
non-exportable with reason `weight_file_compatibility_not_claimed`.

P4.2 does not initialize tensors, create carry values, execute an equation,
claim equation parity, load weights, convert a checkpoint, implement
`from_pretrained` or `save_pretrained`, tie embedding/head weights, or make any
training, quality, performance, multi-device, TPU, teacher, Tome, distillation,
or Phase 5 claim.
