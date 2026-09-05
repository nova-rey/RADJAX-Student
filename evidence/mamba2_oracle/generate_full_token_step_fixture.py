"""Generate the complete upstream Mamba-2 V16/T4 token-step fixture.

Run only inside the pinned CUDA evidence environment.  This script is not a
Student dependency and intentionally imports the authoritative upstream model.
"""
# ruff: noqa: I001

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.utils.generation import InferenceParams


COMMIT = "95d8aba8a8c75aedcaa6143713b11e745e7cd0d9"
SEED = 20260905
TOKENS = (1, 7, 3, 12)
SOURCE_FILES = (
    "modules/mamba2.py",
    "modules/block.py",
    "models/mixer_seq_simple.py",
    "ops/triton/selective_state_update.py",
    "ops/triton/softplus.py",
    "ops/triton/ssd_combined.py",
    "ops/triton/layer_norm.py",
    "ops/triton/layernorm_gated.py",
)


def _config() -> MambaConfig:
    return MambaConfig(
        d_model=8,
        d_intermediate=0,
        n_layer=2,
        vocab_size=16,
        ssm_cfg={
            "layer": "Mamba2",
            "d_state": 4,
            "d_conv": 4,
            "expand": 2,
            "headdim": 4,
            "d_ssm": 16,
            "ngroups": 1,
            "use_mem_eff_path": False,
            "chunk_size": 4,
            "rmsnorm": True,
            "norm_before_gate": False,
            "dt_min": 0.001,
            "dt_max": 0.1,
            "dt_init_floor": 1e-4,
            "dt_limit": (0.0, float("inf")),
            "bias": False,
            "conv_bias": True,
        },
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        tie_embeddings=True,
        pad_vocab_size_multiple=1,
    )


def generate(output: Path) -> None:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = MambaLMHeadModel(_config(), device="cuda", dtype=torch.float32).eval()
    cache = model.backbone.allocate_inference_cache(1, 4)
    for layer, (conv, ssm) in cache.items():
        conv.copy_(
            torch.arange(conv.numel(), device="cuda", dtype=torch.float32).reshape_as(
                conv
            )
            * 0.001
            + (layer + 1) * 0.01
        )
        ssm.copy_(
            torch.arange(ssm.numel(), device="cuda", dtype=torch.float32).reshape_as(
                ssm
            )
            * 0.002
            + (layer + 1) * 0.02
        )
    initial = {
        str(layer): [value.detach().cpu().tolist() for value in values]
        for layer, values in cache.items()
    }
    logits = []
    for offset, token in enumerate(TOKENS, start=1):
        inference = InferenceParams(
            max_seqlen=4,
            max_batch_size=1,
            seqlen_offset=offset,
            key_value_memory_dict=cache,
        )
        with torch.no_grad():
            result = model(
                torch.tensor([[token]], device="cuda"),
                inference_params=inference,
            ).logits
        torch.cuda.synchronize()
        logits.append(result[0, 0].detach().cpu().tolist())
    final = {
        str(layer): [value.detach().cpu().tolist() for value in values]
        for layer, values in cache.items()
    }
    package_root = Path(torch.__file__).parent.parent / "mamba_ssm"
    hashes = {
        relative: hashlib.sha256((package_root / relative).read_bytes()).hexdigest()
        for relative in SOURCE_FILES
    }
    payload = {
        "source": {
            "repository": "state-spaces/mamba",
            "commit": COMMIT,
            "version": "2.2.4",
            "files_sha256": hashes,
        },
        "environment": {
            "image": "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": __import__("triton").__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
        "config": {
            "d_model": 8,
            "n_layer": 2,
            "vocab_size": 16,
            "context": 4,
            "ssm_cfg": {
                **_config().ssm_cfg,
                "dt_limit": {"min": 0.0, "max": "UNBOUNDED"},
            },
            "rms_norm": True,
            "residual_in_fp32": True,
            "fused_add_norm": True,
            "tie_embeddings": True,
            "pad_vocab_size_multiple": 1,
        },
        "seed": SEED,
        "tokens": list(TOKENS),
        "state_initial": initial,
        "state_final": final,
        "logits": logits,
        "model_state_dict": {
            key: value.detach().cpu().tolist()
            for key, value in model.state_dict().items()
        },
        "settings": {
            "tf32": False,
            "float32_matmul_precision": "highest",
            "deterministic_algorithms": True,
            "model_mode": "eval",
            "inference_mode": "InferenceParams token-step with preallocated caches",
            "cache_initialization": "deliberately asymmetric nonuniform values",
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["fixture_sha256"] = hashlib.sha256(canonical).hexdigest()
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(payload["fixture_sha256"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    generate(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
