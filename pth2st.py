#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file


DTYPE = torch.float16
META_KEY = "__meta__"


def _preprocess_state_dict(src_file: str) -> dict[str, torch.Tensor]:
    raw = torch.load(src_file, map_location="cpu")

    n_head, head_size = raw["blocks.0.att.r_k"].shape
    n_embd = n_head * head_size

    processed: dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        if any(
            tag in key
            for tag in (
                "att.g1",
                "att.g2",
                "att.a1",
                "att.a2",
                "att.w1",
                "att.w2",
                "att.v1",
                "att.v2",
                "ffn.value.weight",
            )
        ):
            value = value.t()
        value = value.squeeze().to(dtype=DTYPE).contiguous()
        if key.endswith("att.r_k"):
            value = value.flatten().contiguous()
        processed[key] = value

    processed["emb.weight"] = F.layer_norm(
        processed["emb.weight"],
        (n_embd,),
        weight=processed["blocks.0.ln0.weight"],
        bias=processed["blocks.0.ln0.bias"],
    ).contiguous()
    processed["blocks.0.att.v0"] = processed["blocks.0.att.a0"].clone()
    processed["blocks.0.att.v1"] = processed["blocks.0.att.a1"].clone()
    processed["blocks.0.att.v2"] = processed["blocks.0.att.a2"].clone()

    max_layer = max(
        int(key.split(".")[1]) for key in processed if key.startswith("blocks.")
    )
    vocab_size = processed["head.weight"].size(0)
    processed[META_KEY] = torch.tensor(
        [max_layer + 1, n_head, head_size, n_embd, vocab_size],
        dtype=torch.int64,
    )
    return processed


def convert_pth_to_safetensors(src_file: str, out_file: str) -> None:
    tensors = _preprocess_state_dict(src_file)
    output_path = Path(out_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output_path), metadata={"format": "rwkv-lightning"})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert an RWKV torch checkpoint (.pth/.pt state_dict) to safetensors."
    )
    parser.add_argument("--model", required=True, help="Input checkpoint path")
    parser.add_argument("--out", required=True, help="Output .safetensors path")
    args = parser.parse_args()
    convert_pth_to_safetensors(args.model, args.out)


if __name__ == "__main__":
    main()
