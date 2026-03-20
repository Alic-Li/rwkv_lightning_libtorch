#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F


DTYPE = torch.float16


def _save_tensor_list(tensors, out_file: str) -> None:
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    data = torch._C._pickle_save(tensors)
    with open(out_file, "wb") as f:
        f.write(data)


def export_weights(model_prefix: str, out_file: str) -> None:
    src = model_prefix
    raw = torch.load(src, map_location="cpu")

    n_head, head_size = raw["blocks.0.att.r_k"].shape
    n_embd = n_head * head_size

    processed = {}
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
        int(key.split(".")[1])
        for key in processed
        if key.startswith("blocks.")
    )
    vocab_size = processed["head.weight"].size(0)
    tensors = [
        torch.tensor(
            [max_layer + 1, n_head, head_size, n_embd, vocab_size],
            dtype=torch.int64,
        ),
        processed["emb.weight"],
        processed["ln_out.weight"],
        processed["ln_out.bias"],
        processed["head.weight"],
    ]

    for i in range(max_layer + 1):
        bbb = f"blocks.{i}."
        att = f"{bbb}att."
        ffn = f"{bbb}ffn."
        tensors.extend(
            [
                processed[bbb + "ln1.weight"],
                processed[bbb + "ln1.bias"],
                processed[bbb + "ln2.weight"],
                processed[bbb + "ln2.bias"],
                processed[att + "x_r"],
                processed[att + "x_w"],
                processed[att + "x_k"],
                processed[att + "x_v"],
                processed[att + "x_a"],
                processed[att + "x_g"],
                processed[att + "w0"],
                processed[att + "w1"],
                processed[att + "w2"],
                processed[att + "a0"],
                processed[att + "a1"],
                processed[att + "a2"],
                processed[att + "v0"],
                processed[att + "v1"],
                processed[att + "v2"],
                processed[att + "g1"],
                processed[att + "g2"],
                processed[att + "k_k"],
                processed[att + "k_a"],
                processed[att + "r_k"],
                processed[att + "receptance.weight"],
                processed[att + "key.weight"],
                processed[att + "value.weight"],
                processed[att + "output.weight"],
                processed[att + "ln_x.weight"],
                processed[att + "ln_x.bias"],
                processed[ffn + "x_k"],
                processed[ffn + "key.weight"],
                processed[ffn + "value.weight"],
            ]
        )
    _save_tensor_list(tensors, out_file)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model prefix without .pth")
    parser.add_argument("--out", required=True, help="Output .pt file")
    args = parser.parse_args()
    export_weights(args.model, args.out)


if __name__ == "__main__":
    main()
