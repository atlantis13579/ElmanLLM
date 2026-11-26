"""Compare the native SDPA forward and gradients against independent PyTorch fixtures."""
import argparse
import json
import pathlib
import subprocess

import torch
import torch.nn.functional as F


def tensor(value):
    if value is None:
        return None
    return {"shape": list(value.shape), "values": value.detach().flatten().tolist(),
            "boolean": value.dtype == torch.bool}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=pathlib.Path)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("Build/sdpa_torch_reference.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    cases = []
    geometries = [(7, 11, 5, 3, False, 0, False, False),
                  (7, 11, 5, 3, True, 0, False, False),
                  (11, 7, 9, 5, True, 0, False, False),
                  (7, 17, 13, 5, False, 1, False, True),
                  (7, 17, 13, 5, False, 2, True, False),
                  (2, 1, 1, 1, False, 1, False, False),
                  (5, 33, 7, 9, False, 0, True, True)]
    for length, keys, width, values, causal, mask_kind, gqa, broadcast in geometries:
        heads, kv_heads = 4, 2 if gqa else 4
        q = (torch.randn(2, heads, length, width) * .4).requires_grad_()
        k = (torch.randn(1 if broadcast else 2, kv_heads, keys, width) * .4).requires_grad_()
        v = (torch.randn(2, kv_heads, keys, values) * .4).requires_grad_()
        mask = None
        if mask_kind == 1:
            mask = torch.rand(length, keys) > .25
            mask[0] = False
        elif mask_kind == 2:
            mask = (torch.randn(length, keys) * .2).requires_grad_()
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                           is_causal=causal, scale=.37, enable_gqa=gqa)
        dy = torch.randn_like(y)
        y.backward(dy)
        cases.append({"q": tensor(q), "k": tensor(k), "v": tensor(v), "mask": tensor(mask),
                      "causal": causal, "gqa": gqa, "scale": .37, "dy": tensor(dy),
                      "y": tensor(y), "dq": tensor(q.grad), "dk": tensor(k.grad),
                      "dv": tensor(v.grad), "dm": tensor(mask.grad) if mask_kind == 2 else None})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"torch": torch.__version__, "cases": cases}, indent=2) + "\n", encoding="utf-8")
    subprocess.run([str(args.binary.resolve()), args.backend, str(args.output.resolve())], check=True)


if __name__ == "__main__":
    main()
