#!/usr/bin/env python3
"""Extract the decoder tensors from Diffusers TAESD weights for runtime tests."""

from pathlib import Path
import sys

from safetensors import safe_open
from safetensors.numpy import save_file


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {Path(sys.argv[0]).name} INPUT.safetensors OUTPUT.safetensors", file=sys.stderr)
        return 2

    source, output = sys.argv[1:]
    decoder_tensors = {}
    with safe_open(source, framework="np") as weights:
        for name in weights.keys():
            if name.startswith("decoder.layers."):
                decoder_tensors[name] = weights.get_tensor(name)

    if not decoder_tensors:
        print("FAIL: the input has no decoder.layers.* TAESD tensors", file=sys.stderr)
        return 1

    save_file(decoder_tensors, output)
    print(f"PASS: extracted {len(decoder_tensors)} decoder tensors to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
