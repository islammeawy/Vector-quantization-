import math
from pathlib import Path
import struct
import numpy as np
from PIL import Image


def decode(input_dir="output", output_path="output/reconstructed.png"):
    input_dir = Path(input_dir)

    with open(input_dir / "codebook.bin", "rb") as f:
        _, k, dim, bh, bw = struct.unpack("<4s4I", f.read(20))
        codebook = np.frombuffer(f.read(), dtype="<f8").reshape(k, dim)

    with open(input_dir / "compressed.bin", "rb") as f:
        _, orig_h, orig_w, pad_h, pad_w, bh, bw, k, num_vectors, itemsize = struct.unpack("<4s9I", f.read(40))
        indices = np.frombuffer(f.read(), dtype=f"<u{itemsize}")

    blocks = codebook[indices]
    reconstructed = np.zeros((pad_h, pad_w))

    i = 0
    for r in range(0, pad_h, bh):
        for c in range(0, pad_w, bw):
            reconstructed[r:r + bh, c:c + bw] = blocks[i].reshape(bh, bw)
            i += 1

    final_image = np.clip(np.rint(reconstructed[:orig_h, :orig_w]), 0, 255).astype(np.uint8)
    Image.fromarray(final_image).save(output_path)

    bits_per_label = max(1, math.ceil(math.log2(k)))
    original_bits = orig_h * orig_w * 8
    compressed_bits = (k * bh * bw * 8) + (num_vectors * bits_per_label)
    ratio = original_bits / compressed_bits

    return final_image, ratio
    