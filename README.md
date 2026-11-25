# Vector Quantization Image Compression

## Member 1 interface

Use `generate_preprocessing_and_codebook()` from `codebook_generator.py`:

```python
from codebook_generator import generate_preprocessing_and_codebook

result = generate_preprocessing_and_codebook(
    "image.png",
    block_height=4,
    block_width=4,
    k=16,
)
```

The returned dictionary contains:

- `codebook`: NumPy array with shape `(K, Bh * Bw)`.
- `vectors`: NumPy array with shape `(N, Bh * Bw)`, ordered row-major.
- `original_shape`: `(original_height, original_width)`.
- `padded_shape`: `(padded_height, padded_width)`.
- `block_height`: block height `Bh`.
- `block_width`: block width `Bw`.
- `num_vectors`: number of extracted vectors `N`.
- `num_codewords`: number of codewords `K`.

For encoding, process `vectors` from left to right and top to bottom, and compare each vector to `codebook` using Manhattan distance.

For reconstruction, place each encoded codeword in row-major block order using `Bh` and `Bw`, reshape each codeword to `(Bh, Bw)`, and crop the reconstructed array to `original_shape`.

Member 1 only provides preprocessing and codebook generation. It does not perform encoding, binary file writing, decompression, or compression-ratio calculation.

## Dependencies

```bash
pip install -r requirements.txt
```
