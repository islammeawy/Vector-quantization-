"""Vector Quantization compression encoding."""

from pathlib import Path
import struct
import numpy as np


def assign_vectors(vectors, codebook):
    """Assign every image vector to its nearest codeword using L1 distance."""
    vectors = np.asarray(vectors, dtype=np.float64)
    codebook = np.asarray(codebook, dtype=np.float64)

    if vectors.ndim != 2 or codebook.ndim != 2:
        raise ValueError("Vectors and codebook must be 2D arrays")

    if vectors.shape[1] != codebook.shape[1]:
        raise ValueError("Vector and codeword dimensions must match")

    if vectors.shape[0] == 0:
        raise ValueError("Vectors cannot be empty")

    if codebook.shape[0] == 0:
        raise ValueError("Codebook cannot be empty")

    if not np.all(np.isfinite(vectors)):
        raise ValueError("Vectors must contain only finite values")

    if not np.all(np.isfinite(codebook)):
        raise ValueError("Codebook must contain only finite values")

    # Broadcasting computes all vector/codeword Manhattan distances at once.
    distances = np.sum(np.abs(vectors[:, np.newaxis, :] - codebook[np.newaxis, :, :]), axis=2)
    return np.argmin(distances, axis=1).astype(np.uint32)


def save_codebook(codebook, block_height, block_width, path):
    """Save codebook header and codewords to a binary file."""
    codebook = np.asarray(codebook, dtype=np.float64)

    if codebook.ndim != 2:
        raise ValueError("Codebook must be a two-dimensional array")
    if codebook.shape[0] == 0 or codebook.shape[1] == 0:
        raise ValueError("Codebook dimensions must be greater than zero")
    if not np.all(np.isfinite(codebook)):
        raise ValueError("Codebook must contain only finite values")

    k, dimension = codebook.shape

    with open(path, "wb") as file:
        # Header: magic, k, dimension, block height, block width
        file.write(struct.pack(
            "<4s4I",
            b"VQCB",
            k,
            dimension,
            block_height,
            block_width
        ))

        file.write(codebook.astype("<f8", copy=False).tobytes(order="C"))


def save_compressed(indices, metadata, path):
    """Save image metadata and codeword indices."""
    indices = np.asarray(indices)

    if indices.ndim != 1:
        raise ValueError("Indices must be a one-dimensional array")
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("Indices must contain integers")

    k = metadata["num_codewords"]
    num_vectors = metadata["num_vectors"]

    if not isinstance(k, (int, np.integer)):
        raise ValueError("num_codewords must be an integer")
    k = int(k)
    if k <= 0:
        raise ValueError("num_codewords must be greater than zero")
    if k > 4294967295:
        raise ValueError("Number of codewords exceeds the 32-bit format")

    if not isinstance(num_vectors, (int, np.integer)):
        raise ValueError("num_vectors must be an integer")
    num_vectors = int(num_vectors)
    if num_vectors < 0:
        raise ValueError("num_vectors cannot be negative")

    if np.any(indices < 0) or np.any(indices >= k):
        raise ValueError("Indices must refer to codewords in the codebook")

    if indices.size != num_vectors:
        raise ValueError("Index count does not match num_vectors")

    max_codeword_index = k - 1
    if max_codeword_index <= np.iinfo(np.uint8).max:
        dtype = np.dtype("<u1")
    elif max_codeword_index <= np.iinfo(np.uint16).max:
        dtype = np.dtype("<u2")
    else:
        dtype = np.dtype("<u4")

    with open(path, "wb") as file:
        # Header: magic + 9 integer metadata values
        file.write(struct.pack(
            "<4s9I",
            b"VQIM",
            metadata["original_shape"][0],
            metadata["original_shape"][1],
            metadata["padded_shape"][0],
            metadata["padded_shape"][1],
            metadata["block_height"],
            metadata["block_width"],
            k,
            num_vectors,
            dtype.itemsize
        ))

        file.write(indices.astype(dtype, copy=False).tobytes(order="C"))


def encode(data, output_dir="."):
    """Encode Member 1 output and save binary files."""
    if not isinstance(data, dict):
        raise TypeError("data must be a dictionary")

    required_keys = (
        "vectors",
        "codebook",
        "original_shape",
        "padded_shape",
        "block_height",
        "block_width",
        "num_vectors",
        "num_codewords",
    )
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        raise KeyError("Missing preprocessing data: {}".format(
            ", ".join(missing_keys)
        ))

    vectors = np.asarray(data["vectors"])
    codebook = np.asarray(data["codebook"])
    if vectors.ndim != 2:
        raise ValueError("vectors must be a 2D array")
    if codebook.ndim != 2:
        raise ValueError("codebook must be a 2D array")
    if codebook.shape[0] != data["num_codewords"]:
        raise ValueError("num_codewords does not match the codebook")
    if vectors.shape[0] != data["num_vectors"]:
        raise ValueError("num_vectors does not match the vectors array")

    for shape_name in ("original_shape", "padded_shape"):
        shape = data[shape_name]
        if not isinstance(shape, (tuple, list)) or len(shape) != 2:
            raise ValueError("{} must contain height and width".format(shape_name))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    indices = assign_vectors(vectors, codebook)

    save_codebook(
        codebook,
        data["block_height"],
        data["block_width"],
        output_dir / "codebook.bin"
    )

    save_compressed(
        indices,
        data,
        output_dir / "compressed.bin"
    )

    return indices
