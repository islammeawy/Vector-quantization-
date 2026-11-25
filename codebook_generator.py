"""Image preprocessing and LBG codebook generation for Vector Quantization."""

from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


def load_image(image_path):
    """Load an image as an 8-bit grayscale NumPy array.

    Args:
        image_path: Path to an image file.

    Returns:
        A two-dimensional NumPy array containing grayscale pixels.

    Raises:
        FileNotFoundError: If the image path does not exist.
        ValueError: If the file is not a valid supported image.
    """
    path = Path(image_path)

    if not path.is_file():
        raise FileNotFoundError("Image file not found: {}".format(path))

    try:
        with Image.open(path) as image:
            grayscale_image = image.convert("L")
            pixels = np.asarray(grayscale_image, dtype=np.float64)
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("Invalid or unsupported image: {}".format(path)) from error

    if pixels.ndim != 2:
        raise ValueError("Could not convert image to a two-dimensional grayscale array")

    return pixels


def pad_image(image, block_height, block_width):
    """Pad an image with edge pixels so it is divisible by the block size.

    Args:
        image: Two-dimensional NumPy array.
        block_height: Height of each block.
        block_width: Width of each block.

    Returns:
        A tuple containing:
            - the padded NumPy array;
            - the padded shape as (height, width).

    Raises:
        ValueError: If the image or block sizes are invalid.
    """
    _validate_block_size(block_height, block_width)

    image_array = np.asarray(image)
    if image_array.ndim != 2:
        raise ValueError("Image must be a two-dimensional NumPy array")

    original_height, original_width = image_array.shape
    if original_height == 0 or original_width == 0:
        raise ValueError("Image height and width must be greater than zero")

    padded_height = ((original_height + block_height - 1) // block_height) * block_height
    padded_width = ((original_width + block_width - 1) // block_width) * block_width

    pad_height = padded_height - original_height
    pad_width = padded_width - original_width

    padded_image = np.pad(
        image_array,
        pad_width=((0, pad_height), (0, pad_width)),
        mode="edge"
    )

    return padded_image, (padded_height, padded_width)


def extract_vectors(padded_image, block_height, block_width):
    """Split a padded image into flattened blocks in row-major order.

    Args:
        padded_image: Padded two-dimensional NumPy array.
        block_height: Height of each block.
        block_width: Width of each block.

    Returns:
        A NumPy array with shape (number_of_vectors, Bh * Bw).

    Raises:
        ValueError: If dimensions or block sizes are invalid.
    """
    _validate_block_size(block_height, block_width)

    image_array = np.asarray(padded_image)
    if image_array.ndim != 2:
        raise ValueError("Padded image must be a two-dimensional NumPy array")

    padded_height, padded_width = image_array.shape
    if padded_height % block_height != 0 or padded_width % block_width != 0:
        raise ValueError("Padded image dimensions must be divisible by the block size")

    vectors = []

    for row in range(0, padded_height, block_height):
        for column in range(0, padded_width, block_width):
            block = image_array[
                row:row + block_height,
                column:column + block_width
            ]
            vectors.append(block.flatten())

    vectors_array = np.asarray(vectors, dtype=np.float64)

    expected_num_vectors = (padded_height // block_height) * (padded_width // block_width)
    if vectors_array.shape != (expected_num_vectors, block_height * block_width):
        raise ValueError("Extracted vector count or dimension is incorrect")

    return vectors_array


def calculate_distance(vector, codeword):
    """Calculate Manhattan distance between two vectors."""
    vector_array = np.asarray(vector)
    codeword_array = np.asarray(codeword)

    if vector_array.shape != codeword_array.shape:
        raise ValueError("Vector and codeword must have the same dimension")

    return float(np.sum(np.abs(vector_array - codeword_array)))


def generate_codebook(vectors, k, delta=0.01, max_iterations=100, tolerance=0.001):
    """Generate exactly k codewords using the LBG algorithm.

    The algorithm initializes one codeword to the mean of all vectors. It then
    repeatedly splits codewords into Y = X - delta and Z = X + delta until k
    codewords exist. After every split, vectors are assigned to the nearest
    codeword using Manhattan distance and codewords are replaced by centroids.

    Args:
        vectors: NumPy array with shape (number_of_vectors, vector_dimension).
        k: Number of codewords to generate.
        delta: Positive perturbation used for splitting.
        max_iterations: Maximum refinement iterations after each split.
        tolerance: Maximum allowed Manhattan change for convergence.

    Returns:
        A NumPy array with shape (k, vector_dimension).

    Raises:
        ValueError: If inputs are invalid.
    """
    vectors_array = np.asarray(vectors, dtype=np.float64)

    if vectors_array.ndim != 2:
        raise ValueError("Vectors must be a two-dimensional NumPy array")
    if vectors_array.shape[0] == 0:
        raise ValueError("At least one image vector is required")
    if vectors_array.shape[1] == 0:
        raise ValueError("Vector dimension must be greater than zero")
    if not np.all(np.isfinite(vectors_array)):
        raise ValueError("Vectors must contain only finite numeric values")
    if not isinstance(k, (int, np.integer)):
        raise ValueError("k must be an integer")

    k = int(k)
    num_vectors = vectors_array.shape[0]

    if k <= 0:
        raise ValueError("k must be greater than zero")
    if k > num_vectors:
        raise ValueError("k cannot be greater than the number of input vectors")
    if delta <= 0:
        raise ValueError("delta must be greater than zero")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be greater than zero")
    if tolerance < 0:
        raise ValueError("tolerance must be zero or greater")

    initial_codeword = np.mean(vectors_array, axis=0)
    codebook = np.asarray([initial_codeword], dtype=np.float64)

    while codebook.shape[0] < k:
        codewords_needed = k - codebook.shape[0]
        codewords_to_split = min(codebook.shape[0], codewords_needed)

        split_codebook = []

        for index, codeword in enumerate(codebook):
            if index < codewords_to_split:
                split_codebook.append(codeword - delta)
                split_codebook.append(codeword + delta)
            else:
                split_codebook.append(codeword.copy())

        codebook = np.asarray(split_codebook, dtype=np.float64)

        codebook = _refine_codebook(
            vectors_array,
            codebook,
            max_iterations,
            tolerance
        )

    return codebook


def _refine_codebook(vectors, codebook, max_iterations, tolerance):
    """Assign vectors and update codewords until convergence."""
    for _ in range(max_iterations):
        assignments = np.zeros(vectors.shape[0], dtype=np.int64)

        for vector_index, vector in enumerate(vectors):
            nearest_codeword_index = 0
            nearest_distance = calculate_distance(vector, codebook[0])

            for codeword_index in range(1, codebook.shape[0]):
                distance = calculate_distance(vector, codebook[codeword_index])
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_codeword_index = codeword_index

            assignments[vector_index] = nearest_codeword_index

        new_codebook = codebook.copy()

        for codeword_index in range(codebook.shape[0]):
            cluster_vectors = vectors[assignments == codeword_index]

            if cluster_vectors.shape[0] > 0:
                new_codebook[codeword_index] = np.mean(cluster_vectors, axis=0)

        largest_change = 0.0
        for old_codeword, new_codeword in zip(codebook, new_codebook):
            change = calculate_distance(old_codeword, new_codeword)
            if change > largest_change:
                largest_change = change

        codebook = new_codebook

        if largest_change <= tolerance:
            break

    return codebook


def _validate_block_size(block_height, block_width):
    """Validate positive integer block dimensions."""
    if not isinstance(block_height, (int, np.integer)):
        raise ValueError("Block height must be an integer")
    if not isinstance(block_width, (int, np.integer)):
        raise ValueError("Block width must be an integer")
    if block_height <= 0 or block_width <= 0:
        raise ValueError("Block height and block width must be greater than zero")


def generate_preprocessing_and_codebook(image_path, block_height, block_width,
                                        k, delta=0.01, max_iterations=100,
                                        tolerance=0.001):
    """Run Member 1 preprocessing and LBG codebook generation.

    This function returns all values needed for Member 2 encoding and Member 3
    reconstruction. It does not perform encoding or decompression.

    Returns:
        A dictionary containing the codebook, vectors, shapes, block sizes,
        vector count, and codeword count.
    """
    image = load_image(image_path)
    original_shape = (image.shape[0], image.shape[1])

    padded_image, padded_shape = pad_image(image, block_height, block_width)
    vectors = extract_vectors(padded_image, block_height, block_width)
    codebook = generate_codebook(
        vectors,
        k=k,
        delta=delta,
        max_iterations=max_iterations,
        tolerance=tolerance
    )

    return {
        "codebook": codebook,
        "vectors": vectors,
        "original_shape": original_shape,
        "padded_shape": padded_shape,
        "block_height": int(block_height),
        "block_width": int(block_width),
        "num_vectors": int(vectors.shape[0]),
        "num_codewords": int(codebook.shape[0])
    }
