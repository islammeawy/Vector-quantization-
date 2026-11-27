from codebook_generator import generate_preprocessing_and_codebook
from compression import encode
from decompression import decode


def main():
    # 1. Generate vectors and codebook
    data = generate_preprocessing_and_codebook(
        image_path="input.png",
        block_height=4,
        block_width=4,
        k=16
    )

    # 2. Encode and save the output
    indices = encode(data, output_dir="output")

    # 3. Decode and calculate compression ratio
    _, ratio = decode(input_dir="output", output_path="output/reconstructed.png")

    # 4. Display a simple result
    print("Encoding and decoding complete.")
    print("Encoded vectors:", len(indices))
    print(f"Compression ratio: {ratio:.2f}:1")
    print("Output directory: output/")


if __name__ == "__main__":
    main()