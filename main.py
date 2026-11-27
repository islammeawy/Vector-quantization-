
from codebook_generator import generate_preprocessing_and_codebook
from compression import encode

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

    # 3. Display a simple result
    print("Encoding complete.")
    print("Encoded vectors:", len(indices))
    print("Output directory: output/")

if __name__ == "__main__":
    main()