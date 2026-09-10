from datasets import load_dataset

print("Loading BigEarthNet.txt...")

dataset = load_dataset(
    "BIFOLD-BigEarthNetv2-0/BigEarthNet.txt",
    split="all_data",
    streaming=True
)

print("Dataset loaded successfully!")

print("\nReading one record...\n")

for sample in dataset:

    print("=" * 70)
    print("AVAILABLE FIELDS")
    print("=" * 70)

    for key, value in sample.items():
        print(f"\n{key}:")
        print(value)

    break

print("\nDone!")