from datasets import load_dataset

print("Loading BigEarthNet.txt...")

dataset = load_dataset(
    "BIFOLD-BigEarthNetv2-0/BigEarthNet.txt",
    split="all_data",
    streaming=True
)

print("Dataset loaded successfully!")

print("\nShowing first 3 examples:\n")

count = 0

for sample in dataset:

    if sample["split"] != "train":
        continue

    print("=" * 60)
    print("Example:", count + 1)

    print("\nInput:")
    print(sample["input"])

    print("\nOutput:")
    print(sample["output"])

    print("\nType:")
    print(sample["type"])

    print("\nCategory:")
    print(sample["category"])

    count += 1

    if count == 3:
        break