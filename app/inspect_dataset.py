from datasets import load_dataset

print("Loading BigEarthNet.txt...")

dataset = load_dataset(
    "BIFOLD-BigEarthNetv2-0/BigEarthNet.txt",
    split="all_data",
    streaming=True
)

print("Dataset loaded successfully!")

caption_found = False
bbox_found = False

print("\nSearching for examples...\n")

for sample in dataset:

    if sample["type"] == "captioning" and not caption_found:

        print("=" * 70)
        print("CAPTIONING EXAMPLE")
        print("=" * 70)

        print("\nQuestion:")
        print(sample["input"])

        print("\nAnswer:")
        print(sample["output"])

        print("\nCategory:")
        print(sample["category"])

        caption_found = True

    if sample["type"] == "bounding box" and not bbox_found:

        print("\n" + "=" * 70)
        print("BOUNDING BOX EXAMPLE")
        print("=" * 70)

        print("\nQuestion:")
        print(sample["input"])

        print("\nAnswer:")
        print(sample["output"])

        print("\nCategory:")
        print(sample["category"])

        bbox_found = True

    if caption_found and bbox_found:
        break

print("\nDone!")