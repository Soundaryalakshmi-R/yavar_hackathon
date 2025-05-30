import os
import json
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# === Configurations ===
IMG_DIR = r"C:\Users\pssan\Desktop\sound\data\img_folder"
META_DIR = r"C:\Users\pssan\Desktop\sound\data\metedata_folder"
OUT_DIR = "preprocessed1"
os.makedirs(OUT_DIR, exist_ok=True)
MAX_SAMPLES = 5000  # Limit processing to 10,000 samples

# === Image Transform ===
image_transform = transforms.Compose(
    [
        transforms.Resize((384, 384)),  # can adjust size depending on model
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # standard ImageNet mean/std
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


# === Helper: Parse Metadata ===
def parse_metadata_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    data = {}
    for line in lines:
        if ":" in line:
            key, value = line.split(":", 1)
            data[key.strip()] = value.strip() if value.strip().lower() != "null" else ""

    # Combine all metadata into a single context string
    context_parts = [
        f"Section Header: {data.get('section_header', '')}",
        f"Above Text: {data.get('above_text', '')}",
        f"Caption: {data.get('caption', '')}",
        f"Below Text: {data.get('below_text', '')}",
        f"Footnote: {data.get('footnote', '')}",
    ]
    context_text = "\n".join(
        [part for part in context_parts if part.strip() and not part.endswith(":")]
    )
    return data, context_text


# === Main Processing ===
dataset = []
sample_count = 0

for filename in tqdm(os.listdir(IMG_DIR)):
    if sample_count >= MAX_SAMPLES:
        print(f"Reached {MAX_SAMPLES} samples. Stopping early.")
        break

    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        base_name = os.path.splitext(filename)[0]
        img_path = os.path.join(IMG_DIR, filename)
        meta_path = os.path.join(META_DIR, base_name + ".txt")

        # Load and transform image
        image = Image.open(img_path).convert("RGB")
        image_tensor = image_transform(image)

        # Parse metadata
        if not os.path.exists(meta_path):
            print(f"Metadata file missing for: {filename}")
            continue

        raw_metadata, context_text = parse_metadata_file(meta_path)

        # Save entry
        dataset.append(
            {
                "image_id": filename,
                "image_tensor": image_tensor,  # not serializable - use in-memory
                "context_text": context_text,
                "raw_metadata": raw_metadata,
            }
        )

        sample_count += 1

# Save simplified version (image paths and context only) for inspection
with open(os.path.join(OUT_DIR, "dataset.json"), "w", encoding="utf-8") as f:
    json.dump(
        [
            {"image_id": item["image_id"], "context_text": item["context_text"]}
            for item in dataset
        ],
        f,
        indent=2,
    )

print(f"Preprocessing complete. {len(dataset)} samples ready.")
