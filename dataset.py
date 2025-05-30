import json
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoTokenizer
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize


def get_transform():
    return Compose(
        [
            Resize((224, 224), interpolation=Image.BICUBIC),
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


class BLIP2Dataset(torch.utils.data.Dataset):
    def __init__(
        self, json_path, img_folder, processor, max_length=512, max_samples=None
    ):
        self.transform = get_transform()  # ✅ Use correct transform
        self.processor = processor
        self.img_folder = img_folder

        with open(json_path, "r") as f:
            self.data = json.load(f)

        if max_samples is not None:
            self.data = self.data[:max_samples]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]

        item = self.data[idx]
        img_path = f"{self.img_folder}/{item['image_id']}"
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.transform(image)

        inputs = self.processor(
            text=item["context_text"],
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )

        return {
            "pixel_values": pixel_values.squeeze(0),
            "input_ids": inputs["input_ids"].squeeze().squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "image_id": item["image_id"],
        }
